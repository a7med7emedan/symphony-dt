"""Run the SYMPHONY v2 code of record on the synthetic golden cases and freeze its outputs.

Usage (from the repository root, with the read-only analysis repo and paper folder beside it):

    python -m tests.golden.generate [--analysis ../symphony_v2/src] [--record ../SYMPHONY_v2_PAPER/...]

The code of record is imported or run as it is, read only. The study modules it imports for paths
and the covariate manifest are replaced by the stubs in `tests/golden/stubs/`, which supply
constants and synthetic file locations only. Outputs go to `tests/data/golden/`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from tests.golden import cases

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_ANALYSIS = REPO.parent / "symphony_v2" / "src"
DEFAULT_RECORD = REPO.parent / "SYMPHONY_v2_PAPER" / "04_CODE_OF_RECORD" / "scripts" / "five_fold"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_study_modules(analysis: Path, root: Path) -> dict:
    """Import stan_data, simulate and sbc from the analysis repo with the stubs in front."""
    os.environ["SDT_GOLDEN_ROOT"] = str(root)
    for name in ("paths", "fold_guard", "covariate_manifest", "stan_data", "simulate", "sbc"):
        sys.modules.pop(name, None)
    sys.path[:0] = [str(HERE / "stubs"), str(analysis)]
    try:
        return {name: importlib.import_module(name) for name in ("stan_data", "simulate", "sbc")}
    finally:
        del sys.path[:2]


def run(analysis: Path, record: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "analysis_src": {f: sha(analysis / f) for f in ("stan_data.py", "simulate.py", "sbc.py")},
        "code_of_record": {f: sha(record / f) for f in ("score_fold.py", "pool_folds.py")},
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds, folds, std = cases.stan_data_case()
        cases.write_stan_data_inputs(root, ds, folds, std)
        mods = load_study_modules(analysis, root)

        stan, _ = mods["stan_data"].build(1)
        (out / "stan_data_build.json").write_text(json.dumps(stan, sort_keys=True) + "\n")

        patterns, theta, seed, n = cases.simulate_case()
        study_patterns = [
            {"dt_days": p["dt_days"], "dt_years": p["dt_years"], "observed": p["observed"]} for p in patterns
        ]
        latents: dict = {}
        sim = mods["simulate"].simulate_replicate(
            study_patterns,
            n,
            theta,
            np.random.default_rng(seed),
            censored=(0, 1, 0),
            state_dependent_sigma=1,
            ledd_in_mu_U=1,
            sigma_xbar=(0.53, 0.54, 0.55),
            latents=latents,
        )
        payload = {
            "data": sim,
            "latents": {k: np.asarray(v).tolist() for k, v in latents.items()},
            "flat_theta": mods["simulate"].flatten_theta(theta),
            "inits": mods["simulate"].initial_values(theta, np.random.default_rng(7), n=3, n_steps=4),
        }
        (out / "simulate_replicate.json").write_text(json.dumps(payload, sort_keys=True) + "\n")

        sbc = mods["sbc"]
        rng = np.random.default_rng(505)
        ranks = rng.integers(0, 200, size=150)
        draws = rng.normal(size=1000)
        sbc_out = {
            "retained": {k: sbc.retained_chains(v) for k, v in cases.lp_case().items()},
            "chi2": sbc.chi_squared_uniformity(ranks, 199, alpha=sbc.chi2_alpha_per_parameter(55)),
            "chi2_default": sbc.chi_squared_uniformity(ranks, 199),
            "alpha_55": sbc.chi2_alpha_per_parameter(55),
            "thin": sbc._thin(draws).tolist(),
            "rank": sbc._rank(draws, 0.3),
        }
        (out / "sbc_functions.json").write_text(json.dumps(sbc_out, sort_keys=True) + "\n")

        roots = []
        for fold in (1, 2, 3):
            sp, tests, age, gq = cases.fold_case(fold)
            fold_root = cases.write_fold_dir(root / f"fold_{fold:02d}_ho", sp, tests, age, gq)
            subprocess.run(
                [
                    sys.executable,
                    str(record / "score_fold.py"),
                    "--fold",
                    str(fold),
                    "--test",
                    str(fold_root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            (out / f"fold{fold}_scores.json").write_bytes((fold_root / "score" / "scores.json").read_bytes())
            ts = (fold_root / "score" / "target_scores.csv").read_text()
            (fold_root / "fit_record_cv.json").write_text(
                json.dumps({"converged": True, "max_rhat": 1.004, "min_ess_bulk": 512.0})
            )
            (out / f"fold{fold}_target_scores.csv").write_text(ts)
            roots.append(fold_root)
        pooled_dir = root / "pooled"
        subprocess.run(
            [
                sys.executable,
                str(record / "pool_folds.py"),
                "--roots",
                *map(str, roots),
                "--out",
                str(pooled_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        (out / "pooled_scores.json").write_bytes((pooled_dir / "pooled_scores.json").read_bytes())

    provenance["fixtures"] = {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != "PROVENANCE.json"}
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    ap.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    ap.add_argument("--out", type=Path, default=cases.FIXTURES)
    args = ap.parse_args()
    prov = run(args.analysis, args.record, args.out)
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
