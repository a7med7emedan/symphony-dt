"""Golden tests: the package must reproduce the code of record exactly on the synthetic cases.

The expected outputs in tests/data/golden/ were produced by running the study's own code
(`stan_data.py`, `simulate.py`, `sbc.py`, `score_fold.py`, `pool_folds.py`) read only, by
`tests/golden/generate.py`. These tests need no study file and run in CI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from symphony_dt.calibration import chi_squared_uniformity, sbc_rank, thin_for_rank
from symphony_dt.data import build_stan_data, flatten_theta, initial_values, simulate_replicate
from symphony_dt.diagnostics import retained_chains
from symphony_dt.evaluate import ScoringSpec, freeze_spec, pool_folds, score_fold
from tests.golden import cases

pytestmark = pytest.mark.golden
FIX = cases.FIXTURES


def load(name: str):
    return json.loads((FIX / name).read_text())


def _same_platform() -> bool:
    import platform

    prov = load("PROVENANCE.json")
    return (
        prov.get("platform") == f"{platform.system()}-{platform.machine()}"
        and prov.get("numpy") == np.__version__
    )


def assert_same(ours, theirs, path="root"):
    """Exact on the platform that produced the fixtures; within 1e-12 relative elsewhere.

    Floating-point library differences between operating systems can move the last bit of exp or log.
    The exact comparison is the golden criterion; the tolerance only absorbs platform arithmetic.
    """
    if _same_platform():
        assert json.dumps(ours, sort_keys=True) == json.dumps(theirs, sort_keys=True), path
        return
    if isinstance(theirs, dict):
        assert set(ours) == set(theirs), path
        for key in theirs:
            assert_same(ours[key], theirs[key], f"{path}.{key}")
    elif isinstance(theirs, list):
        assert len(ours) == len(theirs), path
        for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
            assert_same(a, b, f"{path}[{i}]")
    elif isinstance(theirs, float) and isinstance(ours, float | int):
        assert ours == pytest.approx(theirs, rel=1e-12, abs=1e-12, nan_ok=True), path
    else:
        assert ours == theirs, path


def test_fixture_digests_match_provenance():
    import hashlib

    prov = load("PROVENANCE.json")
    for name, digest in prov["fixtures"].items():
        assert hashlib.sha256((FIX / name).read_bytes()).hexdigest() == digest, name
    assert prov["analysis_src"]["stan_data.py"].startswith("06581941")
    assert prov["analysis_src"]["simulate.py"].startswith("602b5487")
    assert prov["analysis_src"]["sbc.py"].startswith("f05d98b0")


def test_stan_data_builder_equals_study_builder():
    ds, _, std = cases.stan_data_case()
    ours = build_stan_data(ds, standardisation=std, pathway=cases.PATHWAY, grainsize=64).data
    expected = load("stan_data_build.json")
    assert_same(json.loads(json.dumps(ours)), expected)


def test_simulator_equals_study_simulator_draw_for_draw():
    patterns, theta, seed, n = cases.simulate_case()
    latents: dict = {}
    ours = simulate_replicate(
        patterns,
        n,
        theta,
        np.random.default_rng(seed),
        censored=(0, 1, 0),
        state_dependent_sigma=1,
        ledd_in_mu_U=1,
        sigma_xbar=(0.53, 0.54, 0.55),
        latents=latents,
    )
    expected = load("simulate_replicate.json")
    assert_same(json.loads(json.dumps(ours)), expected["data"])
    for key, value in latents.items():
        assert_same(np.asarray(value).tolist(), expected["latents"][key], key)
    assert_same(flatten_theta(theta), expected["flat_theta"])
    assert_same(initial_values(theta, np.random.default_rng(7), n=3, n_steps=4), expected["inits"])


def test_sbc_functions_equal_study_functions():
    expected = load("sbc_functions.json")
    for name, lp in cases.lp_case().items():
        ours = retained_chains(lp)
        theirs = expected["retained"][name]
        for key in (
            "keep_index",
            "retained_chains",
            "excluded_chains",
            "n_retained",
            "cut_lp",
            "sd_within",
            "per_chain_mean_lp",
            "trapping_rate",
            "meets_min_retained",
        ):
            assert ours[key] == theirs[key], (name, key)
    rng = np.random.default_rng(505)
    ranks = rng.integers(0, 200, size=150)
    draws = rng.normal(size=1000)
    assert_same(chi_squared_uniformity(ranks, 199, alpha=0.01 / 55), expected["chi2"])
    assert_same(chi_squared_uniformity(ranks, 199), expected["chi2_default"])
    assert_same(thin_for_rank(draws).tolist(), expected["thin"])
    assert sbc_rank(draws, 0.3) == expected["rank"]


@pytest.fixture(scope="module")
def fold_scores():
    spec = ScoringSpec.default()
    frozen = freeze_spec(spec)
    out = {}
    for fold in (1, 2, 3):
        sp, tests, age, draws = cases.fold_case(fold)
        out[fold] = score_fold(
            sp.fit.data, sp.targets.data, sp.meta, draws, tests, spec=spec, frozen=frozen, age=age, fold=fold
        )
    return out


@pytest.mark.parametrize("fold", [1, 2, 3])
def test_score_fold_equals_code_of_record(fold_scores, fold):
    ours = fold_scores[fold]
    if _same_platform():
        assert ours.to_json() == (FIX / f"fold{fold}_scores.json").read_text()
    assert_same(json.loads(ours.to_json()), load(f"fold{fold}_scores.json"))
    theirs = pd.read_csv(FIX / f"fold{fold}_target_scores.csv")
    mine = pd.read_csv(pd.io.common.StringIO(ours.target_scores.to_csv(index=False)))
    pd.testing.assert_frame_equal(mine, theirs, check_exact=_same_platform(), rtol=1e-10)


def test_pool_folds_equals_code_of_record(fold_scores):
    spec = ScoringSpec.default()
    folds = []
    for fold in (1, 2, 3):
        ts = pd.read_csv(pd.io.common.StringIO(fold_scores[fold].target_scores.to_csv(index=False)))
        folds.append((ts, {"converged": True, "max_rhat": 1.004, "min_ess_bulk": 512.0}))
    pooled = pool_folds(folds, spec=spec, frozen=freeze_spec(spec))
    if _same_platform():
        assert pooled.to_json() == (FIX / "pooled_scores.json").read_text()
    assert_same(json.loads(pooled.to_json()), load("pooled_scores.json"))


ANALYSIS = Path(
    os.environ.get("SYMPHONY_V2_SRC", Path(__file__).resolve().parents[3] / "symphony_v2" / "src")
)
RECORD = Path(
    os.environ.get(
        "SYMPHONY_CODE_OF_RECORD",
        Path(__file__).resolve().parents[3]
        / "SYMPHONY_v2_PAPER"
        / "04_CODE_OF_RECORD"
        / "scripts"
        / "five_fold",
    )
)


@pytest.mark.skipif(
    not (ANALYSIS.exists() and RECORD.exists()), reason="the study code is not on this machine"
)
def test_fixtures_still_match_a_fresh_run_of_the_code_of_record(tmp_path):
    from tests.golden.generate import run

    prov = run(ANALYSIS, RECORD, tmp_path)
    assert prov["analysis_src"] == load("PROVENANCE.json")["analysis_src"]
    assert prov["code_of_record"] == load("PROVENANCE.json")["code_of_record"]
    if _same_platform():
        assert prov["fixtures"] == load("PROVENANCE.json")["fixtures"]
