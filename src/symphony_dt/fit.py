"""Fit a registered model with CmdStan NUTS, with chain-level checkpoint and resume.

Chains are written to `chains/` and named by chain id. A chain whose CSV holds every requested draw is
complete. With `resume=True`, complete chains are kept and only missing or incomplete chains run,
with the same seed and chain id, so a resumed fit equals an uninterrupted one. Every fit writes
`fit_manifest.json` beside the chains.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from symphony_dt import constants as C
from symphony_dt.data.simulate import initial_values
from symphony_dt.data.stan_data import StanData
from symphony_dt.diagnostics import GateOverride, GateReport, assess, chain_csvs_in_order
from symphony_dt.labels import validation

__all__ = ["FitResult", "chain_is_complete", "fit", "load_fit"]


@validation("TESTED")
def chain_is_complete(csv_file: str | Path, iter_sampling: int) -> bool:
    """Return True when a CmdStan chain CSV holds `iter_sampling` draws and its timing footer.

    Validation: TESTED.
    """
    path = Path(csv_file)
    if not path.exists():
        return False
    draws, footer = 0, False
    header_seen = False
    with path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                if header_seen and "Elapsed Time" in line:
                    footer = True
                continue
            if not line.strip():
                continue
            if not header_seen:
                header_seen = True
                continue
            draws += 1
    return footer and draws == iter_sampling


@validation("TESTED")
@dataclass
class FitResult:
    """The result of `fit`: chain CSV files, settings and the run manifest.

    Validation: TESTED.
    """

    csv_files: list[str]
    output_dir: Path
    settings: dict[str, Any]
    manifest: dict[str, Any] = field(default_factory=dict)
    stan_data_path: Path | None = None

    def draws(self, name: str) -> np.ndarray:
        """Return draws of one column, shaped (draws, chains).

        Validation: TESTED.
        """
        frames = [pd.read_csv(p, comment="#", usecols=[name]) for p in self.csv_files]
        return np.column_stack([f[name].to_numpy(float) for f in frames])

    def columns(self) -> list[str]:
        """Return the column names of the chain CSVs.

        Validation: TESTED.
        """
        with open(self.csv_files[0]) as handle:
            for line in handle:
                if not line.startswith("#"):
                    return line.rstrip("\n").split(",")
        return []

    def gates(self, *, override: GateOverride | None = None) -> GateReport:
        """Evaluate the convergence gates on every chain.

        Validation: VALIDATED.
        """
        return assess(
            self.csv_files, override=override, max_treedepth=int(self.settings.get("max_treedepth", 10))
        )

    def to_inference_data(self, *, globals_only: bool = True) -> Any:
        """Return an ArviZ InferenceData. With `globals_only`, per-participant blocks are dropped.

        Validation: TESTED.
        """
        import arviz as az
        import cmdstanpy

        fit = cmdstanpy.from_csv(self.csv_files)
        idata = az.from_cmdstanpy(posterior=fit)
        if globals_only:
            drop = [v for v in idata.posterior.data_vars if v in ("s0", "b_raw", "eps", "bmat")]
            idata.posterior = idata.posterior.drop_vars(drop)
        return idata

    def save_manifest(self) -> Path:
        """Write the manifest to `fit_manifest.json` in the output folder.

        Validation: TESTED.
        """
        path = self.output_dir / "fit_manifest.json"
        path.write_text(json.dumps(self.manifest, indent=2, default=str) + "\n")
        return path


def _chains_by_id(folder: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in sorted(folder.glob("*.csv")):
        if path.name.startswith("STALE_"):
            continue
        try:
            cid = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        found[cid] = path
    return found


@validation("TESTED")
def fit(
    stan_data: StanData,
    *,
    model: str = "pd_v2",
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    seed: int = C.MASTER_SEED,
    chain_ids: Sequence[int] | None = None,
    threads_per_chain: int = 1,
    parallel_chains: int | None = None,
    inits: Literal["prior", "random"] | dict[str, Any] = "prior",
    output_dir: str | Path | None = None,
    resume: bool = False,
    adapt_delta: float = 0.8,
    max_treedepth: int = 10,
    show_progress: bool = False,
) -> FitResult:
    """Sample the posterior of `model` given `stan_data` and return the chains.

    `inits="prior"` starts every chain at the prior location with per-chain jitter (the study's
    SBC rule); `"random"` uses CmdStan's default uniform(-2, 2) box, as the study's held-out fits
    did. With `resume=True`, complete chains in `output_dir` are kept and only the rest run.

    Validation: TESTED. The sampler settings the study used are the defaults of its own drivers;
    this function runs them, it does not validate them.
    """
    from symphony_dt.governance.manifest import RunManifest
    from symphony_dt.models import compile_model, get_model

    spec = get_model(model)
    ids = list(chain_ids) if chain_ids is not None else list(range(1, chains + 1))
    out = (
        Path(output_dir)
        if output_dir is not None
        else Path.cwd() / "symphony_runs" / f"fit-{int(time.time())}"
    )
    out.mkdir(parents=True, exist_ok=True)
    data_path = stan_data.to_json(out / "stan_data.json")
    exe = compile_model(model, "model")

    settings = {
        "model": model,
        "chains": len(ids),
        "chain_ids": ids,
        "iter_warmup": iter_warmup,
        "iter_sampling": iter_sampling,
        "seed": seed,
        "threads_per_chain": threads_per_chain,
        "adapt_delta": adapt_delta,
        "max_treedepth": max_treedepth,
        "inits": inits if isinstance(inits, str) else "user",
        "resume": resume,
    }
    started = time.time()
    chain_dir = out / "chains"
    chain_dir.mkdir(parents=True, exist_ok=True)
    found = _chains_by_id(chain_dir)
    if found and not resume:
        raise FileExistsError(f"{chain_dir} already holds chains; pass resume=True or a new output_dir")
    complete = {
        cid: path for cid, path in found.items() if cid in ids and chain_is_complete(path, iter_sampling)
    }
    for cid, path in found.items():
        if cid not in complete:
            path.rename(path.with_name(f"STALE_{int(time.time())}_{path.name}"))
    missing = [cid for cid in ids if cid not in complete]
    csv_files = [str(p) for p in complete.values()]
    if missing:
        if inits == "prior":
            theta_like = {"Amat": np.zeros((3, int(stan_data.data["P"])))}
            chain_inits: Any = [
                initial_values(
                    theta_like,
                    np.random.default_rng(seed + cid),
                    n=int(stan_data.data["N"]),
                    n_steps=int(stan_data.data["n_steps"]),
                )
                for cid in missing
            ]
        elif inits == "random":
            chain_inits = None
        else:
            chain_inits = inits
        run = exe.sample(
            data=str(data_path),
            chains=len(missing),
            chain_ids=missing,
            seed=seed,
            parallel_chains=parallel_chains or len(missing),
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            threads_per_chain=threads_per_chain,
            inits=chain_inits,
            adapt_delta=adapt_delta,
            max_treedepth=max_treedepth,
            output_dir=str(chain_dir),
            show_progress=show_progress,
            show_console=False,
        )
        csv_files.extend(str(f) for f in run.runset.csv_files)
    reused = sorted(complete)
    csv_files = chain_csvs_in_order(csv_files)
    result = FitResult(csv_files=csv_files, output_dir=out, settings=settings, stan_data_path=data_path)
    manifest = RunManifest.create(
        kind="fit",
        model=model,
        model_sha256=spec.digests["model"],
        data_sha256=stan_data.sha256(),
        dataset_fingerprint=stan_data.audit.get("dataset_fingerprint"),
        seeds={"sampler": seed},
        settings={**settings, "reused_chains": reused},
        outputs={"csv_files": [Path(f).name for f in csv_files]},
        started=started,
    )
    result.manifest = manifest.to_dict()
    result.save_manifest()
    return result


@validation("TESTED")
def load_fit(output_dir: str | Path) -> FitResult:
    """Load a fit written by `fit` from its output folder.

    Validation: TESTED.
    """
    out = Path(output_dir)
    manifest_path = out / "fit_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    files = [str(p) for p in _chains_by_id(out / "chains").values()]
    if not files:
        raise FileNotFoundError(f"{out} holds no chain CSV files")
    settings = manifest.get("settings", {})
    data_path = out / "stan_data.json"
    return FitResult(
        csv_files=chain_csvs_in_order(files),
        output_dir=out,
        settings=settings,
        manifest=manifest,
        stan_data_path=data_path if data_path.exists() else None,
    )
