from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from symphony_dt.config import AppConfig, load_config
from symphony_dt.cv.splits import fold_train_test_ids, make_stratified_folds
from symphony_dt.inference_backend.backend import detect_backend_from_notebook
from symphony_dt.inference_backend.runner import ScaffoldRunConfig, run_scaffold
from symphony_dt.inference_backend.postprocess import DrawExtractionConfig, extract_draws
from symphony_dt.inference_backend.conditioning_numpyro import NumPyroConditioningConfig, run_conditioning_numpyro
from symphony_dt.inference_backend.forecasting import PosteriorForecastConfig, posterior_predict_instance
from symphony_dt.inference.store import load_drawset, save_drawset
from symphony_dt.io.ppmi import load_ppmi_inputs
from symphony_dt.logging import configure_logging, get_logger
from symphony_dt.forecast.one_step import build_one_step_dataset
from symphony_dt.forecast.horizon import build_horizon_dataset
from symphony_dt.forecast.io import write_table_atomic

console = Console()
app = typer.Typer(add_completion=False)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _select_patient_ids(cfg: AppConfig, clinic: pd.DataFrame, mode: str, shard_id: Optional[int], num_shards: Optional[int]) -> List[int]:
    all_ids = sorted(clinic["patient_id"].unique().tolist())
    if mode == "full":
        return all_ids
    if mode == "demo":
        return all_ids[: int(cfg.demo.n_patients)]
    if mode == "shard":
        if shard_id is None or num_shards is None:
            raise typer.BadParameter("--shard-id and --num-shards are required for mode=shard")
        # deterministic hash sharding implemented elsewhere; here simple stride fallback for NUTS stages
        return all_ids[int(shard_id) :: int(num_shards)]
    raise typer.BadParameter("mode must be one of: demo|full|shard")


@app.command()
def fit_scaffold(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    fold_id: int = typer.Option(0, "--fold-id"),
    out_dir: Path = typer.Option(..., "--out-dir", help="Directory to write scaffold drawset (per fold)"),
    seed: int = typer.Option(0, "--seed"),
    num_warmup: int = typer.Option(1000, "--num-warmup"),
    num_samples: int = typer.Option(1000, "--num-samples"),
    num_chains: int = typer.Option(4, "--num-chains"),
) -> None:
    """
    Fit scaffold posterior (NUTS) on the training fold and write drawset to out_dir.

    This does NOT change the baseline pipeline; it is an opt-in workflow stage.
    """
    cfg = load_config(config)
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)

    ppmi = load_ppmi_inputs(cfg)
    baseline = ppmi.baseline.copy()
    clinic = ppmi.clinic.copy()

    patient_ids = sorted(clinic["patient_id"].unique().tolist())
    folds = make_stratified_folds(baseline=baseline, patient_ids=patient_ids, n_folds=cfg.cv.n_folds, seed=cfg.run.seed)
    train_ids, _ = fold_train_test_ids(folds, fold_id=fold_id)

    # Run scaffold (NumPyro)
    run_cfg = ScaffoldRunConfig(
        backend=detect_backend_from_notebook(Path("")).backend if False else __import__("symphony_dt.inference_backend.backend").inference_backend.backend.InferenceBackend.NUMPYRO,  # type: ignore
        seed=int(seed),
        P=0,
        numpyro=__import__("symphony_dt.inference_backend.numpyro_impl").inference_backend.numpyro_impl.NumPyroRunConfig(  # type: ignore
            num_warmup=int(num_warmup),
            num_samples=int(num_samples),
            num_chains=int(num_chains),
            seed=int(seed),
        ),
    )

    out_dir = _ensure_dir(out_dir)
    drawset = run_scaffold(
        clinic=clinic,
        baseline=baseline,
        patient_ids_train=train_ids,
        output_dir=out_dir,
        cfg=run_cfg,
    )

    # Extract L=200 draws deterministically for downstream stages
    extracted = extract_draws(drawset, DrawExtractionConfig(L_target=200, seed=cfg.run.seed))
    save_drawset(out_dir, extracted, overwrite=True)

    log.info("Scaffold fit complete. Drawset written to %s", out_dir)
    console.print(f"[bold green]OK[/bold green] Scaffold drawset written to: {out_dir}")


@app.command()
def condition(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    drawset_dir: Path = typer.Option(..., "--drawset-dir", exists=True, file_okay=False),
    out_dir: Path = typer.Option(..., "--out-dir"),
    mode: str = typer.Option("demo", "--mode", help="demo|full|shard"),
    shard_id: Optional[int] = typer.Option(None, "--shard-id"),
    num_shards: Optional[int] = typer.Option(None, "--num-shards"),
    max_patients: int = typer.Option(0, "--max-patients", help="0 means no limit"),
    visit_index: int = typer.Option(0, "--visit-index", help="Condition at this visit index (0-based)"),
    l_cond: int = typer.Option(10, "--l-cond"),
    seed: int = typer.Option(0, "--seed"),
    num_warmup: int = typer.Option(250, "--num-warmup"),
    num_samples: int = typer.Option(250, "--num-samples"),
    num_chains: int = typer.Option(4, "--num-chains"),
) -> None:
    """
    Run per-patient conditioning (NUTS) for selected global draws and save conditional artifacts.

    HPC usage: run this as a job array by patient shards.
    """
    cfg = load_config(config)
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)

    ppmi = load_ppmi_inputs(cfg)
    clinic = ppmi.clinic.copy()

    patient_ids = _select_patient_ids(cfg, clinic, mode=mode, shard_id=shard_id, num_shards=num_shards)
    if max_patients and max_patients > 0:
        patient_ids = patient_ids[: int(max_patients)]

    out_dir = _ensure_dir(out_dir)
    drawset = load_drawset(drawset_dir)
    # Choose first l_cond draws by default; replace with selected indices when you wire L_cond selector
    draw_indices = list(range(int(l_cond)))

    # Global params per draw index (simple dict per draw)
    for pid in patient_ids:
        pdf = clinic[clinic["patient_id"] == pid].copy()
        if pdf.empty:
            continue
        for ell in draw_indices:
            global_params = {}
            for k, arr in drawset.arrays.items():
                if arr.ndim >= 1 and arr.shape[0] > ell:
                    global_params[k] = np.asarray(arr[ell]).tolist() if arr.ndim <= 2 else np.asarray(arr[ell]).tolist()

            run_conditioning_numpyro(
                patient_id=int(pid),
                visit_index=int(visit_index),
                global_draw_index=int(ell),
                patient_clinic=pdf,
                global_params=global_params,
                out_dir=out_dir,
                cfg=NumPyroConditioningConfig(
                    num_warmup=int(num_warmup),
                    num_samples=int(num_samples),
                    num_chains=int(num_chains),
                    seed=int(seed),
                ),
            )

    log.info("Conditioning complete. Wrote artifacts to %s", out_dir)
    console.print(f"[bold green]OK[/bold green] Conditioning artifacts written to: {out_dir}")


@app.command()
def forecast(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    drawset_dir: Path = typer.Option(..., "--drawset-dir", exists=True, file_okay=False),
    conditional_dir: Path = typer.Option(..., "--conditional-dir", exists=True, file_okay=False),
    out_results_dir: Path = typer.Option(..., "--out-results-dir"),
    mode: str = typer.Option("demo", "--mode", help="demo|full|shard"),
    shard_id: Optional[int] = typer.Option(None, "--shard-id"),
    num_shards: Optional[int] = typer.Option(None, "--num-shards"),
    l_cond: int = typer.Option(10, "--l-cond"),
    tau_conf: float = typer.Option(1.0, "--tau-conf"),
    seed: int = typer.Option(0, "--seed"),
    n_draws_predictive: int = typer.Option(2000, "--n-draws-predictive"),
) -> None:
    """
    Generate posterior-simulation forecasts using:
      - global drawset
      - per-patient conditional artifacts

    Writes standardized forecast tables into out_results_dir.
    """
    cfg = load_config(config)
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)

    ppmi = load_ppmi_inputs(cfg)
    clinic = ppmi.clinic.copy()

    patient_ids = _select_patient_ids(cfg, clinic, mode=mode, shard_id=shard_id, num_shards=num_shards)
    out_results_dir = _ensure_dir(out_results_dir)

    # Build instances and then overwrite preds using posterior simulation
    inst1 = build_one_step_dataset(clinic[clinic["patient_id"].isin(patient_ids)].copy())
    if inst1.empty:
        console.print("[yellow]No one-step instances for this shard.[/yellow]")
        return

    # Use selected draw indices 0..l_cond-1 by default (replace with saved indices from selector when wired)
    selected_draw_indices = list(range(int(l_cond)))

    rows = []
    for _, r in inst1.iterrows():
        pid = int(r["patient_id"])
        t_anchor = float(r["t_anchor_days"])
        t_target = float(r["t_target_days"])
        y_anchor = np.array([
            float(r["y_anchor_updrs3"]) if pd.notna(r["y_anchor_updrs3"]) else np.nan,
            float(r["y_anchor_moca"]) if pd.notna(r["y_anchor_moca"]) else np.nan,
            float(r["y_anchor_scopa"]) if pd.notna(r["y_anchor_scopa"]) else np.nan,
        ])
        missing_anchor = (not np.isfinite(y_anchor[0]), not np.isfinite(y_anchor[1]), not np.isfinite(y_anchor[2]))

        # anchor_visit_id is visit_id; anchor_visit_index not stored in baseline tables, so we use 0 (first visit) as initial wiring.
        # For full fidelity, wire anchor_visit_index per patient from sorted visits (future patch).
        pred, reasons = posterior_predict_instance(
            patient_id=pid,
            anchor_visit_index=0,
            t_anchor=t_anchor,
            t_target=t_target,
            y_anchor=y_anchor,
            missing_anchor=missing_anchor,
            n_visits_total=2,
            drawset_dir=drawset_dir,
            conditional_root=conditional_dir,
            selected_draw_indices=selected_draw_indices,
            l_cond=int(l_cond),
            tau_conf=float(tau_conf),
            cfg=PosteriorForecastConfig(n_draws_predictive=int(n_draws_predictive), seed=int(seed)),
        )

        if pred is None:
            continue

        out = dict(r.to_dict())
        out.update(pred)
        out["engine"] = "posterior_sim"
        rows.append(out)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        console.print("[yellow]No posterior forecasts written (all suppressed or missing artifacts).[/yellow]")
        return

    write_table_atomic(out_results_dir / "forecast_one_step_predictions.csv", out_df)
    log.info("Posterior forecasts written to %s", out_results_dir)
    console.print(f"[bold green]OK[/bold green] Posterior forecasts written to: {out_results_dir}")
