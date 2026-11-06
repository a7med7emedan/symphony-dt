from __future__ import annotations

import math
from collections import Counter
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
from symphony_dt.results import export_json
from symphony_dt.trust.tau_conf import TauConfSelectionConfig, load_tau_conf_json, select_tau_conf, selection_report

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


def _one_step_instances(clinic: pd.DataFrame, patient_ids: List[int], visit_index: int) -> pd.DataFrame:
    """One-step instances anchored at `visit_index`, with anchor_visit_index and n_visits_total attached."""
    inst1 = build_one_step_dataset(clinic[clinic["patient_id"].isin(patient_ids)].copy())

    # Map anchor_visit_id -> anchor_visit_index (must match conditioning visit_index definition)
    v = clinic[clinic["patient_id"].isin(patient_ids)][["patient_id", "visit_id", "t_days_since_baseline"]].copy()
    v["t_days_since_baseline"] = pd.to_numeric(v["t_days_since_baseline"], errors="coerce")
    v = v.sort_values(["patient_id", "t_days_since_baseline", "visit_id"]).reset_index(drop=True)
    v["visit_index"] = v.groupby("patient_id").cumcount()

    n_vis = v.groupby("patient_id")["visit_index"].max().add(1).rename("n_visits_total").reset_index()
    anchor_map = v[["patient_id", "visit_id", "visit_index"]].rename(
        columns={"visit_id": "anchor_visit_id", "visit_index": "anchor_visit_index"}
    )

    inst1 = inst1.merge(anchor_map, on=["patient_id", "anchor_visit_id"], how="left").merge(n_vis, on="patient_id", how="left")
    inst1 = inst1[inst1["anchor_visit_index"].notna()].copy()
    inst1["anchor_visit_index"] = inst1["anchor_visit_index"].astype(int)
    inst1["n_visits_total"] = pd.to_numeric(inst1["n_visits_total"], errors="coerce").fillna(0).astype(int)

    # Forecast only anchors for the conditioned visit_index
    return inst1[inst1["anchor_visit_index"] == int(visit_index)].copy()


def _anchor_values(r: pd.Series) -> np.ndarray:
    return np.array([
        float(r["y_anchor_updrs3"]) if pd.notna(r["y_anchor_updrs3"]) else np.nan,
        float(r["y_anchor_moca"]) if pd.notna(r["y_anchor_moca"]) else np.nan,
        float(r["y_anchor_scopa"]) if pd.notna(r["y_anchor_scopa"]) else np.nan,
    ])


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

            try:
                try:
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
                except ValueError as e:
                    log.warning('Skipping patient_id=%s visit_index=%s draw=%s: %s', pid, visit_index, ell, e)
                    continue
            except ValueError as e:
                if "visit_index out of range" in str(e):
                    log.warning("Skipping patient %s for visit_index=%s: %s", pid, visit_index, e)
                    continue
                raise

    log.info("Conditioning complete. Wrote artifacts to %s", out_dir)
    console.print(f"[bold green]OK[/bold green] Conditioning artifacts written to: {out_dir}")


def _resolve_tau_conf(tau_conf: Optional[float], tau_conf_json: Optional[Path], *, visit_index: int, l_cond: int) -> float:
    if tau_conf is not None and tau_conf_json is not None:
        raise typer.BadParameter("pass either --tau-conf or --tau-conf-json, not both")
    if tau_conf_json is not None:
        try:
            data = load_tau_conf_json(tau_conf_json)
        except ValueError as e:
            raise typer.BadParameter(str(e)) from e
        for key, used in (("visit_index", visit_index), ("l_cond", l_cond)):
            if key in data and data[key] != used:
                console.print(f"[yellow]tau_conf was selected with {key}={data[key]}, this forecast uses {used}.[/yellow]")
        if data.get("feasible") is False:
            console.print("[yellow]tau_conf came from the fallback rule: no grid value kept enough anchors.[/yellow]")
        return float(data["tau_conf"])
    if tau_conf is None:
        console.print("[yellow]No --tau-conf given: using 1.0, which leaves gating rule 3 inactive. "
                      "Run 'symphony-dt nuts select-tau-conf' to choose it on the training fold.[/yellow]")
        return 1.0
    return float(tau_conf)


@app.command("select-tau-conf")
def select_tau_conf_cmd(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    drawset_dir: Path = typer.Option(..., "--drawset-dir", exists=True, file_okay=False),
    conditional_dir: Path = typer.Option(..., "--conditional-dir", exists=True, file_okay=False),
    out_json: Path = typer.Option(..., "--out-json", dir_okay=False, help="Where to write tau_conf and grid diagnostics"),
    fold_id: int = typer.Option(0, "--fold-id", help="Training fold, as for fit-scaffold"),
    visit_index: int = typer.Option(0, "--visit-index", help="Anchor visit index (must match conditioning)"),
    l_cond: int = typer.Option(10, "--l-cond"),
    seed: int = typer.Option(0, "--seed"),
    n_draws_predictive: int = typer.Option(2000, "--n-draws-predictive"),
) -> None:
    """
    Choose tau_conf for gating rule 3 on the training fold and write it, with grid diagnostics, to JSON.

    For each training-fold anchor at --visit-index, Delta and the one-step central 90% interval
    come from the conditioning outputs, with rule 3 switched off. Anchors suppressed by rules 1, 2
    or 4 are left out, because no tau_conf would show them. The chosen value is the grid point whose
    coverage is closest to 0.90 while keeping at least half of the anchors.
    Pass the file to 'nuts forecast --tau-conf-json'.
    """
    cfg = load_config(config)
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)

    ppmi = load_ppmi_inputs(cfg)
    baseline = ppmi.baseline.copy()
    clinic = ppmi.clinic.copy()

    all_ids = sorted(clinic["patient_id"].unique().tolist())
    folds = make_stratified_folds(baseline=baseline, patient_ids=all_ids, n_folds=cfg.cv.n_folds, seed=cfg.run.seed)
    train_ids, _ = fold_train_test_ids(folds, fold_id=fold_id)

    inst = _one_step_instances(clinic, list(train_ids), visit_index)
    selected_draw_indices = list(range(int(l_cond)))
    endpoints = ("updrs3", "moca", "scopa")

    delta: List[float] = []
    y_true: List[List[float]] = []
    lo: List[List[float]] = []
    hi: List[List[float]] = []
    suppressed = Counter()
    for _, r in inst.iterrows():
        y_anchor = _anchor_values(r)
        pred, reasons = posterior_predict_instance(
            patient_id=int(r["patient_id"]),
            anchor_visit_index=int(r["anchor_visit_index"]),
            t_anchor=float(r["t_anchor_days"]),
            t_target=float(r["t_target_days"]),
            y_anchor=y_anchor,
            missing_anchor=tuple(not np.isfinite(v) for v in y_anchor),
            n_visits_total=int(r["n_visits_total"]),
            drawset_dir=drawset_dir,
            conditional_root=conditional_dir,
            selected_draw_indices=selected_draw_indices,
            l_cond=int(l_cond),
            tau_conf=math.inf,  # rule 3 off: Delta is what is being thresholded
            cfg=PosteriorForecastConfig(n_draws_predictive=int(n_draws_predictive), seed=int(seed)),
        )
        if pred is None:
            for rule in reasons or {"unknown": ""}:
                suppressed[str(rule)] += 1
            continue
        delta.append(float(pred["delta_iqr"]))
        y_true.append([float(r[f"y_target_{e}"]) if pd.notna(r[f"y_target_{e}"]) else np.nan for e in endpoints])
        lo.append([float(pred[f"pred_p05_{e}"]) for e in endpoints])
        hi.append([float(pred[f"pred_p95_{e}"]) for e in endpoints])

    if not delta:
        console.print(
            f"[bold red]No eligible anchors[/bold red] at visit_index={visit_index} in fold {fold_id} "
            f"({len(inst)} considered, suppressed: {dict(suppressed)}). Condition the training-fold patients first."
        )
        raise typer.Exit(code=1)

    sel_cfg = TauConfSelectionConfig()
    res = select_tau_conf(np.asarray(delta), np.asarray(y_true), np.asarray(lo), np.asarray(hi), cfg=sel_cfg)
    report = selection_report(
        res,
        sel_cfg,
        fold_id=int(fold_id),
        visit_index=int(visit_index),
        l_cond=int(l_cond),
        seed=int(seed),
        n_draws_predictive=int(n_draws_predictive),
        n_training_patients=len(train_ids),
        n_anchors_considered=int(len(inst)),
        n_anchors_eligible=len(delta),
        suppressed_by_rule=dict(sorted(suppressed.items())),
    )
    export_json(out_json, report)
    log.info("tau_conf=%.2f (coverage %s, fraction kept %.2f) written to %s",
             res.tau_conf, report["achieved_coverage"], res.fraction_kept, out_json)
    status = "OK" if res.feasible else "FALLBACK"
    console.print(f"[bold green]{status}[/bold green] tau_conf={res.tau_conf:.2f} written to: {out_json}")


@app.command()
def forecast(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    drawset_dir: Path = typer.Option(..., "--drawset-dir", exists=True, file_okay=False),
    conditional_dir: Path = typer.Option(..., "--conditional-dir", exists=True, file_okay=False),
    out_results_dir: Path = typer.Option(..., "--out-results-dir"),
    mode: str = typer.Option("demo", "--mode", help="demo|full|shard"),
    visit_index: int = typer.Option(0, "--visit-index", help="Only forecast anchors at this visit_index (must match conditioning)"),
    shard_id: Optional[int] = typer.Option(None, "--shard-id"),
    num_shards: Optional[int] = typer.Option(None, "--num-shards"),
    l_cond: int = typer.Option(10, "--l-cond"),
    tau_conf: Optional[float] = typer.Option(
        None, "--tau-conf", help="Gating threshold for rule 3. Without this or --tau-conf-json, 1.0 (rule 3 inactive)."
    ),
    tau_conf_json: Optional[Path] = typer.Option(
        None, "--tau-conf-json", exists=True, dir_okay=False, help="Read tau_conf from a select-tau-conf output file."
    ),
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
    tau_conf = _resolve_tau_conf(tau_conf, tau_conf_json, visit_index=visit_index, l_cond=l_cond)

    ppmi = load_ppmi_inputs(cfg)
    clinic = ppmi.clinic.copy()

    patient_ids = _select_patient_ids(cfg, clinic, mode=mode, shard_id=shard_id, num_shards=num_shards)
    out_results_dir = _ensure_dir(out_results_dir)

    # Build instances and then overwrite preds using posterior simulation
    inst1 = _one_step_instances(clinic, patient_ids, visit_index)
    if inst1.empty:
        console.print("[yellow]No one-step instances for this shard.[/yellow]")
        return

    # Use selected draw indices 0..l_cond-1 by default (replace with saved indices from selector when wired)
    selected_draw_indices = list(range(int(l_cond)))

    rows = []
    reason_counts = Counter()
    for _, r in inst1.iterrows():
        pid = int(r["patient_id"])
        t_anchor = float(r["t_anchor_days"])
        t_target = float(r["t_target_days"])
        y_anchor = _anchor_values(r)
        missing_anchor = (not np.isfinite(y_anchor[0]), not np.isfinite(y_anchor[1]), not np.isfinite(y_anchor[2]))

        # anchor_visit_id is visit_id; anchor_visit_index not stored in baseline tables, so we use 0 (first visit) as initial wiring.
        # For full fidelity, wire anchor_visit_index per patient from sorted visits (future patch).
        pred, reasons = posterior_predict_instance(
            patient_id=pid,
            anchor_visit_index=int(r['anchor_visit_index']),
            t_anchor=t_anchor,
            t_target=t_target,
            y_anchor=y_anchor,
            missing_anchor=missing_anchor,
            n_visits_total=int(r['n_visits_total']),
            drawset_dir=drawset_dir,
            conditional_root=conditional_dir,
            selected_draw_indices=selected_draw_indices,
            l_cond=int(l_cond),
            tau_conf=float(tau_conf),
            cfg=PosteriorForecastConfig(n_draws_predictive=int(n_draws_predictive), seed=int(seed)),
        )

        if pred is None:
            for rr in (reasons or []):
                reason_counts[str(rr)] += 1
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
