from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console

from symphony_dt.aggregate.all import aggregate_all_results
from symphony_dt.clinical.events import (
    build_clinical_event_pairs,
    build_clinical_event_rates_by_state,
    build_clinical_event_rates_by_trajectory,
    build_clinical_event_rates_by_trust,
    build_clinician_summary_risk_table,
    build_patient_panel,
    build_personalization_diagnostics,
)
from symphony_dt.config import AppConfig, load_config
from symphony_dt.cv.splits import fold_train_test_ids, make_stratified_folds
from symphony_dt.evaluation.stratified import (
    StratifyConfig,
    evaluation_stratified_coverage_table,
    evaluation_stratified_mae_table,
)
from symphony_dt.evaluation.tables import evaluation_coverage_table, evaluation_mae_table
from symphony_dt.forecast.derive import (
    build_horizon_states,
    build_risk_curves,
    horizon_long_to_uq,
    horizon_long_to_uq_refined,
    horizon_long_to_wide,
)
from symphony_dt.forecast.horizon import build_horizon_dataset, predict_horizon_locf
from symphony_dt.forecast.io import write_table_atomic
from symphony_dt.forecast.one_step import build_one_step_dataset, predict_one_step_locf
from symphony_dt.forecast.schema import validate_forecast_one_step
from symphony_dt.forecast.schema_horizon import validate_forecast_horizon_long
from symphony_dt.forecast.schema_extra import (
    validate_forecast_horizon_long_uq,
    validate_forecast_horizon_long_uq_refined,
    validate_forecast_horizon_states,
    validate_forecast_horizon_wide,
    validate_forecast_risk_curves,
)
from symphony_dt.io.ppmi import load_ppmi_inputs
from symphony_dt.logging import configure_logging, get_logger
from symphony_dt.notebook.ingest import ingest_notebook
from symphony_dt.orchestration.plan import build_fold_plan
from symphony_dt.parity import parity_check_run_dir
from symphony_dt.preprocessing.report import export_preprocessing_artifacts
from symphony_dt.results import RunContext, create_run_context, export_dataframe, export_json, export_text
from symphony_dt.state.states import build_state_occupancy, build_states_long
from symphony_dt.trajectory.clusters import build_trajectory_clusters
from symphony_dt.trajectory.transitions import build_transitions_edge_list, build_transitions_long
from symphony_dt.utils.seed import set_global_seed
from symphony_dt.utils.sharding import assign_patients_to_shard
from symphony_dt.utils.subset import select_demo_patient_ids
from symphony_dt.utils.subset_clinical import select_demo_patient_ids_clinical_best

app = typer.Typer(add_completion=False)

from symphony_dt.nuts_cli import app as nuts_app
app.add_typer(nuts_app, name="nuts")
console = Console()


@app.command()
def run(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False, readable=True),
    mode: str = typer.Option("demo", "--mode", help="demo|full|shard"),
    shard_id: Optional[int] = typer.Option(None, "--shard-id"),
    num_shards: Optional[int] = typer.Option(None, "--num-shards"),
    notebook: Optional[Path] = typer.Option(None, "--notebook", help="Optional notebook path for introspection artifact"),
) -> None:
    """Run pipeline and export standardized tables into outputs/<run_id>/results/."""
    cfg: AppConfig = load_config(config)
    configure_logging(cfg.logging.level)
    log = get_logger(__name__)
    set_global_seed(cfg.run.seed)

    run_ctx: RunContext = create_run_context(
        outputs_root=Path(cfg.run.outputs_root),
        project=cfg.run.project,
        seed=cfg.run.seed,
        overwrite=cfg.run.overwrite,
        mode=mode,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    export_text(run_ctx.run_dir / "config_snapshot.yaml", Path(config).read_text(encoding="utf-8"))
    export_json(run_ctx.run_dir / "metadata.json", run_ctx.metadata)

    ppmi = load_ppmi_inputs(cfg)

    if notebook is not None:
        if not notebook.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook}")
        ingest_notebook(notebook, out_path=run_ctx.run_dir / "notebook_introspection.json")

    if mode not in {"demo", "full", "shard"}:
        raise typer.BadParameter("mode must be one of: demo|full|shard")

    all_patient_ids = sorted(ppmi.clinic["patient_id"].unique().tolist())
    if mode == "demo":
        if cfg.demo.patient_id_strategy == "clinical_best":
            patient_ids = select_demo_patient_ids_clinical_best(ppmi.clinic, cfg.demo.n_patients)
        else:
            patient_ids = select_demo_patient_ids(all_patient_ids, cfg.demo.n_patients, cfg.demo.patient_id_strategy, cfg.run.seed)
    elif mode == "shard":
        if shard_id is None or num_shards is None:
            raise typer.BadParameter("--shard-id and --num-shards are required for mode=shard")
        patient_ids = assign_patients_to_shard(all_patient_ids, shard_id=shard_id, num_shards=num_shards, method=cfg.sharding.method).patient_ids
    else:
        patient_ids = all_patient_ids

    baseline = ppmi.baseline[ppmi.baseline["patient_id"].isin(patient_ids)].copy()
    clinic = ppmi.clinic[ppmi.clinic["patient_id"].isin(patient_ids)].copy()
    omics = ppmi.omics[ppmi.omics["patient_id"].isin(patient_ids)].copy()

    out_inputs = run_ctx.run_dir / "inputs"
    out_quality = run_ctx.run_dir / "data_quality"
    out_splits = run_ctx.run_dir / "splits"
    out_results = run_ctx.run_dir / "results"
    out_inputs.mkdir(parents=True, exist_ok=True)
    out_quality.mkdir(parents=True, exist_ok=True)
    out_splits.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    export_dataframe(out_inputs / "baseline.csv", baseline)
    export_dataframe(out_inputs / "clinic_visits.csv", clinic)
    export_dataframe(out_inputs / "omics_prebaseline.csv", omics)

    # Basic data quality table (missingness profile)
    quality_rows = []
    for name, df, key_cols in [
        ("baseline", baseline, ["patient_id"]),
        ("clinic_visits", clinic, ["patient_id","visit_id","t_days_since_baseline"]),
        ("omics_prebaseline", omics, ["patient_id"]),
    ]:
        missing = {c: int(df[c].isna().sum()) for c in df.columns}
        quality_rows.append({
            "table": name,
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "n_patients": int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0,
            "key_columns_present": all(c in df.columns for c in key_cols),
            "missing_by_column_json": json.dumps(missing, sort_keys=True),
        })
    export_dataframe(out_quality / "data_quality_missingness.csv", pd.DataFrame(quality_rows))


    export_preprocessing_artifacts(
        input_dir=Path(cfg.io.input_dir),
        run_dir=run_ctx.run_dir,
        baseline=baseline,
        clinic=clinic,
        omics=omics,
    )

    folds = make_stratified_folds(baseline=baseline, patient_ids=patient_ids, n_folds=cfg.cv.n_folds, seed=cfg.run.seed)
    export_dataframe(out_splits / "fold_assignments.csv", folds.assignment)
    plan = build_fold_plan(folds.assignment, n_folds=cfg.cv.n_folds)
    export_json(out_splits / "fold_plan.json", {"n_folds": plan.n_folds, "folds": plan.folds})
    per_fold = {}
    for k in range(cfg.cv.n_folds):
        train_ids, test_ids = fold_train_test_ids(folds, fold_id=k)
        per_fold[str(k)] = {"train_patient_ids": train_ids, "test_patient_ids": test_ids}
    export_json(out_splits / "fold_train_test_ids.json", per_fold)

    # Forecasts
    inst1 = build_one_step_dataset(clinic)
    pred1 = validate_forecast_one_step(predict_one_step_locf(inst1))
    write_table_atomic(out_results / "forecast_one_step_predictions.csv", pred1)

    insth = build_horizon_dataset(clinic, horizons_days=(180.0, 365.0, 730.0), tolerance_days=30.0)
    predh = validate_forecast_horizon_long(predict_horizon_locf(insth))
    write_table_atomic(out_results / "forecast_horizon_long.csv", predh)

    # Derived forecast tables
    write_table_atomic(out_results / "forecast_horizon_wide.csv", validate_forecast_horizon_wide(horizon_long_to_wide(predh)))
    uq = validate_forecast_horizon_long_uq(horizon_long_to_uq(predh))
    write_table_atomic(out_results / "forecast_horizon_long_uq.csv", uq)
    write_table_atomic(out_results / "forecast_horizon_long_uq_refined.csv", validate_forecast_horizon_long_uq_refined(horizon_long_to_uq_refined(uq)))
    risk = validate_forecast_risk_curves(build_risk_curves(predh))
    write_table_atomic(out_results / "forecast_risk_curves.csv", risk)
    horizon_states = validate_forecast_horizon_states(build_horizon_states(predh))
    write_table_atomic(out_results / "forecast_horizon_states.csv", horizon_states)

    # State + transitions + clusters
    states_long = build_states_long(horizon_states)
    write_table_atomic(out_results / "states_long.csv", states_long)
    write_table_atomic(out_results / "state_occupancy.csv", build_state_occupancy(states_long))
    transitions_long = build_transitions_long(states_long)
    write_table_atomic(out_results / "transitions_long.csv", transitions_long)
    write_table_atomic(out_results / "transitions_edge_list.csv", build_transitions_edge_list(transitions_long))
    traj = build_trajectory_clusters(states_long)
    write_table_atomic(out_results / "trajectory_clusters.csv", traj)

    # Evaluation
    mae = pd.concat([evaluation_mae_table(pred1, "one_step"), evaluation_mae_table(predh, "horizon_long")], ignore_index=True)
    write_table_atomic(out_results / "evaluation_mae.csv", mae)
    cov_base = pd.concat([evaluation_coverage_table(pred1, "one_step"), evaluation_coverage_table(predh, "horizon_long")], ignore_index=True)

    strat_cfg = StratifyConfig()
    mae_strat = pd.concat([
        evaluation_stratified_mae_table(pred1, baseline, clinic, "one_step", strat_cfg),
        evaluation_stratified_mae_table(predh, baseline, clinic, "horizon_long", strat_cfg),
    ], ignore_index=True)
    write_table_atomic(out_results / "evaluation_stratified_mae.csv", mae_strat)

    cov_strat = pd.concat([
        evaluation_stratified_coverage_table(pred1, baseline, clinic, "one_step", strat_cfg),
        evaluation_stratified_coverage_table(predh, baseline, clinic, "horizon_long", strat_cfg),
    ], ignore_index=True)
    cov = pd.concat([cov_base.assign(stratum_type="", stratum_value=""), cov_strat], ignore_index=True)
    write_table_atomic(out_results / "evaluation_coverage.csv", cov)

    # Clinical + reporting tables
    write_table_atomic(out_results / "clinical_event_rates_by_state.csv", build_clinical_event_rates_by_state(risk, horizon_states))
    write_table_atomic(out_results / "clinical_event_rates_by_trajectory.csv", build_clinical_event_rates_by_trajectory(risk, traj))
    write_table_atomic(out_results / "clinical_event_rates_by_trust.csv", build_clinical_event_rates_by_trust(risk))
    write_table_atomic(out_results / "clinical_event_pairs.csv", build_clinical_event_pairs(risk))
    write_table_atomic(out_results / "clinician_summary_risk_table.csv", build_clinician_summary_risk_table(risk, horizon_states, traj))
    write_table_atomic(out_results / "patient_panel.csv", build_patient_panel(baseline, clinic, traj, states_long, risk))
    write_table_atomic(out_results / "personalization_diagnostics.csv", build_personalization_diagnostics(
        baseline=baseline, clinic=clinic, pred_one_step=pred1, pred_horizon_long=predh, risk=risk,
        states_long=states_long, transitions_long=transitions_long, trajectory_clusters=traj,
    ))

    console.print(f"[bold green]OK[/bold green] Run complete. Results: {out_results}")


@app.command()
def aggregate_all(
    runs_root: Path = typer.Option(..., "--runs-root", exists=True, file_okay=False),
    out_dir: Path = typer.Option(..., "--out-dir"),
) -> None:
    """Aggregate all standardized outputs across shard runs."""
    configure_logging("INFO")
    written = aggregate_all_results(runs_root=runs_root, out_dir=out_dir)
    console.print(f"[bold green]OK[/bold green] Aggregated {len(written)} tables into: {out_dir}")


@app.command()
def parity_check(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
    notebook: Optional[Path] = typer.Option(None, "--notebook"),
) -> None:
    """Validate that a run directory contains required outputs and that key schemas validate."""
    configure_logging("INFO")
    res = parity_check_run_dir(run_dir=run_dir, notebook=notebook)
    if res.passed:
        console.print("[bold green]OK[/bold green] Parity check passed.")
        return
    if res.missing_files:
        console.print("[bold red]Missing files:[/bold red] " + ", ".join(res.missing_files))
    if res.errors:
        for k, v in res.errors.items():
            console.print(f"[bold red]Schema error[/bold red] {k}: {v}")
    raise typer.Exit(code=2)


@app.command()
def aggregate_forecast_one_step(
    runs_root: Path = typer.Option(..., "--runs-root", exists=True, file_okay=False),
    out_dir: Path = typer.Option(..., "--out-dir"),
) -> None:
    """Aggregate forecast_one_step_predictions.csv across run dirs."""
    from symphony_dt.forecast.io import aggregate_tables
    configure_logging("INFO")
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = []
    for rd in sorted([p for p in runs_root.glob("*") if p.is_dir()]):
        p = rd / "results" / "forecast_one_step_predictions.csv"
        if p.exists():
            inputs.append(p)
    if not inputs:
        raise typer.BadParameter("No forecast_one_step_predictions.csv found under run dirs.")
    out_path = out_dir / "forecast_one_step_predictions.csv"
    aggregate_tables(inputs=inputs, out_path=out_path, dedup_keys=["patient_id","anchor_visit_id","target_visit_id"])
    console.print(f"[bold green]OK[/bold green] Wrote: {out_path}")


@app.command()
def aggregate_forecast_horizon_long(
    runs_root: Path = typer.Option(..., "--runs-root", exists=True, file_okay=False),
    out_dir: Path = typer.Option(..., "--out-dir"),
) -> None:
    """Aggregate forecast_horizon_long.csv across run dirs."""
    from symphony_dt.forecast.io import aggregate_tables
    configure_logging("INFO")
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = []
    for rd in sorted([p for p in runs_root.glob("*") if p.is_dir()]):
        p = rd / "results" / "forecast_horizon_long.csv"
        if p.exists():
            inputs.append(p)
    if not inputs:
        raise typer.BadParameter("No forecast_horizon_long.csv found under run dirs.")
    out_path = out_dir / "forecast_horizon_long.csv"
    aggregate_tables(inputs=inputs, out_path=out_path, dedup_keys=["patient_id","anchor_visit_id","target_visit_id","horizon_center_days"])
    console.print(f"[bold green]OK[/bold green] Wrote: {out_path}")


@app.command()
def aggregate_evaluation(
    runs_root: Path = typer.Option(..., "--runs-root", exists=True, file_okay=False),
    out_dir: Path = typer.Option(..., "--out-dir"),
) -> None:
    """Aggregate evaluation_mae.csv and evaluation_coverage.csv across run dirs."""
    configure_logging("INFO")
    out_dir.mkdir(parents=True, exist_ok=True)
    mae_tables = []
    cov_tables = []
    for rd in sorted([p for p in runs_root.glob("*") if p.is_dir()]):
        p_mae = rd / "results" / "evaluation_mae.csv"
        p_cov = rd / "results" / "evaluation_coverage.csv"
        if p_mae.exists():
            mae_tables.append(pd.read_csv(p_mae))
        if p_cov.exists():
            cov_tables.append(pd.read_csv(p_cov))
    if not mae_tables or not cov_tables:
        raise typer.BadParameter("Missing evaluation tables under run dirs.")
    mae_all = pd.concat(mae_tables, ignore_index=True)
    cov_all = pd.concat(cov_tables, ignore_index=True)
    mae_out = mae_all.groupby("scope", dropna=False).mean(numeric_only=True).reset_index()
    cov_out = cov_all.groupby(["scope","stratum_type","stratum_value"], dropna=False).mean(numeric_only=True).reset_index()
    write_table_atomic(out_dir / "evaluation_mae.csv", mae_out)
    write_table_atomic(out_dir / "evaluation_coverage.csv", cov_out)
    console.print(f"[bold green]OK[/bold green] Wrote evaluation aggregates to: {out_dir}")