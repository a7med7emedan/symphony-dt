from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symphony_dt.forecast.io import aggregate_tables, write_table_atomic


@dataclass(frozen=True)
class AggregateSpec:
    filename: str
    dedup_keys: Optional[List[str]] = None
    groupby_mean_keys: Optional[List[str]] = None


DEFAULT_SPECS: List[AggregateSpec] = [
    AggregateSpec("forecast_one_step_predictions.csv", dedup_keys=["patient_id", "anchor_visit_id", "target_visit_id"]),
    AggregateSpec("forecast_horizon_long.csv", dedup_keys=["patient_id", "anchor_visit_id", "target_visit_id", "horizon_center_days"]),
    AggregateSpec("forecast_horizon_wide.csv", dedup_keys=["patient_id", "anchor_visit_id", "target_visit_id", "horizon_center_days"]),
    AggregateSpec("forecast_horizon_long_uq.csv", dedup_keys=["patient_id", "anchor_visit_id", "target_visit_id", "horizon_center_days"]),
    AggregateSpec("forecast_horizon_long_uq_refined.csv", dedup_keys=["patient_id", "anchor_visit_id", "target_visit_id", "horizon_center_days"]),
    AggregateSpec("forecast_risk_curves.csv", dedup_keys=["patient_id", "anchor_visit_id", "target_visit_id", "horizon_center_days", "endpoint", "direction", "delta_threshold"]),
    AggregateSpec("forecast_horizon_states.csv", dedup_keys=["patient_id", "anchor_visit_id", "target_visit_id", "horizon_center_days"]),
    AggregateSpec("states_long.csv", dedup_keys=["patient_id", "visit_id"]),
    AggregateSpec("state_occupancy.csv", groupby_mean_keys=["state_engine", "state_id"]),
    AggregateSpec("transitions_long.csv", dedup_keys=["patient_id", "from_visit_id", "to_visit_id"]),
    AggregateSpec("transitions_edge_list.csv", groupby_mean_keys=["state_engine", "from_state_id", "to_state_id"]),
    AggregateSpec("trajectory_clusters.csv", dedup_keys=["patient_id"]),
    AggregateSpec("clinical_event_rates_by_state.csv", groupby_mean_keys=["state_engine", "state_id", "endpoint", "direction", "delta_threshold", "horizon_center_days"]),
    AggregateSpec("clinical_event_rates_by_trajectory.csv", groupby_mean_keys=["cluster_engine", "trajectory_cluster", "endpoint", "direction", "delta_threshold", "horizon_center_days"]),
    AggregateSpec("clinical_event_rates_by_trust.csv", groupby_mean_keys=["trust_label", "endpoint", "direction", "delta_threshold", "horizon_center_days"]),
    AggregateSpec("clinical_event_pairs.csv", groupby_mean_keys=["event_a", "event_b"]),
    AggregateSpec("personalization_diagnostics.csv", dedup_keys=["metric"]),
    AggregateSpec("clinician_summary_risk_table.csv", groupby_mean_keys=["state_engine", "state_id", "cluster_engine", "trajectory_cluster", "horizon_center_days", "endpoint", "direction", "delta_threshold"]),
    AggregateSpec("patient_panel.csv", dedup_keys=["patient_id"]),
    AggregateSpec("evaluation_mae.csv", groupby_mean_keys=["scope"]),
    AggregateSpec("evaluation_coverage.csv", groupby_mean_keys=["scope", "stratum_type", "stratum_value"]),
    AggregateSpec("evaluation_stratified_mae.csv", groupby_mean_keys=["scope", "stratum_type", "stratum_value"]),
]


def _find_result_files(runs_root: Path, filename: str) -> List[Path]:
    files: List[Path] = []
    for rd in sorted([p for p in runs_root.glob("*") if p.is_dir()]):
        p = rd / "results" / filename
        if p.exists():
            files.append(p)
    return files


def _groupby_mean(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    for k in keys:
        if k not in df.columns:
            raise ValueError(f"Cannot groupby_mean; missing key column: {k}")
    return df.groupby(keys, dropna=False).mean(numeric_only=True).reset_index()


def aggregate_all_results(
    runs_root: Path,
    out_dir: Path,
    specs: Optional[List[AggregateSpec]] = None,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = specs or DEFAULT_SPECS

    written: Dict[str, Path] = {}
    for spec in specs:
        inputs = _find_result_files(runs_root, spec.filename)
        if not inputs:
            continue

        out_path = out_dir / spec.filename

        if spec.groupby_mean_keys:
            dfs = [pd.read_csv(p) for p in inputs]
            df = pd.concat(dfs, ignore_index=True)
            df = _groupby_mean(df, spec.groupby_mean_keys)
            write_table_atomic(out_path, df)
        else:
            aggregate_tables(inputs=inputs, out_path=out_path, dedup_keys=spec.dedup_keys)

        written[spec.filename] = out_path

    return written
