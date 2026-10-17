from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symphony_dt.forecast.schema import validate_forecast_one_step
from symphony_dt.forecast.schema_horizon import validate_forecast_horizon_long
from symphony_dt.forecast.schema_extra import (
    validate_evaluation_coverage,
    validate_forecast_horizon_long_uq,
    validate_forecast_horizon_long_uq_refined,
    validate_forecast_horizon_states,
    validate_forecast_horizon_wide,
    validate_forecast_risk_curves,
)
from symphony_dt.notebook.ingest import ingest_notebook


@dataclass(frozen=True)
class ParityCheckResult:
    passed: bool
    checked_files: List[str]
    missing_files: List[str]
    errors: Dict[str, str]
    # Non-fatal findings, e.g. deprecated cov95_* columns in tables or notebooks.
    warnings: List[str] = field(default_factory=list)


_REQUIRED = [
    "forecast_one_step_predictions.csv",
    "forecast_horizon_long.csv",
    "forecast_horizon_wide.csv",
    "forecast_horizon_long_uq.csv",
    "forecast_horizon_long_uq_refined.csv",
    "forecast_risk_curves.csv",
    "forecast_horizon_states.csv",
    "states_long.csv",
    "state_occupancy.csv",
    "trajectory_clusters.csv",
    "transitions_long.csv",
    "transitions_edge_list.csv",
    "clinical_event_rates_by_state.csv",
    "clinical_event_rates_by_trajectory.csv",
    "clinical_event_rates_by_trust.csv",
    "clinical_event_pairs.csv",
    "evaluation_mae.csv",
    "evaluation_stratified_mae.csv",
    "evaluation_coverage.csv",
    "personalization_diagnostics.csv",
    "clinician_summary_risk_table.csv",
    "patient_panel.csv",
]

_VALIDATORS = {
    "forecast_one_step_predictions.csv": validate_forecast_one_step,
    "forecast_horizon_long.csv": validate_forecast_horizon_long,
    "forecast_horizon_wide.csv": validate_forecast_horizon_wide,
    "forecast_horizon_long_uq.csv": validate_forecast_horizon_long_uq,
    "forecast_horizon_long_uq_refined.csv": validate_forecast_horizon_long_uq_refined,
    "forecast_risk_curves.csv": validate_forecast_risk_curves,
    "forecast_horizon_states.csv": validate_forecast_horizon_states,
    "evaluation_coverage.csv": validate_evaluation_coverage,
}


def parity_check_run_dir(run_dir: Path, notebook: Optional[Path] = None) -> ParityCheckResult:
    results_dir = run_dir / "results"
    missing: List[str] = []
    errors: Dict[str, str] = {}
    checked: List[str] = []

    warnings: List[str] = []

    if notebook is not None and notebook.exists():
        info = ingest_notebook(notebook, out_path=run_dir / "notebook_introspection.json")
        old_cols = sorted({r["column"] for r in info.coverage_column_refs if r["deprecated"]})
        if old_cols:
            warnings.append(
                f"{notebook.name} reads deprecated coverage columns {old_cols}; use the cov90_* names instead"
            )

    for fn in _REQUIRED:
        p = results_dir / fn
        if not p.exists():
            missing.append(fn)
            continue
        checked.append(fn)
        if fn in _VALIDATORS:
            try:
                df = pd.read_csv(p)
                _VALIDATORS[fn](df)
                if fn == "evaluation_coverage.csv" and not any(c.startswith("cov90_") for c in df.columns):
                    warnings.append(f"{fn} has only the deprecated cov95_* columns; rerun to write cov90_*")
            except Exception as e:  # noqa: BLE001
                errors[fn] = str(e)

    passed = (len(missing) == 0) and (len(errors) == 0)
    return ParityCheckResult(passed=passed, checked_files=checked, missing_files=missing, errors=errors, warnings=warnings)
