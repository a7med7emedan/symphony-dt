from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symphony_dt.forecast.schema import validate_forecast_one_step
from symphony_dt.forecast.schema_horizon import validate_forecast_horizon_long
from symphony_dt.forecast.schema_extra import (
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
}


def parity_check_run_dir(run_dir: Path, notebook: Optional[Path] = None) -> ParityCheckResult:
    results_dir = run_dir / "results"
    missing: List[str] = []
    errors: Dict[str, str] = {}
    checked: List[str] = []

    if notebook is not None and notebook.exists():
        ingest_notebook(notebook, out_path=run_dir / "notebook_introspection.json")

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
            except Exception as e:  # noqa: BLE001
                errors[fn] = str(e)

    passed = (len(missing) == 0) and (len(errors) == 0)
    return ParityCheckResult(passed=passed, checked_files=checked, missing_files=missing, errors=errors)
