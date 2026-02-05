from __future__ import annotations

from typing import Dict

import pandas as pd

from symphony_dt.evaluation.metrics import (
    interval_coverage_by_patient,
    mae_by_patient,
    summarize_coverage,
    summarize_mae,
)


def evaluation_mae_table(pred: pd.DataFrame, scope: str) -> pd.DataFrame:
    """
    Create evaluation_mae.csv table (minimal version).

    scope: label, e.g. 'one_step' or 'horizon'
    """
    per_patient = mae_by_patient(pred)
    summ = summarize_mae(per_patient)
    row = {"scope": scope, **summ, "n_patients": int(per_patient["patient_id"].nunique())}
    return pd.DataFrame([row])


def evaluation_coverage_table(pred: pd.DataFrame, scope: str) -> pd.DataFrame:
    """
    Create evaluation_coverage.csv table (minimal version).

    scope: label, e.g. 'one_step' or 'horizon'
    """
    per_patient = interval_coverage_by_patient(pred)
    summ = summarize_coverage(per_patient)
    row = {"scope": scope, **summ, "n_patients": int(per_patient["patient_id"].nunique())}
    return pd.DataFrame([row])
