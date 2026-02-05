from __future__ import annotations

import pandas as pd

from symphony_dt.schemas.contracts import ColumnRule, TableContract, validate_table


FORECAST_HORIZON_WIDE_CONTRACT = TableContract(
    table_name="forecast_horizon_wide",
    primary_key=("patient_id","anchor_visit_id","target_visit_id","horizon_center_days"),
    required=(
        ColumnRule("patient_id", int, nullable=False),
        ColumnRule("anchor_visit_id", int, nullable=False),
        ColumnRule("target_visit_id", int, nullable=False),
        ColumnRule("horizon_center_days", float, nullable=False),
        ColumnRule("t_anchor_days", float, nullable=False),
        ColumnRule("t_target_days", float, nullable=False),
        ColumnRule("dt_days", float, nullable=False),
        ColumnRule("pred_mean_updrs3", float, nullable=False, min_value=0.0, max_value=132.0),
        ColumnRule("pred_mean_moca", float, nullable=False, min_value=0.0, max_value=30.0),
        ColumnRule("pred_mean_scopa", float, nullable=False, min_value=0.0, max_value=69.0),
        ColumnRule("engine", str, nullable=False),
    ),
)

FORECAST_HORIZON_LONG_UQ_CONTRACT = TableContract(
    table_name="forecast_horizon_long_uq",
    primary_key=("patient_id","anchor_visit_id","target_visit_id","horizon_center_days"),
    required=(
        ColumnRule("patient_id", int, nullable=False),
        ColumnRule("anchor_visit_id", int, nullable=False),
        ColumnRule("target_visit_id", int, nullable=False),
        ColumnRule("horizon_center_days", float, nullable=False),
        ColumnRule("pred_width_90_updrs3", float, nullable=True, min_value=0.0),
        ColumnRule("pred_width_90_moca", float, nullable=True, min_value=0.0),
        ColumnRule("pred_width_90_scopa", float, nullable=True, min_value=0.0),
        ColumnRule("engine", str, nullable=False),
    ),
)

FORECAST_HORIZON_LONG_UQ_REFINED_CONTRACT = TableContract(
    table_name="forecast_horizon_long_uq_refined",
    primary_key=("patient_id","anchor_visit_id","target_visit_id","horizon_center_days"),
    required=(
        ColumnRule("patient_id", int, nullable=False),
        ColumnRule("anchor_visit_id", int, nullable=False),
        ColumnRule("target_visit_id", int, nullable=False),
        ColumnRule("horizon_center_days", float, nullable=False),
        ColumnRule("pred_rel_width_90_updrs3", float, nullable=True, min_value=0.0),
        ColumnRule("pred_rel_width_90_moca", float, nullable=True, min_value=0.0),
        ColumnRule("pred_rel_width_90_scopa", float, nullable=True, min_value=0.0),
        ColumnRule("engine", str, nullable=False),
    ),
)

FORECAST_RISK_CURVES_CONTRACT = TableContract(
    table_name="forecast_risk_curves",
    primary_key=("patient_id","anchor_visit_id","target_visit_id","horizon_center_days","endpoint","direction","delta_threshold"),
    required=(
        ColumnRule("patient_id", int, nullable=False),
        ColumnRule("anchor_visit_id", int, nullable=False),
        ColumnRule("target_visit_id", int, nullable=False),
        ColumnRule("horizon_center_days", float, nullable=False),
        ColumnRule("t_anchor_days", float, nullable=False),
        ColumnRule("t_target_days", float, nullable=False),
        ColumnRule("dt_days", float, nullable=False),
        ColumnRule("endpoint", str, nullable=False),
        ColumnRule("direction", str, nullable=False),
        ColumnRule("delta_threshold", float, nullable=False),
        ColumnRule("prob_event", float, nullable=False, min_value=0.0, max_value=1.0),
        ColumnRule("engine", str, nullable=False),
    ),
)

FORECAST_HORIZON_STATES_CONTRACT = TableContract(
    table_name="forecast_horizon_states",
    primary_key=("patient_id","anchor_visit_id","target_visit_id","horizon_center_days"),
    required=(
        ColumnRule("patient_id", int, nullable=False),
        ColumnRule("anchor_visit_id", int, nullable=False),
        ColumnRule("target_visit_id", int, nullable=False),
        ColumnRule("horizon_center_days", float, nullable=False),
        ColumnRule("t_anchor_days", float, nullable=False),
        ColumnRule("t_target_days", float, nullable=False),
        ColumnRule("dt_days", float, nullable=False),
        ColumnRule("state_id", str, nullable=False),
        ColumnRule("state_engine", str, nullable=False),
    ),
)


def validate_forecast_horizon_wide(df: pd.DataFrame) -> pd.DataFrame:
    return validate_table(df, FORECAST_HORIZON_WIDE_CONTRACT)


def validate_forecast_horizon_long_uq(df: pd.DataFrame) -> pd.DataFrame:
    return validate_table(df, FORECAST_HORIZON_LONG_UQ_CONTRACT)


def validate_forecast_horizon_long_uq_refined(df: pd.DataFrame) -> pd.DataFrame:
    return validate_table(df, FORECAST_HORIZON_LONG_UQ_REFINED_CONTRACT)


def validate_forecast_risk_curves(df: pd.DataFrame) -> pd.DataFrame:
    return validate_table(df, FORECAST_RISK_CURVES_CONTRACT)


def validate_forecast_horizon_states(df: pd.DataFrame) -> pd.DataFrame:
    return validate_table(df, FORECAST_HORIZON_STATES_CONTRACT)
