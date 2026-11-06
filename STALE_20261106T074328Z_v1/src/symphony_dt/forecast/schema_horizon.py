from __future__ import annotations

import pandas as pd

from symphony_dt.schemas.contracts import ColumnRule, TableContract, validate_table


FORECAST_HORIZON_LONG_CONTRACT = TableContract(
    table_name="forecast_horizon_long",
    primary_key=("patient_id", "anchor_visit_id", "target_visit_id", "horizon_center_days"),
    required=(
        ColumnRule("patient_id", int, nullable=False),
        ColumnRule("anchor_visit_id", int, nullable=False),
        ColumnRule("target_visit_id", int, nullable=False),
        ColumnRule("horizon_center_days", float, nullable=False),
        ColumnRule("t_anchor_days", float, nullable=False, min_value=0.0),
        ColumnRule("t_target_days", float, nullable=False, min_value=0.0),
        ColumnRule("dt_days", float, nullable=False, min_value=0.0),

        ColumnRule("y_anchor_updrs3", float, nullable=True, min_value=0.0, max_value=132.0),
        ColumnRule("y_anchor_moca", float, nullable=True, min_value=0.0, max_value=30.0),
        ColumnRule("y_anchor_scopa", float, nullable=True, min_value=0.0, max_value=69.0),

        ColumnRule("y_target_updrs3", float, nullable=True, min_value=0.0, max_value=132.0),
        ColumnRule("y_target_moca", float, nullable=True, min_value=0.0, max_value=30.0),
        ColumnRule("y_target_scopa", float, nullable=True, min_value=0.0, max_value=69.0),

        ColumnRule("ledd_anchor", float, nullable=True, min_value=0.0),
        ColumnRule("ledd_target", float, nullable=True, min_value=0.0),

        ColumnRule("pred_mean_updrs3", float, nullable=False, min_value=0.0, max_value=132.0),
        ColumnRule("pred_p05_updrs3", float, nullable=False, min_value=0.0, max_value=132.0),
        ColumnRule("pred_p95_updrs3", float, nullable=False, min_value=0.0, max_value=132.0),

        ColumnRule("pred_mean_moca", float, nullable=False, min_value=0.0, max_value=30.0),
        ColumnRule("pred_p05_moca", float, nullable=False, min_value=0.0, max_value=30.0),
        ColumnRule("pred_p95_moca", float, nullable=False, min_value=0.0, max_value=30.0),

        ColumnRule("pred_mean_scopa", float, nullable=False, min_value=0.0, max_value=69.0),
        ColumnRule("pred_p05_scopa", float, nullable=False, min_value=0.0, max_value=69.0),
        ColumnRule("pred_p95_scopa", float, nullable=False, min_value=0.0, max_value=69.0),

        ColumnRule("engine", str, nullable=False),
    ),
)


def validate_forecast_horizon_long(df: pd.DataFrame) -> pd.DataFrame:
    return validate_table(df, FORECAST_HORIZON_LONG_CONTRACT)
