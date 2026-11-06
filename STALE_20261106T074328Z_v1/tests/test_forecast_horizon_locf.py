import pandas as pd

from symphony_dt.forecast.horizon import build_horizon_dataset, predict_horizon_locf
from symphony_dt.forecast.schema_horizon import validate_forecast_horizon_long


def test_build_horizon_and_predict_and_validate() -> None:
    clinic = pd.DataFrame(
        {
            "patient_id": [1, 1, 1],
            "visit_id": [10, 11, 12],
            "t_days_since_baseline": [0.0, 190.0, 370.0],
            "updrs3_total": [10.0, 12.0, 13.0],
            "moca_total": [28.0, 27.0, 26.0],
            "scopa_aut_total": [5.0, 6.0, 7.0],
            "ledd_total": [0.0, 100.0, 200.0],
        }
    )
    inst = build_horizon_dataset(clinic, horizons_days=(180.0, 365.0), tolerance_days=30.0)
    assert len(inst) >= 1
    pred = predict_horizon_locf(inst)
    out = validate_forecast_horizon_long(pred)
    assert len(out) == len(pred)
