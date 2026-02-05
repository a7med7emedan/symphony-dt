from pathlib import Path

import pandas as pd

from symphony_dt.forecast.io import write_table_atomic
from symphony_dt.forecast.schema import validate_forecast_one_step
from symphony_dt.cli import aggregate_forecast_one_step


def _make_df(pid: int, a: int, t: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": pid,
                "anchor_visit_id": a,
                "target_visit_id": t,
                "t_anchor_days": 0.0,
                "t_target_days": 100.0,
                "dt_days": 100.0,
                "y_anchor_updrs3": 10.0,
                "y_anchor_moca": 28.0,
                "y_anchor_scopa": 5.0,
                "y_target_updrs3": 12.0,
                "y_target_moca": 27.0,
                "y_target_scopa": 6.0,
                "ledd_anchor": 0.0,
                "ledd_target": 100.0,
                "pred_mean_updrs3": 10.0,
                "pred_p05_updrs3": 0.0,
                "pred_p95_updrs3": 22.0,
                "pred_mean_moca": 28.0,
                "pred_p05_moca": 25.0,
                "pred_p95_moca": 30.0,
                "pred_mean_scopa": 5.0,
                "pred_p05_scopa": 0.0,
                "pred_p95_scopa": 12.0,
                "engine": "locf",
            }
        ]
    )


def test_aggregate_command(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    r0 = runs_root / "r0" / "results"
    r1 = runs_root / "r1" / "results"
    r0.mkdir(parents=True, exist_ok=True)
    r1.mkdir(parents=True, exist_ok=True)

    df0 = validate_forecast_one_step(_make_df(1, 10, 11))
    df1 = validate_forecast_one_step(_make_df(2, 20, 21))
    write_table_atomic(r0 / "forecast_one_step_predictions.csv", df0)
    write_table_atomic(r1 / "forecast_one_step_predictions.csv", df1)

    out_dir = tmp_path / "agg"
    aggregate_forecast_one_step(runs_root=runs_root, out_dir=out_dir)

    out_path = out_dir / "forecast_one_step_predictions.csv"
    assert out_path.exists()
    out = pd.read_csv(out_path)
    assert len(out) == 2
