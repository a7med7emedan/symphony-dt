from pathlib import Path

import pandas as pd

from symphony_dt.aggregate.all import aggregate_all_results
from symphony_dt.forecast.io import write_table_atomic


def test_aggregate_all_results(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    r0 = runs_root / "r0" / "results"
    r1 = runs_root / "r1" / "results"
    r0.mkdir(parents=True, exist_ok=True)
    r1.mkdir(parents=True, exist_ok=True)

    df1 = pd.DataFrame([{
        "patient_id": 1, "anchor_visit_id": 10, "target_visit_id": 11,
        "t_anchor_days": 0.0, "t_target_days": 100.0, "dt_days": 100.0,
        "y_anchor_updrs3": 10.0, "y_anchor_moca": 28.0, "y_anchor_scopa": 5.0,
        "y_target_updrs3": 12.0, "y_target_moca": 27.0, "y_target_scopa": 6.0,
        "ledd_anchor": 0.0, "ledd_target": 100.0,
        "pred_mean_updrs3": 10.0, "pred_p05_updrs3": 0.0, "pred_p95_updrs3": 20.0,
        "pred_mean_moca": 28.0, "pred_p05_moca": 25.0, "pred_p95_moca": 30.0,
        "pred_mean_scopa": 5.0, "pred_p05_scopa": 0.0, "pred_p95_scopa": 10.0,
        "engine": "locf"
    }])
    write_table_atomic(r0 / "forecast_one_step_predictions.csv", df1)
    write_table_atomic(r1 / "forecast_one_step_predictions.csv", df1.assign(patient_id=2))

    out_dir = tmp_path / "out"
    written = aggregate_all_results(runs_root=runs_root, out_dir=out_dir)
    assert "forecast_one_step_predictions.csv" in written
    out = pd.read_csv(out_dir / "forecast_one_step_predictions.csv")
    assert len(out) == 2
