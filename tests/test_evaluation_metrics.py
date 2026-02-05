import pandas as pd

from symphony_dt.evaluation.metrics import interval_coverage_by_patient, mae_by_patient
from symphony_dt.evaluation.tables import evaluation_coverage_table, evaluation_mae_table


def test_eval_tables() -> None:
    pred = pd.DataFrame(
        {
            "patient_id": [1, 1, 2],
            "y_target_updrs3": [12.0, 11.0, 10.0],
            "y_target_moca": [27.0, 27.0, 28.0],
            "y_target_scopa": [6.0, 6.0, 5.0],
            "pred_mean_updrs3": [10.0, 10.0, 10.0],
            "pred_mean_moca": [28.0, 28.0, 28.0],
            "pred_mean_scopa": [5.0, 5.0, 5.0],
            "pred_p05_updrs3": [0.0, 0.0, 0.0],
            "pred_p95_updrs3": [20.0, 20.0, 20.0],
            "pred_p05_moca": [25.0, 25.0, 25.0],
            "pred_p95_moca": [30.0, 30.0, 30.0],
            "pred_p05_scopa": [0.0, 0.0, 0.0],
            "pred_p95_scopa": [10.0, 10.0, 10.0],
        }
    )

    mae = mae_by_patient(pred)
    cov = interval_coverage_by_patient(pred)
    assert len(mae) == 2
    assert len(cov) == 2

    t1 = evaluation_mae_table(pred, scope="one_step")
    t2 = evaluation_coverage_table(pred, scope="one_step")
    assert "mae_overall" in t1.columns
    assert "cov95_overall" in t2.columns
