import pandas as pd

from symphony_dt.evaluation.stratified import evaluation_stratified_mae_table


def test_stratified_mae_table_smoke() -> None:
    pred = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2],
            "ledd_anchor": [0.0, 100.0, 0.0, 50.0],
            "y_target_updrs3": [12.0, 11.0, 10.0, 11.0],
            "y_target_moca": [27.0, 27.0, 28.0, 28.0],
            "y_target_scopa": [6.0, 6.0, 5.0, 5.0],
            "pred_mean_updrs3": [10.0, 10.0, 10.0, 10.0],
            "pred_mean_moca": [28.0, 28.0, 28.0, 28.0],
            "pred_mean_scopa": [5.0, 5.0, 5.0, 5.0],
        }
    )
    baseline = pd.DataFrame(
        {"patient_id": [1, 2], "sex": ["M", "F"], "age_baseline_years": [60.0, 55.0]}
    )
    clinic = pd.DataFrame({"patient_id": [1, 1, 2, 2], "visit_id": [10, 11, 20, 21], "t_days_since_baseline": [0, 100, 0, 120]})

    out = evaluation_stratified_mae_table(pred, baseline, clinic, scope="one_step")
    assert len(out) > 0
    assert {"scope", "stratum_type", "stratum_value", "mae_overall"}.issubset(out.columns)
