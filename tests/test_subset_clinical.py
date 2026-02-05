import pandas as pd

from symphony_dt.utils.subset_clinical import select_demo_patient_ids_clinical_best


def test_clinical_best_prefers_more_visits() -> None:
    clinic = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2, 2],
            "visit_id": [10, 11, 20, 21, 22],
            "t_days_since_baseline": [0, 100, 0, 180, 365],
            "updrs3_total": [10, 11, 9, 10, 12],
            "moca_total": [28, 27, 29, 28, 27],
            "scopa_aut_total": [5, 6, 4, 5, 6],
            "miss_updrs3": [False, False, False, False, False],
            "miss_moca": [False, False, False, False, False],
            "miss_scopa": [False, False, False, False, False],
        }
    )
    chosen = select_demo_patient_ids_clinical_best(clinic, n_patients=1)
    assert chosen == [2]
