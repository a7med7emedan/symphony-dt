import pandas as pd

from symphony_dt.cv.splits import make_stratified_folds


def test_stratified_folds_balanced_and_deterministic() -> None:
    baseline = pd.DataFrame(
        {
            "patient_id": list(range(1, 21)),
            "sex": ["M"] * 10 + ["F"] * 10,
            "age_baseline_years": [60.0] * 20,
            "is_pd_like": [True] * 20,
        }
    )
    ids = baseline["patient_id"].tolist()

    a1 = make_stratified_folds(baseline, ids, n_folds=5, seed=123).assignment
    a2 = make_stratified_folds(baseline, ids, n_folds=5, seed=123).assignment
    assert a1.equals(a2)

    for k in range(5):
        fold = a1[a1["fold_id"] == k]
        assert (fold["sex"] == "M").sum() == 2
        assert (fold["sex"] == "F").sum() == 2
