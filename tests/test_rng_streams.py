import numpy as np
from symphony_dt.math.rng import RNGStreams


def test_rng_patient_deterministic() -> None:
    rs = RNGStreams.from_seed(123)
    a = rs.patient_rng(patient_id=42, fold_id=1).normal(size=5)
    b = rs.patient_rng(patient_id=42, fold_id=1).normal(size=5)
    assert np.allclose(a, b)


def test_rng_patient_differs_by_fold() -> None:
    rs = RNGStreams.from_seed(123)
    a = rs.patient_rng(patient_id=42, fold_id=1).normal(size=5)
    b = rs.patient_rng(patient_id=42, fold_id=2).normal(size=5)
    assert not np.allclose(a, b)
