import numpy as np

from symphony_dt.math.core import sigmoid
from symphony_dt.model.dynamics import induce_visit_time_trajectory
from symphony_dt.model.params import CouplingMatrix, GlobalDynamicsParams


def test_monotone_x_progression_componentwise() -> None:
    dyn = GlobalDynamicsParams(
        C=CouplingMatrix(c_mm=0.1, c_cm=0.05, c_cc=0.1, c_cn=0.02, c_nm=0.03, c_nn=0.1),
        sigma_xi=(0.15, 0.15, 0.15),
    )
    times = [0.0, 10.0, 40.0, 100.0]
    s0 = np.array([0.0, 0.0, 0.0])
    r = np.array([0.1, 0.1, 0.1])

    # Deterministic epsilons all zeros => xi=exp(-0.5 dt sigma^2) > 0 and increments non-negative.
    eps = [np.zeros(3) for _ in range(len(times) - 1)]
    S = induce_visit_time_trajectory(times, s0=s0, r_i=r, dyn=dyn, epsilons=eps)

    X = sigmoid(S)
    # componentwise non-decreasing over time
    assert np.all(np.diff(X, axis=0) >= -1e-12)
