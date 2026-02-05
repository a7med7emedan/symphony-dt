import numpy as np

from symphony_dt.model.dynamics import propagate_state
from symphony_dt.model.params import CouplingMatrix


def test_batch_matches_scalar_calls() -> None:
    C = CouplingMatrix(c_mm=0.1, c_cm=0.0, c_cc=0.1, c_cn=0.0, c_nm=0.0, c_nn=0.1).as_array()
    sigma_xi = np.array([0.1, 0.1, 0.1])
    dt = 30.0

    s = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]])
    r = np.array([[0.05, 0.05, 0.05], [0.1, 0.1, 0.1]])
    eps = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.1]])

    out_batch = propagate_state(s, r, C=C, sigma_xi=sigma_xi, dt=30.0, epsilon=eps)
    out0 = propagate_state(s[0], r[0], C=C, sigma_xi=sigma_xi, dt=30.0, epsilon=eps[0])
    out1 = propagate_state(s[1], r[1], C=C, sigma_xi=sigma_xi, dt=30.0, epsilon=eps[1])

    assert np.allclose(out_batch[0], out0)
    assert np.allclose(out_batch[1], out1)
