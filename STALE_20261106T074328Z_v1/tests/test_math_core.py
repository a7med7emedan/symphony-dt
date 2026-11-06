import numpy as np
from symphony_dt.math.core import logit, sigmoid, softplus


def test_sigmoid_logit_roundtrip() -> None:
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=float)
    p = sigmoid(x)
    xr = logit(p)
    assert np.allclose(x, xr, atol=1e-8)


def test_softplus_monotone() -> None:
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=float)
    y = softplus(x)
    assert np.all(np.diff(y) > 0)
