import numpy as np

from symphony_dt.ppc.pit import ks_distance_uniform, pit_values
from symphony_dt.model.observation import SUPPORTS


def test_pit_and_ks() -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform([SUPPORTS["U"].L, SUPPORTS["Q"].L, SUPPORTS["S"].L],
                    [SUPPORTS["U"].U, SUPPORTS["Q"].U, SUPPORTS["S"].U],
                    size=(100,3))
    mu = y.copy()
    sigma = np.array([5.0, 1.5, 3.0])
    u = pit_values(y, mu, sigma)
    d = ks_distance_uniform(u)
    assert 0.0 <= d <= 1.0
