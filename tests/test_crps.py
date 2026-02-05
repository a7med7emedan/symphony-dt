import numpy as np
from symphony_dt.inference_backend.eval_draws import crps_from_draws


def test_crps_runs() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    draws = rng.normal(size=(50,200))
    v = crps_from_draws(draws, y)
    assert np.isfinite(v)
