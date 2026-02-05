import numpy as np

from symphony_dt.inference.store import PosteriorDrawSet
from symphony_dt.inference_backend.postprocess import DrawExtractionConfig, extract_draws


def test_extract_draws_deterministic() -> None:
    rng = np.random.default_rng(0)
    ds = PosteriorDrawSet(arrays={"mu_r": rng.normal(size=(500, 3))}, metadata={}, diagnostics={})
    a = extract_draws(ds, DrawExtractionConfig(L_target=200, seed=123)).arrays["mu_r"]
    b = extract_draws(ds, DrawExtractionConfig(L_target=200, seed=123)).arrays["mu_r"]
    assert np.allclose(a, b)
    assert a.shape[0] == 200
