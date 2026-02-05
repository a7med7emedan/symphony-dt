import numpy as np

from symphony_dt.inference_backend.bootstrap import BootstrapStabilityConfig, pathway_sign_stability


def test_pathway_sign_stability() -> None:
    samples = {
        (0, 0): np.ones(200),
        (1, 0): -np.ones(200),
        (2, 0): np.concatenate([np.ones(180), -np.ones(20)]),
    }
    out = pathway_sign_stability(samples, cfg=BootstrapStabilityConfig(B=200, sign_freq_threshold=0.90))
    assert out[(0, 0)]["stable"] == 1.0
    assert out[(1, 0)]["stable"] == 0.0  # sign is negative, freq>0 is 0
    assert out[(2, 0)]["stable"] == 1.0
