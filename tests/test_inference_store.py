from pathlib import Path

import numpy as np

from symphony_dt.inference.store import DrawStore, PosteriorDrawSet


def test_save_load_roundtrip(tmp_path: Path) -> None:
    ds = PosteriorDrawSet(
        arrays={
            "mu_r": np.random.default_rng(0).normal(size=(10, 3)).astype(np.float64),
            "sigma_xi": np.ones((10, 3), dtype=np.float64),
        },
        metadata={"fold_id": 0, "sampler": "unit_test"},
        diagnostics={"divergences": 0, "rhat": {"mu_r[0]": 1.0}, "ess_bulk": {"mu_r[0]": 500}},
    )
    store = DrawStore(tmp_path / "draws")
    store.save(ds)
    ds2 = store.load()
    assert ds2.n_draws() == 10
    assert ds2.metadata["fold_id"] == 0
    assert np.allclose(ds2.arrays["mu_r"], ds.arrays["mu_r"])
