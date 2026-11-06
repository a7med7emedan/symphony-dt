"""SBC statistics, the multiplicity-controlled judge, the SBC driver and PPC summaries."""

from __future__ import annotations

import numpy as np
import pytest

from symphony_dt.calibration import (
    SBCDesign,
    boundary_statistics,
    chi_squared_uniformity,
    judge,
    ks_uniformity,
    pit_values,
    ppc_coverage,
    ppc_summary,
    rank_histogram_bands,
    run_sbc,
)


def test_judge_methods_and_reported_risk():
    p = {"a": 0.0001, "b": 0.004, "c": 0.3, "d": 0.9}
    bon = judge(p, method="bonferroni")
    assert bon.rejected == ["a"] and bon.adjusted_threshold["c"] == pytest.approx(0.0025)
    raw = judge(p, method="none")
    assert raw.rejected == ["a", "b"]
    bh = judge(p, method="bh")
    assert bh.rejected == ["a", "b"]
    assert bon.fwer_uncorrected == pytest.approx(1 - 0.99**4)
    assert not bon.passed and judge({"x": 0.5}).passed
    assert judge({f"p{i}": 0.5 for i in range(55)}).fwer_uncorrected == pytest.approx(0.4246, abs=1e-3)
    assert bon.to_dict()["m"] == 4
    with pytest.raises(ValueError):
        judge({})
    with pytest.raises(ValueError):
        judge(p, method="other")  # type: ignore[arg-type]


def test_uniformity_tests():
    rng = np.random.default_rng(0)
    uni = rng.integers(0, 200, size=2000)
    assert chi_squared_uniformity(uni, 199)["uniform"]
    skew = np.minimum(rng.integers(0, 200, size=2000), rng.integers(0, 200, size=2000))
    assert not chi_squared_uniformity(skew, 199)["uniform"]
    with pytest.raises(ValueError, match="equal bins"):
        chi_squared_uniformity(uni, 198)
    assert ks_uniformity(uni, 199)["p"] > 0.01
    lo, hi = rank_histogram_bands(200, 20)
    assert lo < 10 < hi


def test_run_sbc_on_a_conjugate_model():
    """Normal mean with known variance: the exact posterior makes SBC ranks uniform."""

    def generator(rng):
        mu = rng.normal()
        return rng.normal(mu, 1.0, size=10), {"mu": mu}

    def fitter(data, seed):
        rng = np.random.default_rng(seed)
        post_var = 1 / (1 + len(data))
        if seed % 7 == 0:
            return None
        return {"mu": rng.normal(post_var * data.sum(), np.sqrt(post_var), size=400)}

    res = run_sbc(SBCDesign(generator, fitter), n_replicates=200, seed=1)
    assert res.verdict.passed and res.failed and len(res.ranks) == 200 - len(res.failed)
    assert res.to_dict()["n_replicates"] == len(res.ranks)
    with pytest.raises(ValueError, match="posterior-regime"):
        SBCDesign(generator, fitter, regime="posterior", fitting_prior_matches_generator=False)
    with pytest.raises(ValueError, match="multiple"):
        SBCDesign(generator, fitter, rank_draws=100)
    with pytest.raises(ValueError, match="no replicate"):
        run_sbc(SBCDesign(generator, lambda d, s: None), n_replicates=3)


def test_ppc_functions():
    rng = np.random.default_rng(1)
    draws = rng.normal(size=(500, 300))
    y = rng.normal(size=300)
    pit = pit_values(draws, y)
    assert pit.shape == (300,) and pit.min() >= 0 and pit.max() <= 1
    cov = ppc_coverage(draws, y)
    assert abs(cov["0.95"] - 0.95) < 0.05
    clipped = np.clip(draws * 5 + 2, 0, 10)
    b = boundary_statistics(clipped, np.clip(y * 5 + 2, 0, 10), 0, 10)
    assert b["floor"]["inside"] and set(b) == {"floor", "ceiling"}
    rounded = boundary_statistics(clipped, np.round(np.clip(y * 5 + 2, 0, 10)), 0, 10, round_to_integer=True)
    assert rounded["floor"]["replicated_mean"] >= b["floor"]["replicated_mean"]
    data = {
        "y": list(np.clip(y * 5 + 2, 0, 10)),
        "obs_k": [1] * 100 + [2] * 100 + [3] * 100,
        "Lk": [0, 0, 0],
        "Uk": [10, 10, 10],
    }
    summary = ppc_summary(clipped, data)
    assert set(summary) == {"updrs3", "moca", "scopa_aut"} and sum(summary["moca"]["pit_histogram"]) == 100
