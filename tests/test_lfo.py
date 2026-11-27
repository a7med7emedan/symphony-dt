"""Leave-future-out log predictive density: known densities, Monte Carlo truth and path rebuild."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from symphony_dt import constants as C
from symphony_dt.config import ModelConfig
from symphony_dt.data import simulate_cohort
from symphony_dt.data.simulate import draw_prior
from symphony_dt.evaluate import (
    AnchorDraws,
    leave_future_out,
    observation_draws,
    predictive_log_density,
)
from symphony_dt.forecast import _endpoint_logpdf, split_at_anchor

BOUNDS = (0.0, 30.0)


def _density(y: float, mu: float, sd: float, censored: int) -> float:
    return float(predictive_log_density([y], [[mu]], [[sd]], BOUNDS[0], BOUNDS[1], censored)[0, 0])


def test_wide_bounds_give_the_normal_density():
    y = np.array([1.0, 2.5, -0.3])
    mu = np.array([[0.5, 2.0, 0.0], [1.5, 3.0, -1.0]])
    sd = np.array([[1.0, 0.7, 2.0], [0.5, 1.2, 0.9]])
    for cens in (0, 1):
        got = predictive_log_density(y, mu, sd, -1e6, 1e6, cens)
        np.testing.assert_allclose(got, stats.norm.logpdf(y, mu, sd), rtol=1e-12)


@pytest.mark.parametrize(("mu", "sd"), [(15.0, 4.0), (29.0, 3.0), (-2.0, 1.5), (60.0, 5.0)])
def test_truncated_density_integrates_to_one(mu, sd):
    total, _ = integrate.quad(lambda v: np.exp(_density(v, mu, sd, 0)), *BOUNDS, limit=200)
    assert total == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(("mu", "sd"), [(15.0, 4.0), (29.0, 3.0), (1.0, 2.0)])
def test_censored_density_integrates_to_one_with_its_point_masses(mu, sd):
    interior, _ = integrate.quad(lambda v: np.exp(_density(v, mu, sd, 1)), *BOUNDS, limit=200)
    masses = np.exp(_density(BOUNDS[0], mu, sd, 1)) + np.exp(_density(BOUNDS[1], mu, sd, 1))
    assert interior + masses == pytest.approx(1.0, abs=1e-6)
    assert np.exp(_density(BOUNDS[1], mu, sd, 1)) == pytest.approx(stats.norm.sf(BOUNDS[1], mu, sd))


def test_density_matches_the_fast_route_and_stays_finite_far_in_the_tail():
    rng = np.random.default_rng(3)
    mu, sd = rng.uniform(-5, 35, size=50), rng.uniform(0.5, 6, size=50)
    for cens in (0, 1):
        for y in (0.0, 12.3, 30.0):
            ours = predictive_log_density(np.full(50, y), mu[None], sd[None], *BOUNDS, cens)[0]
            ref = _endpoint_logpdf(y, mu, sd, BOUNDS[0], BOUNDS[1], cens)
            np.testing.assert_allclose(ours, ref, rtol=1e-9, atol=1e-9)
    far = predictive_log_density([29.0], [[200.0]], [[1.0]], *BOUNDS, 0)
    assert np.isfinite(far).all()


def test_predictive_density_rejects_bad_input():
    with pytest.raises(ValueError, match="positive"):
        predictive_log_density([1.0], [[0.0]], [[0.0]], *BOUNDS, 0)
    with pytest.raises(ValueError, match="shape"):
        predictive_log_density([1.0], [[0.0]], [[1.0, 1.0]], *BOUNDS, 0)
    with pytest.raises(ValueError, match="lower"):
        predictive_log_density([1.0], [[0.0]], [[1.0]], 5.0, 5.0, 0)


def _draws(targets: pd.DataFrame, mu, sigma, wide: bool = True) -> AnchorDraws:
    lo, hi = (-1e6, 1e6) if wide else BOUNDS
    return AnchorDraws(targets, np.asarray(mu), np.asarray(sigma), (lo,) * 3, (hi,) * 3, (0, 0, 0))


def test_elpd_recovers_a_known_gaussian_predictive():
    """Draws of the mean from N(m, tau) with noise sd s give the predictive N(m, sqrt(s^2 + tau^2))."""
    rng = np.random.default_rng(7)
    m, tau, s, n_draw = np.array([3.0, -1.0, 10.0]), 2.0, 1.5, 200_000
    y = np.array([4.0, 2.0, 9.0])
    targets = pd.DataFrame({"i": [0, 0, 1], "k": [1, 2, 3], "y": y})
    mu = m + tau * rng.normal(size=(n_draw, 3))
    scores = leave_future_out({2.0: _draws(targets, mu, np.full_like(mu, s))})
    truth = stats.norm.logpdf(y, m, np.hypot(s, tau))
    pw = scores.pointwise["elpd"].to_numpy()
    np.testing.assert_allclose(pw, truth, atol=4 * scores.pointwise["mcse"].max())
    assert scores.pointwise["mcse"].max() < 0.01


def test_joint_density_keeps_the_dependence_of_a_shared_path():
    """Two targets sharing one draw of the mean are bivariate normal with covariance tau^2."""
    rng = np.random.default_rng(8)
    tau, s, n_draw = 1.0, 0.5, 400_000
    shared = tau * rng.normal(size=(n_draw, 1))
    targets = pd.DataFrame({"i": [5, 5], "k": [1, 1], "y": [0.8, 0.9], "t": [3.0, 4.0]})
    scores = leave_future_out({1.0: _draws(targets, np.hstack([shared, shared]), np.full((n_draw, 2), s))})
    cov = np.full((2, 2), tau**2) + np.eye(2) * s**2
    truth = stats.multivariate_normal(mean=[0, 0], cov=cov).logpdf([0.8, 0.9])
    row = scores.joint.iloc[0]
    assert row["elpd_joint"] == pytest.approx(truth, abs=0.02)
    assert row["elpd_pointwise_sum"] < row["elpd_joint"]


def test_constant_draws_give_the_plain_density_and_a_summed_joint():
    targets = pd.DataFrame({"i": [0, 0, 1], "k": [1, 2, 1], "y": [5.0, 20.0, 29.0]})
    mu = np.tile([[6.0, 22.0, 25.0]], (10, 1))
    sd = np.full_like(mu, 3.0)
    scores = leave_future_out([(1.0, _draws(targets, mu, sd, wide=False))])
    lp = predictive_log_density(targets["y"], mu[:1], sd[:1], *BOUNDS, 0)[0]
    np.testing.assert_allclose(scores.pointwise["elpd"], lp)
    np.testing.assert_allclose(scores.joint.set_index("i")["elpd_joint"], [lp[:2].sum(), lp[2]])
    np.testing.assert_allclose(scores.pointwise["mcse"], 0.0, atol=1e-12)


def test_summary_sums_by_anchor_and_endpoint_with_a_cluster_error():
    targets = pd.DataFrame({"i": [0, 0, 1, 2], "k": [1, 2, 1, 2], "y": [1.0, 2.0, 3.0, 4.0]})
    mu = np.tile([[1.0, 2.5, 2.0, 4.0]], (5, 1))
    one = _draws(targets, mu, np.ones_like(mu))
    two = _draws(targets.assign(y=targets["y"] + 1), mu, np.ones_like(mu))
    scores = leave_future_out({1.0: one, 2.0: two})
    pw = scores.pointwise
    first = scores.summary[(scores.summary["anchor"] == 1.0) & (scores.summary["endpoint"] == "all")].iloc[0]
    sums = pw[pw["anchor"] == 1.0].groupby("i")["elpd"].sum().to_numpy()
    assert first["elpd"] == pytest.approx(sums.sum())
    assert first["se"] == pytest.approx(np.sqrt(3 * np.var(sums, ddof=1)))
    assert first["n_targets"] == 4 and first["n_participants"] == 3
    pooled = scores.summary[(scores.summary["anchor"] == "all") & (scores.summary["endpoint"] == "all")].iloc[0]
    assert pooled["elpd"] == pytest.approx(pw["elpd"].sum())
    assert pooled["n_participants"] == 3
    moca = scores.summary[(scores.summary["anchor"] == 2.0) & (scores.summary["endpoint"] == C.ENDPOINTS[1])]
    assert int(moca["n_targets"].iloc[0]) == 2
    assert "| all | all | 8 | 3 |" in scores.to_markdown()


def test_compare_pairs_the_shared_targets():
    rng = np.random.default_rng(1)
    targets = pd.DataFrame({"i": np.repeat(np.arange(20), 2), "k": 1, "y": rng.normal(size=40), "t": 1.0})
    targets["t"] = np.tile([1.0, 2.0], 20)
    good = leave_future_out({1.0: _draws(targets, np.zeros((50, 40)), np.ones((50, 40)))})
    poor = leave_future_out({1.0: _draws(targets, np.full((50, 40), 3.0), np.ones((50, 40)))})
    diff = good.compare(poor)
    row = diff[(diff["anchor"] == 1.0) & (diff["endpoint"] == "all")].iloc[0]
    expected = (good.pointwise["elpd"] - poor.pointwise["elpd"]).sum()
    assert row["elpd_diff"] == pytest.approx(expected)
    assert row["elpd_diff"] > 0 and row["se"] > 0


def test_anchor_draws_check_their_shapes():
    targets = pd.DataFrame({"i": [0], "k": [1], "y": [1.0]})
    with pytest.raises(ValueError, match="draws"):
        _draws(targets, np.zeros((3, 2)), np.ones((3, 2)))
    with pytest.raises(ValueError, match="columns"):
        _draws(targets.drop(columns="y"), np.zeros((3, 1)), np.ones((3, 1)))
    with pytest.raises(ValueError, match="anchor"):
        leave_future_out({})


def _params_frame(cohort, copies: int = 2) -> pd.DataFrame:
    """Write the true parameters and latents as CmdStan CSV columns, one row per copy."""
    th, lat = cohort.theta, cohort.latents
    row: dict[str, float] = {}
    for name, value in th.items():
        arr = np.asarray(value, float)
        if arr.ndim == 0:
            row[name] = float(arr)
        elif arr.ndim == 1:
            row.update({f"{name}.{j + 1}": float(v) for j, v in enumerate(arr)})
        else:
            for r in range(arr.shape[0]):
                row.update({f"{name}.{r + 1}.{c + 1}": float(arr[r, c]) for c in range(arr.shape[1])})
    for name in ("s0", "b_raw", "eps"):
        arr = np.asarray(lat[name], float)
        for r in range(arr.shape[0]):
            row.update({f"{name}.{r + 1}.{c + 1}": float(arr[r, c]) for c in range(arr.shape[1])})
    return pd.DataFrame([row] * copies)


@pytest.mark.parametrize("config", [ModelConfig.primary(), ModelConfig.a01_sensitivity()])
def test_observation_draws_rebuild_the_true_mean(config):
    """With almost no observation noise, an interior observation sits at its true mean."""
    rng = np.random.default_rng(5)
    theta = draw_prior(rng, 1, config.hyperparameters())
    span = np.array([b - a for a, b in (C.SCALE_BOUNDS[k] for k in C.ENDPOINTS)])
    theta["sigma_obs"] = 1e-5 * span
    cohort = simulate_cohort(12, seed=4, theta=theta, n_pathway=1, config=config)
    split = split_at_anchor(cohort.stan_data, list(range(12)), anchor_years=1.0)
    mu, sigma = observation_draws(_params_frame(cohort), split.targets)
    y = np.asarray(split.targets.data["y"], float)
    lo = np.asarray(split.targets.data["Lk"], float)[np.asarray(split.targets.data["obs_k"]) - 1]
    hi = np.asarray(split.targets.data["Uk"], float)[np.asarray(split.targets.data["obs_k"]) - 1]
    inside = (y > lo + 1e-3) & (y < hi - 1e-3)
    assert inside.sum() > 10
    np.testing.assert_allclose(mu[0, inside], y[inside], atol=1e-2)
    np.testing.assert_array_equal(mu[0], mu[1])
    k = np.asarray(split.targets.data["obs_k"]) - 1
    if config.state_dependent_sigma:
        ratio = sigma[0] / theta["sigma_obs"][k]
        assert np.all(ratio <= np.exp(np.abs(theta["kappa"][k])) + 1e-12)
        assert not np.allclose(ratio, 1.0)
    else:
        np.testing.assert_allclose(sigma[0], theta["sigma_obs"][k])
