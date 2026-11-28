"""PSIS-LOO and stacking: Pareto fits, an analytic LOO, ArviZ reference values and known weights."""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from symphony_dt.data import simulate_cohort
from symphony_dt.evaluate import (
    LFOScores,
    fit_generalized_pareto,
    fit_log_likelihood,
    loo_compare,
    observation_draws,
    pareto_k_threshold,
    predictive_log_density,
    pseudo_bma_weights,
    psis,
    psis_loo,
    relative_efficiency,
    stacking_weights,
)
from tests.test_lfo import _params_frame


@pytest.fixture(scope="module")
def eight_schools():
    """The centred eight schools posterior that ships with ArviZ: 4 chains of 500 draws."""
    return az.load_arviz_data("centered_eight")


@pytest.mark.parametrize("shape", [0.2, 0.5, 0.8])
def test_pareto_fit_recovers_a_known_shape_and_scale(shape):
    x = stats.genpareto.rvs(shape, scale=2.0, size=20000, random_state=np.random.default_rng(1))
    k, sigma = fit_generalized_pareto(x)
    assert k == pytest.approx(shape, abs=0.04)
    assert sigma == pytest.approx(2.0, rel=0.05)


def test_pareto_fit_shrinks_small_samples_towards_one_half():
    x = stats.genpareto.rvs(0.0, size=8, random_state=np.random.default_rng(2))
    k, _ = fit_generalized_pareto(x)
    raw = np.mean(np.log1p(x))  # crude reference: the prior holds more weight than 8 points
    assert abs(k - 0.5) < abs(raw - 0.5) + 0.5
    assert np.isnan(fit_generalized_pareto(np.r_[np.zeros(6), 1.0])[0])
    with pytest.raises(ValueError, match="at least 5"):
        fit_generalized_pareto([1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        fit_generalized_pareto([-1.0, 1, 2, 3, 4])


def test_k_threshold_follows_the_sample_size():
    assert pareto_k_threshold(2000) == pytest.approx(1 - 1 / np.log10(2000))
    assert pareto_k_threshold(100000) == 0.7
    with pytest.raises(ValueError):
        pareto_k_threshold(1)


def test_psis_keeps_the_order_caps_the_tail_and_normalises():
    rng = np.random.default_rng(3)
    lr = rng.standard_t(3, size=(4000, 3))
    out = psis(lr)
    np.testing.assert_allclose(out.weights.sum(axis=0), 1.0)
    for j in range(3):
        order = np.argsort(lr[:, j], kind="stable")
        assert np.all(np.diff(out.log_weights[order, j]) >= -1e-12)
        body = order[0]  # an untouched draw, so differences to it are on the raw scale
        gap = out.log_weights[:, j] - out.log_weights[body, j]
        assert gap.max() <= lr[:, j].max() - lr[body, j] + 1e-9
    assert np.all(out.n_eff > 0) and np.all(out.n_eff <= 4000)
    assert psis(lr[:, 0]).log_weights.shape == (4000, 1)


def test_psis_of_equal_ratios_gives_equal_weights():
    out = psis(np.zeros(1000))
    np.testing.assert_allclose(out.weights, 1 / 1000)
    assert out.n_eff[0] == pytest.approx(1000)


def test_psis_rejects_bad_input():
    with pytest.raises(ValueError, match="finite"):
        psis([0.0, np.inf, 1.0])
    with pytest.raises(ValueError, match="tail of at least 5"):
        psis(np.arange(10.0))
    with pytest.raises(ValueError, match="r_eff"):
        psis(np.zeros((100, 2)), r_eff=[1.0, 0.0])


def test_psis_matches_arviz_on_eight_schools(eight_schools):
    pytest.importorskip("arviz_stats")
    da = eight_schools.log_likelihood["obs"]
    ll = da.transpose("chain", "draw", "school").to_numpy().reshape(-1, 8)
    ref_lw, ref_k = da.azstats.psislw(r_eff=0.5, dim=["chain", "draw"])
    ours = psis(-ll, r_eff=0.5)
    np.testing.assert_allclose(ours.pareto_k, ref_k.to_numpy(), rtol=1e-10)
    ref = ref_lw.transpose("chain", "draw", "school").to_numpy().reshape(-1, 8)
    # Tied draws may swap places, so compare the sorted weights.
    np.testing.assert_allclose(np.sort(ours.log_weights, axis=0), np.sort(ref, axis=0), atol=1e-10)


def test_psis_loo_reproduces_the_arviz_reference_for_eight_schools(eight_schools):
    """ArviZ 1.x prints elpd_loo -30.76 (SE 1.43) and p_loo 0.95 for this posterior."""
    pytest.importorskip("arviz_stats")  # ArviZ 0.x ships different draws of this example
    ll = eight_schools.log_likelihood["obs"].to_numpy()
    r_eff = 0.241280875241094  # ArviZ's relative efficiency for this posterior, the mean ESS over draws
    result = psis_loo(ll, r_eff=r_eff)
    assert result.elpd_loo == pytest.approx(-30.76, abs=0.005)
    assert result.se == pytest.approx(1.43, abs=0.005)
    assert result.p_loo == pytest.approx(0.95, abs=0.005)
    assert result.looic == pytest.approx(-2 * result.elpd_loo)
    ref = az.loo(eight_schools, pointwise=True)
    np.testing.assert_allclose(result.pointwise["pareto_k"], ref.pareto_k.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(result.pointwise["elpd_loo"], ref.elpd_i.to_numpy(), rtol=1e-10)
    assert result.n_unreliable == 0
    assert result.k_table()["count"].sum() == 8
    assert "elpd_loo | -30.76" in result.to_markdown()


def test_psis_loo_matches_exact_leave_one_out_for_a_conjugate_normal():
    """With y_i ~ N(theta, 1) and a N(0, 10^2) prior, every leave-one-out predictive is known."""
    rng = np.random.default_rng(4)
    n, tau = 25, 10.0
    y = rng.normal(1.0, 1.0, n)
    prec = 1 / tau**2 + n
    theta = rng.normal(y.sum() / prec, 1 / np.sqrt(prec), 8000)
    ll = stats.norm.logpdf(y[None, :], theta[:, None], 1.0)
    result = psis_loo(ll)
    prec_i = prec - 1
    exact = stats.norm.logpdf(y, (y.sum() - y) / prec_i, np.sqrt(1 + 1 / prec_i))
    np.testing.assert_allclose(result.pointwise["elpd_loo"], exact, atol=0.03)  # Monte Carlo error
    assert result.p_loo == pytest.approx(1.0, abs=0.15)
    assert result.pointwise["reliable"].all()


def test_grouped_loo_sums_the_group_first():
    rng = np.random.default_rng(5)
    ll = rng.normal(-1.0, 0.3, size=(2000, 6))
    groups = np.array(["a", "a", "b", "b", "b", "c"])
    grouped = psis_loo(ll, groups=groups)
    summed = np.stack([ll[:, :2].sum(1), ll[:, 2:5].sum(1), ll[:, 5]], axis=1)
    plain = psis_loo(summed)
    assert list(grouped.pointwise["unit"]) == ["a", "b", "c"]
    np.testing.assert_allclose(grouped.pointwise["elpd_loo"], plain.pointwise["elpd_loo"])
    assert psis_loo(ll.reshape(2, 1000, 6)).elpd_loo == pytest.approx(psis_loo(ll).elpd_loo)
    with pytest.raises(ValueError, match="one label"):
        psis_loo(ll, groups=groups[:3])
    with pytest.raises(ValueError, match="one r_eff"):
        psis_loo(ll, groups=groups, r_eff=np.ones(6))


def test_relative_efficiency_is_one_for_independent_draws_and_follows_an_ar1_chain():
    """For AR(1) draws with coefficient phi the integrated autocorrelation time is (1 + phi) / (1 - phi)."""
    rng = np.random.default_rng(6)
    iid = rng.normal(size=(4, 1000, 3))
    assert relative_efficiency(iid) == pytest.approx(np.ones(3), abs=0.15)
    phi, n = 0.8, 20000
    z = np.zeros((2, n))
    noise = rng.normal(size=(2, n))
    for t in range(1, n):
        z[:, t] = phi * z[:, t - 1] + noise[:, t]
    # A likelihood linear in the chain has the autocorrelation of the chain itself.
    linear = relative_efficiency(np.log(z - z.min() + 1.0)[:, :, None])[0]
    assert linear == pytest.approx((1 - phi) / (1 + phi), rel=0.25)
    with pytest.raises(ValueError):
        relative_efficiency(iid[0])


def test_stacking_of_two_mirrored_points_is_even():
    lpd = {"a": np.log([0.9, 0.1]), "b": np.log([0.1, 0.9])}
    np.testing.assert_allclose(stacking_weights(lpd).to_numpy(), [0.5, 0.5], atol=1e-6)


def test_stacking_matches_the_closed_form_for_two_models():
    """For two points the optimum of log(w a1 + (1-w) b1) + log(w a2 + (1-w) b2) is closed form."""
    a, b = np.array([0.8, 0.3]), np.array([0.2, 0.6])
    d = a - b
    w_star = -(b[0] * d[1] + b[1] * d[0]) / (2 * d[0] * d[1])
    got = stacking_weights({"a": np.log(a), "b": np.log(b)})
    assert got["a"] == pytest.approx(w_star, abs=1e-6)


def test_stacking_puts_all_weight_on_a_dominant_model_and_matches_arviz(eight_schools):
    rng = np.random.default_rng(7)
    base = rng.normal(-1, 0.2, 40)
    weights = stacking_weights({"good": base + 0.5, "bad": base})
    assert weights["good"] == pytest.approx(1.0, abs=1e-6)
    azs = pytest.importorskip("arviz_stats")

    nc = az.load_arviz_data("non_centered_eight")
    ref = azs.compare({"c": eight_schools, "nc": nc})
    ours = stacking_weights(
        {
            "c": psis_loo(eight_schools.log_likelihood["obs"].to_numpy()),
            "nc": psis_loo(nc.log_likelihood["obs"].to_numpy()),
        }
    )
    assert ours["nc"] == pytest.approx(float(ref.loc["nc", "weight"]), abs=1e-4)


def test_stacking_weights_sum_to_one_and_ignore_row_order():
    rng = np.random.default_rng(8)
    lpd = rng.normal(size=(60, 3))
    scores = {f"m{j}": lpd[:, j] for j in range(3)}
    w = stacking_weights(scores)
    assert w.sum() == pytest.approx(1.0)
    perm = rng.permutation(60)
    w2 = stacking_weights({k: v[perm] for k, v in scores.items()})
    np.testing.assert_allclose(w, w2, atol=1e-6)
    with pytest.raises(ValueError, match="two models"):
        stacking_weights({"m0": lpd[:, 0]})


def test_pseudo_bma_weights_follow_the_totals():
    lpd = {"a": np.full(10, -1.0), "b": np.full(10, -1.1)}
    plain = pseudo_bma_weights(lpd, n_boot=0)
    assert plain["a"] == pytest.approx(1 / (1 + np.exp(-1.0)))
    boot = pseudo_bma_weights(lpd, n_boot=200, seed=1)
    assert boot["a"] == pytest.approx(plain["a"], abs=1e-9)  # constant differences survive reweighting
    assert boot.sum() == pytest.approx(1.0)


def test_compare_ranks_models_and_pairs_the_differences():
    rng = np.random.default_rng(9)
    base = rng.normal(-1, 0.5, 30)
    table = loo_compare({"worse": base - 0.1, "better": base})
    assert list(table.index) == ["better", "worse"]
    assert table.loc["worse", "elpd_diff"] == pytest.approx(-3.0)
    assert table.loc["worse", "se_diff"] == pytest.approx(0.0, abs=1e-9)
    assert table.loc["better", "se_diff"] == 0.0


def test_stacking_aligns_lfo_scores_on_shared_targets():
    rows = pd.DataFrame({"anchor": 1.0, "i": [0, 0, 1, 2], "k": [1, 2, 1, 1], "t": [1.5, 1.5, 2.0, 2.5]})
    one = LFOScores(pointwise=rows.assign(elpd=[-1.0, -2.0, -1.5, -0.5]), joint=pd.DataFrame(), summary=pd.DataFrame())
    two = LFOScores(
        pointwise=rows.iloc[:3].assign(elpd=[-1.2, -1.8, -1.0]), joint=pd.DataFrame(), summary=pd.DataFrame()
    )
    table = loo_compare({"one": one, "two": two})
    assert table.loc["one", "elpd"] == pytest.approx(-4.5)
    assert table.loc["two", "elpd"] == pytest.approx(-4.0)


def test_fit_log_likelihood_scores_the_model_density_at_its_own_data():
    cohort = simulate_cohort(10, seed=11)
    frame = _params_frame(cohort, copies=3)
    ll, table = fit_log_likelihood(frame, cohort.stan_data)
    d = cohort.stan_data.data
    assert ll.shape == (3, int(d["n_obs"]))
    mu, sigma = observation_draws(frame, cohort.stan_data)
    k = np.asarray(d["obs_k"], int) - 1
    expected = predictive_log_density(
        d["y"], mu, sigma, np.asarray(d["Lk"])[k], np.asarray(d["Uk"])[k], np.asarray(d["censored"])[k]
    )
    np.testing.assert_allclose(ll, expected)
    counts = table.groupby("i").size().reindex(range(10), fill_value=0).to_numpy()
    np.testing.assert_array_equal(counts, d["obs_count"])
    assert np.isfinite(ll).all()
