"""The NumPyro likelihood must be the truncated Gaussian of the Methods, not a clipped Normal."""

import math

import numpy as np
import pytest

pytest.importorskip("numpyro")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from numpyro import handlers  # noqa: E402

from symphony_dt.inference_backend.likelihood import LOWER, UPPER, observe_bounded  # noqa: E402


def _truncnorm_logpdf(y, mu, sigma, lo, hi):
    phi = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))  # noqa: E731
    z = (y - mu) / sigma
    log_kernel = -0.5 * z * z - 0.5 * math.log(2.0 * math.pi) - math.log(sigma)
    return log_kernel - math.log(phi((hi - mu) / sigma) - phi((lo - mu) / sigma))


def _site_log_prob(mu, sigma, y, mask):
    def model():
        observe_bounded("y", mu, sigma, y, mask)

    tr = handlers.trace(model).get_trace()
    site = tr["y"]
    return site["fn"].log_prob(site["value"])


def test_matches_closed_form_at_floor_and_ceiling():
    mu = jnp.array([10.0, 29.0, 5.0])
    sigma = jnp.array([4.0, 1.5, 3.0])
    y = jnp.array([0.0, 30.0, 0.0])  # UPDRS floor, MoCA ceiling, SCOPA-AUT floor
    lp = np.asarray(_site_log_prob(mu, sigma, y, jnp.array([True, True, True])))
    expected = [_truncnorm_logpdf(float(y[k]), float(mu[k]), float(sigma[k]), LOWER[k], UPPER[k]) for k in range(3)]
    np.testing.assert_allclose(lp, expected, rtol=1e-4)


def test_truncation_differs_from_plain_normal_near_ceiling():
    # A MoCA of 30 with mean 29: the plain Normal understates the density by the
    # mass it places above 30.
    mu, sigma, y = 29.0, 3.0, 30.0
    trunc = _truncnorm_logpdf(y, mu, sigma, LOWER[1], UPPER[1])
    plain = -0.5 * ((y - mu) / sigma) ** 2 - 0.5 * math.log(2 * math.pi) - math.log(sigma)
    assert trunc - plain > 0.4


def test_missing_entries_add_nothing_and_keep_gradients_finite():
    sigma = jnp.array([4.0, 1.5, 3.0])
    y = jnp.array([[20.0, jnp.nan, 7.0]])
    mask = jnp.array([[True, False, True]])

    def total(mu):
        def model():
            observe_bounded("y", mu, sigma, y, mask)

        tr = handlers.trace(model).get_trace()
        return tr["y"]["fn"].log_prob(tr["y"]["value"]).sum()

    mu0 = jnp.array([[18.0, 25.0, 6.0]])
    g = jax.grad(total)(mu0)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(g[0, 1]) == 0.0
    full = float(total(mu0))
    observed_only = sum(
        _truncnorm_logpdf(float(y[0, k]), float(mu0[0, k]), float(sigma[k]), LOWER[k], UPPER[k]) for k in (0, 2)
    )
    assert full == pytest.approx(observed_only, rel=1e-4)
