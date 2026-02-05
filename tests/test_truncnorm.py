import numpy as np
from symphony_dt.math.truncnorm import truncnorm_cdf, truncnorm_ppf, truncnorm_rvs


def test_truncnorm_cdf_bounds() -> None:
    mu, sigma, a, b = 0.0, 1.0, -1.0, 1.0
    assert truncnorm_cdf(-2.0, mu, sigma, a, b) == 0.0
    assert truncnorm_cdf(2.0, mu, sigma, a, b) == 1.0


def test_truncnorm_ppf_inverts_cdf_mid() -> None:
    mu, sigma, a, b = 0.0, 1.0, -1.0, 1.0
    qs = np.array([0.1, 0.5, 0.9])
    xs = truncnorm_ppf(qs, mu, sigma, a, b)
    qs2 = truncnorm_cdf(xs, mu, sigma, a, b)
    assert np.allclose(qs, qs2, atol=1e-6)


def test_truncnorm_rvs_in_bounds() -> None:
    mu, sigma, a, b = 10.0, 2.0, 8.0, 11.0
    x = truncnorm_rvs(mu, sigma, a, b, size=1000)
    assert np.all(x >= a - 1e-12)
    assert np.all(x <= b + 1e-12)
