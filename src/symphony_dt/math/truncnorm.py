from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def _Phi(x: ArrayLike) -> ArrayLike:
    if isinstance(x, (float, int)):
        xx = float(x)
        return 0.5 * (1.0 + math.erf(xx / math.sqrt(2.0)))
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _inv_Phi(u: ArrayLike) -> ArrayLike:
    try:
        from scipy.stats import norm  # type: ignore
        if isinstance(u, (float, int)):
            return float(norm.ppf(float(u)))
        return norm.ppf(np.asarray(u, dtype=np.float64))
    except Exception:  # noqa: BLE001
        return _inv_Phi_acklam(u)


def _inv_Phi_acklam(u: ArrayLike) -> ArrayLike:
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow

    def _scalar(p: float) -> float:
        p = min(max(p, 1e-15), 1.0 - 1e-15)
        if p < plow:
            q = math.sqrt(-2.0 * math.log(p))
            num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
            den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
            return num / den
        if p > phigh:
            q = math.sqrt(-2.0 * math.log(1.0 - p))
            num = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
            den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
            return num / den
        q = p - 0.5
        r = q*q
        num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
        den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
        return num / den

    if isinstance(u, (float, int)):
        return _scalar(float(u))
    uu = np.asarray(u, dtype=np.float64)
    vec = np.vectorize(_scalar, otypes=[np.float64])
    return vec(uu)


@dataclass(frozen=True)
class TruncNorm:
    mu: float
    sigma: float
    a: float
    b: float

    def cdf(self, x: ArrayLike) -> ArrayLike:
        return truncnorm_cdf(x, self.mu, self.sigma, self.a, self.b)

    def ppf(self, q: ArrayLike) -> ArrayLike:
        return truncnorm_ppf(q, self.mu, self.sigma, self.a, self.b)

    def rvs(self, size: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return truncnorm_rvs(self.mu, self.sigma, self.a, self.b, size=size, rng=rng)


def truncnorm_cdf(x: ArrayLike, mu: float, sigma: float, a: float, b: float) -> ArrayLike:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    Z = _Phi(beta) - _Phi(alpha)

    def _cdf_scalar(xx: float) -> float:
        if xx <= a:
            return 0.0
        if xx >= b:
            return 1.0
        z = (xx - mu) / sigma
        return float((_Phi(z) - _Phi(alpha)) / Z)

    if isinstance(x, (float, int)):
        return _cdf_scalar(float(x))

    xx = np.asarray(x, dtype=np.float64)
    out = np.empty_like(xx, dtype=np.float64)
    out[xx <= a] = 0.0
    out[xx >= b] = 1.0
    mid = (xx > a) & (xx < b)
    z = (xx[mid] - mu) / sigma
    out[mid] = (_Phi(z) - _Phi(alpha)) / Z
    return out


def truncnorm_ppf(q: ArrayLike, mu: float, sigma: float, a: float, b: float) -> ArrayLike:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    Phi_a = _Phi(alpha)
    Phi_b = _Phi(beta)
    Z = Phi_b - Phi_a

    def _ppf_scalar(qq: float) -> float:
        qq = min(max(float(qq), 0.0), 1.0)
        u = Phi_a + qq * Z
        z = float(_inv_Phi(u))
        return float(mu + sigma * z)

    if isinstance(q, (float, int)):
        return _ppf_scalar(float(q))

    qq = np.asarray(q, dtype=np.float64)
    qq = np.clip(qq, 0.0, 1.0)
    u = Phi_a + qq * Z
    z = _inv_Phi(u)
    return mu + sigma * np.asarray(z, dtype=np.float64)


def truncnorm_rvs(mu: float, sigma: float, a: float, b: float, size: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    n = 1 if size is None else int(size)
    u = rng.uniform(0.0, 1.0, size=n)
    x = truncnorm_ppf(u, mu, sigma, a, b)
    return np.asarray(x, dtype=np.float64).reshape(n)
