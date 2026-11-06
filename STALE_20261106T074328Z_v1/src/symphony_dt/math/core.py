from __future__ import annotations

import math
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def sigmoid(x: ArrayLike) -> ArrayLike:
    if isinstance(x, (float, int)):
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logit(p: ArrayLike, eps: float = 1e-12) -> ArrayLike:
    if isinstance(p, (float, int)):
        pp = min(max(float(p), eps), 1.0 - eps)
        return math.log(pp / (1.0 - pp))
    p = np.asarray(p, dtype=np.float64)
    pp = np.clip(p, eps, 1.0 - eps)
    return np.log(pp / (1.0 - pp))


def softplus(x: ArrayLike) -> ArrayLike:
    if isinstance(x, (float, int)):
        xx = float(x)
        if xx > 50:
            return xx
        if xx < -50:
            return math.exp(xx)
        return math.log1p(math.exp(xx))

    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    hi = x > 50
    lo = x < -50
    mid = ~(hi | lo)
    out[hi] = x[hi]
    out[lo] = np.exp(x[lo])
    out[mid] = np.log1p(np.exp(x[mid]))
    return out
