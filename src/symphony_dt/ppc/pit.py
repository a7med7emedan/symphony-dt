from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from symphony_dt.math.truncnorm import truncnorm_cdf
from symphony_dt.model.observation import SUPPORTS


def pit_values(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Methods: PIT under truncation.

    y, mu: shape (N,3) (U,Q,S)
    sigma: shape (3,)
    Returns u: (N,3) with u_k = F_trunc(y_k).
    """
    yy = np.asarray(y, dtype=np.float64)
    mm = np.asarray(mu, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64).reshape(3)
    if yy.shape != mm.shape or yy.shape[1] != 3:
        raise ValueError("y and mu must be (N,3)")
    out = np.zeros_like(yy, dtype=np.float64)
    L = np.array([SUPPORTS["U"].L, SUPPORTS["Q"].L, SUPPORTS["S"].L], dtype=np.float64)
    U = np.array([SUPPORTS["U"].U, SUPPORTS["Q"].U, SUPPORTS["S"].U], dtype=np.float64)
    for k in range(3):
        out[:, k] = truncnorm_cdf(yy[:, k], mm[:, k], sig[k], L[k], U[k])
    return out


def ks_distance_uniform(u: np.ndarray) -> float:
    """
    Kolmogorov–Smirnov distance between empirical CDF of u and Uniform(0,1).
    """
    x = np.asarray(u, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    x = np.sort(np.clip(x, 0.0, 1.0))
    n = x.size
    ecdf = np.arange(1, n + 1) / n
    d = np.max(np.abs(ecdf - x))
    return float(d)
