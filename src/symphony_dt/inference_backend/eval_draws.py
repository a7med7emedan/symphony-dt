from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def crps_from_draws(draws: np.ndarray, y: np.ndarray) -> float:
    """
    CRPS estimate from samples (energy score for 1D), averaged over observations.
    draws: (S,N) or (N,S)
    y: (N,)
    """
    D = np.asarray(draws, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError("draws must be 2D")
    if D.shape[0] != y.shape[0]:
        D = D.T
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    if D.shape[0] != yy.shape[0]:
        raise ValueError("draws and y incompatible")

    # CRPS = E|X - y| - 0.5 E|X - X'|
    # Monte Carlo estimate
    e1 = np.mean(np.abs(D - yy[:, None]))
    # pairwise term: sample two independent draws via random pairing
    rng = np.random.default_rng(0)
    idx1 = rng.integers(0, D.shape[1], size=D.shape[0])
    idx2 = rng.integers(0, D.shape[1], size=D.shape[0])
    e2 = np.mean(np.abs(D[np.arange(D.shape[0]), idx1] - D[np.arange(D.shape[0]), idx2]))
    return float(e1 - 0.5 * e2)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    no_coupling: bool = False
    no_omics: bool = False
    linear_link: bool = False
