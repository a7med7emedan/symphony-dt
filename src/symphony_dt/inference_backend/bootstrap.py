from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BootstrapStabilityConfig:
    """
    Methods: bootstrap sign stability for pathway effects A_{j,p}.

    - Refit scaffold on B=200 patient-level bootstrap resamples of the training set.
    - Require empirical sign frequency >= 0.90 for A_{j,p} to be used in narratives.

    This module computes sign frequencies from already-fitted bootstrap draw summaries.
    The actual refitting loop is done in Module 29/30 where inference is available at scale.
    """
    B: int = 200
    sign_freq_threshold: float = 0.90


def compute_sign_frequency(samples: np.ndarray) -> float:
    """Compute Pr(sample > 0) empirically."""
    s = np.asarray(samples, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float(np.mean(s > 0))


def pathway_sign_stability(
    bootstrap_A_samples: Dict[Tuple[int, int], np.ndarray],
    cfg: BootstrapStabilityConfig = BootstrapStabilityConfig(),
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Inputs:
      bootstrap_A_samples[(j,p)] = array of length B containing bootstrap point estimates
        (e.g., posterior means) for pathway effect A_{j,p}.

    Returns:
      dict keyed by (j,p) with:
        - sign_freq
        - stable (0/1)
    """
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for key, arr in bootstrap_A_samples.items():
        freq = compute_sign_frequency(arr)
        stable = float(freq >= cfg.sign_freq_threshold) if np.isfinite(freq) else float(0.0)
        out[key] = {"sign_freq": float(freq), "stable": float(stable)}
    return out
