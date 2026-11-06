from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ConditioningRunDiagnostics:
    """
    Diagnostics for a single conditional run (i,n,ell) as in Methods.

    The sequential personalization step marks a run as failed if any of:
      - divergences after warm-up > 0
      - mean acceptance probability outside [0.7, 0.9]
      - Rhat > 1.01 for any component of r_i

    We store the required scalar fields so downstream filtering is deterministic.
    """
    divergences: int
    accept_prob_mean: float
    rhat_r_m: float
    rhat_r_c: float
    rhat_r_n: float

    def passes(self) -> bool:
        if self.divergences != 0:
            return False
        if not (0.7 <= self.accept_prob_mean <= 0.9):
            return False
        if max(self.rhat_r_m, self.rhat_r_c, self.rhat_r_n) > 1.01:
            return False
        return True


@dataclass(frozen=True)
class ConditionalRun:
    """
    Represents one successful/failed conditional posterior run for a specific:
      - patient i
      - conditioning visit index n
      - global draw index ell

    samples_x_tn:
      samples of x(t_{i,n}) = sigmoid(s(t_{i,n})) for this run.
      shape: (S, 3) where S is posterior sample count for that conditional run.

    This module does not implement NUTS; it defines the storage interface for later modules.
    """
    patient_id: int
    visit_index: int
    global_draw_index: int
    diagnostics: ConditioningRunDiagnostics
    samples_x_tn: np.ndarray  # (S,3)


@dataclass(frozen=True)
class ConditioningMixture:
    """
    Mixture over successful conditional runs at a conditioning visit (i,n).
    """
    patient_id: int
    visit_index: int
    l_cond: int
    successful_runs: List[ConditionalRun]

    def n_success(self) -> int:
        return len(self.successful_runs)

    def sample_x_pool(self) -> np.ndarray:
        """Concatenate x samples across successful runs."""
        if not self.successful_runs:
            return np.zeros((0, 3), dtype=np.float64)
        xs = [np.asarray(r.samples_x_tn, dtype=np.float64) for r in self.successful_runs]
        return np.concatenate(xs, axis=0)


def filter_successful_runs(
    runs: Sequence[ConditionalRun],
    l_cond: int,
) -> Tuple[List[ConditionalRun], int]:
    """
    Apply Methods failure handling and compute minimum required successful runs:
      min_pass = max(10, ceil(0.4 * L_cond))
    Returns (successful_runs, min_pass).
    """
    min_pass = max(10, int(np.ceil(0.4 * float(l_cond))))
    ok = [r for r in runs if r.diagnostics.passes()]
    return ok, min_pass
