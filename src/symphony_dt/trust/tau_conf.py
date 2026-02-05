from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TauConfSelectionConfig:
    """
    Methods 'Confidence gating and do-not-trust rules' — fold-specific tau_conf selection.

    Grid:
      tau in {0.05, 0.06, ..., 0.60}

    Objective:
      minimize | Cov_0.05(tau) - 0.95 |  subject to  Pr(Delta <= tau) >= 0.5

    Where:
      - Delta is per-anchor latent uncertainty (max IQR over x dimensions at anchor visit)
      - Cov_0.05(tau) is empirical 95% interval coverage for one-step-ahead predictions
        restricted to anchors with Delta <= tau, on the training fold.

    Inputs to selection are precomputed per-anchor:
      - delta: float
      - y_true: shape (N,3) for (U,Q,S) at target visit (one-step)
      - pi_low, pi_high: shape (N,3) predictive interval bounds at 95% for one-step
    """
    tau_grid: Tuple[float, ...] = tuple(np.round(np.arange(0.05, 0.601, 0.01), 2).tolist())
    target_coverage: float = 0.95
    min_fraction_covered: float = 0.50


@dataclass(frozen=True)
class TauConfSelectionResult:
    tau_conf: float
    achieved_coverage: float
    fraction_kept: float
    grid_diagnostics: Dict[str, float]  # keyed summaries for reproducibility


def _coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    # y, lo, hi shape (M,3). Coverage is proportion of componentwise-in-interval events averaged over all entries.
    inside = (y >= lo) & (y <= hi)
    return float(np.mean(inside))


def select_tau_conf(
    delta: np.ndarray,
    y_true: np.ndarray,
    pi_low: np.ndarray,
    pi_high: np.ndarray,
    cfg: TauConfSelectionConfig = TauConfSelectionConfig(),
) -> TauConfSelectionResult:
    """
    Select tau_conf per Methods. Returns best tau meeting min_fraction_covered constraint.

    If no tau meets the constraint, chooses tau that maximizes fraction_kept, and reports achieved coverage.
    """
    d = np.asarray(delta, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(pi_low, dtype=np.float64)
    hi = np.asarray(pi_high, dtype=np.float64)

    if y.shape != lo.shape or y.shape != hi.shape:
        raise ValueError("y_true, pi_low, pi_high must have same shape")
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError("y_true must be shape (N,3) for (U,Q,S)")
    if d.shape[0] != y.shape[0]:
        raise ValueError("delta length must match number of rows in y_true")

    N = y.shape[0]
    diagnostics: Dict[str, float] = {}

    best_tau: Optional[float] = None
    best_obj = float("inf")
    best_cov = float("nan")
    best_frac = 0.0

    feasible_found = False

    for tau in cfg.tau_grid:
        mask = d <= float(tau)
        frac = float(np.mean(mask)) if N > 0 else 0.0
        if frac <= 0:
            cov = float("nan")
        else:
            cov = _coverage(y[mask], lo[mask], hi[mask])

        diagnostics[f"frac_le_{tau:.2f}"] = frac
        if frac > 0:
            diagnostics[f"cov95_le_{tau:.2f}"] = cov

        if frac >= cfg.min_fraction_covered and frac > 0:
            feasible_found = True
            obj = abs(cov - cfg.target_coverage)
            # tie-break: prefer smaller tau if obj equal within 1e-6
            if (obj + 1e-6 < best_obj) or (abs(obj - best_obj) <= 1e-6 and (best_tau is None or tau < best_tau)):
                best_tau = float(tau)
                best_obj = float(obj)
                best_cov = float(cov)
                best_frac = float(frac)

    if feasible_found and best_tau is not None:
        return TauConfSelectionResult(
            tau_conf=best_tau,
            achieved_coverage=best_cov,
            fraction_kept=best_frac,
            grid_diagnostics=diagnostics,
        )

    # If no feasible tau, pick tau with maximal fraction kept (and smallest tau among ties), report coverage.
    max_frac = -1.0
    pick_tau = float(cfg.tau_grid[-1])
    pick_cov = float("nan")

    for tau in cfg.tau_grid:
        mask = d <= float(tau)
        frac = float(np.mean(mask)) if N > 0 else 0.0
        cov = _coverage(y[mask], lo[mask], hi[mask]) if frac > 0 else float("nan")
        if frac > max_frac + 1e-12 or (abs(frac - max_frac) <= 1e-12 and tau < pick_tau):
            max_frac = frac
            pick_tau = float(tau)
            pick_cov = float(cov)

    return TauConfSelectionResult(
        tau_conf=pick_tau,
        achieved_coverage=pick_cov,
        fraction_kept=float(max_frac),
        grid_diagnostics=diagnostics,
    )
