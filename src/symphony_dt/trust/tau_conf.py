from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TauConfSelectionConfig:
    """
    Methods 'Confidence gating and do-not-trust rules': fold-specific tau_conf selection.

    Grid:
      tau in {0.05, 0.06, ..., 0.60}

    Objective:
      minimise | Cov(tau) - target_coverage |  subject to  Pr(Delta <= tau) >= min_fraction_covered

    Where:
      - Delta is per-anchor latent uncertainty (max IQR over x dimensions at the anchor visit);
      - Cov(tau) is the empirical coverage of the one-step central 90% predictive interval
        [p05, p95], restricted to anchors with Delta <= tau, on the training fold.

    The target equals the nominal level of the interval actually produced. Every forecast
    engine writes the 5th and 95th percentiles, so the target is 0.90.

    Inputs to selection are precomputed per anchor:
      - delta: float
      - y_true: shape (N,3) for (U,Q,S) at the target visit (one-step), NaN where missing
      - pi_low, pi_high: shape (N,3) bounds of the central 90% predictive interval
    """
    tau_grid: Tuple[float, ...] = tuple(np.round(np.arange(0.05, 0.601, 0.01), 2).tolist())
    target_coverage: float = 0.90
    min_fraction_covered: float = 0.50


@dataclass(frozen=True)
class TauConfSelectionResult:
    tau_conf: float
    achieved_coverage: float
    fraction_kept: float
    grid_diagnostics: Dict[str, float]  # keyed summaries for reproducibility
    # False when no tau kept min_fraction_covered of anchors and the fallback rule was used.
    feasible: bool = True


def _coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    # y, lo, hi shape (M,3). Coverage is the proportion of componentwise-in-interval events,
    # pooled over all finite (anchor, endpoint) entries. Missing targets are skipped, not counted as misses.
    finite = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not finite.any():
        return float("nan")
    inside = (y[finite] >= lo[finite]) & (y[finite] <= hi[finite])
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

    Ties within 1e-6 of the objective go to the smaller tau.

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
            diagnostics[f"cov90_le_{tau:.2f}"] = cov

        if frac >= cfg.min_fraction_covered and frac > 0 and np.isfinite(cov):
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
        feasible=False,
    )


def _finite_or_none(v: float) -> Optional[float]:
    f = float(v)
    return f if np.isfinite(f) else None


def selection_report(
    result: TauConfSelectionResult,
    cfg: TauConfSelectionConfig = TauConfSelectionConfig(),
    **context: Any,
) -> Dict[str, Any]:
    """JSON-ready record of a selection: the chosen tau_conf, its grid and the run context.

    Non-finite values become None, so the file stays valid JSON.
    """
    grid: List[Dict[str, Any]] = []
    for tau in cfg.tau_grid:
        frac = result.grid_diagnostics.get(f"frac_le_{tau:.2f}", float("nan"))
        cov = result.grid_diagnostics.get(f"cov90_le_{tau:.2f}", float("nan"))
        grid.append({"tau": float(tau), "fraction_kept": _finite_or_none(frac), "coverage": _finite_or_none(cov)})
    return {
        "tau_conf": float(result.tau_conf),
        "feasible": bool(result.feasible),
        "achieved_coverage": _finite_or_none(result.achieved_coverage),
        "fraction_kept": _finite_or_none(result.fraction_kept),
        "target_coverage": float(cfg.target_coverage),
        "interval": "central 90% (5th to 95th percentile)",
        "min_fraction_kept": float(cfg.min_fraction_covered),
        **context,
        "grid": grid,
    }


def load_tau_conf_json(path: Path) -> Dict[str, Any]:
    """Read a file written by `symphony-dt nuts select-tau-conf` and check its tau_conf."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tau = data.get("tau_conf") if isinstance(data, dict) else None
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not (0.0 < float(tau) <= 1.0):
        raise ValueError(f"{path}: tau_conf must be a number in (0, 1], got {tau!r}")
    return data
