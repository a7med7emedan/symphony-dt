from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from symphony_dt.model.observation import SUPPORTS
from symphony_dt.personalization.conditioning import ConditioningMixture


@dataclass(frozen=True)
class TrustConfig:
    """
    Implements Methods 'Confidence gating and do-not-trust rules'.

    tau_conf is fold-specific and will be selected on training fold later.
    Here we provide the decision logic once tau_conf is known.

    Suppress if:
      1) fewer than two clinic visits exist for the participant;
      2) any of U,Q,S missing at conditioning visit;
      3) Delta_i,n > tau_conf;
      4) fewer than max(10, ceil(0.4*L_cond)) conditional runs pass diagnostics;
      5) any conditioning outcome equals its boundary => only intervals shown, no threshold event probs.
    """
    tau_conf: float


@dataclass(frozen=True)
class TrustDecision:
    allow_forecasts: bool
    allow_threshold_events: bool
    reasons: Dict[str, str]
    delta_iqr: Optional[float] = None
    n_success: Optional[int] = None
    min_required_success: Optional[int] = None


def compute_delta_iqr(x_samples: np.ndarray) -> float:
    """
    Delta_{i,n} = max_j IQR(x_j(t_{i,n})) across (M,C,N).

    x_samples: shape (S,3)
    """
    xs = np.asarray(x_samples, dtype=np.float64)
    if xs.ndim != 2 or xs.shape[1] != 3:
        raise ValueError("x_samples must be shape (S,3)")
    if xs.shape[0] == 0:
        return float("inf")
    q75 = np.quantile(xs, 0.75, axis=0)
    q25 = np.quantile(xs, 0.25, axis=0)
    iqr = q75 - q25
    return float(np.max(iqr))


def _is_at_boundary(y: float, L: float, U: float) -> bool:
    return (y == L) or (y == U)


def decide_trust_and_suppression(
    n_visits_total: int,
    missing_at_anchor: Tuple[bool, bool, bool],  # (miss_U, miss_Q, miss_S)
    anchor_values: Tuple[Optional[float], Optional[float], Optional[float]],  # (U,Q,S)
    mixture: ConditioningMixture,
    cfg: TrustConfig,
    min_required_success: int,
) -> TrustDecision:
    reasons: Dict[str, str] = {}

    if n_visits_total < 2:
        reasons["rule_1"] = "fewer_than_two_visits"
        return TrustDecision(False, False, reasons)

    miss_u, miss_q, miss_s = missing_at_anchor
    if miss_u or miss_q or miss_s:
        reasons["rule_2"] = "missing_anchor_outcome"
        return TrustDecision(False, False, reasons)

    # compute Delta from mixture x samples
    x_pool = mixture.sample_x_pool()
    delta = compute_delta_iqr(x_pool)
    if delta > cfg.tau_conf:
        reasons["rule_3"] = "latent_uncertainty_above_tau_conf"
        return TrustDecision(False, False, reasons, delta_iqr=delta, n_success=mixture.n_success(), min_required_success=min_required_success)

    if mixture.n_success() < min_required_success:
        reasons["rule_4"] = "insufficient_successful_conditioning_runs"
        return TrustDecision(False, False, reasons, delta_iqr=delta, n_success=mixture.n_success(), min_required_success=min_required_success)

    # boundary rule: allow forecasts, suppress threshold events
    u, q, s = anchor_values
    allow_events = True
    if u is not None and _is_at_boundary(u, SUPPORTS["U"].L, SUPPORTS["U"].U):
        allow_events = False
        reasons["rule_5_U"] = "anchor_at_boundary"
    if q is not None and _is_at_boundary(q, SUPPORTS["Q"].L, SUPPORTS["Q"].U):
        allow_events = False
        reasons["rule_5_Q"] = "anchor_at_boundary"
    if s is not None and _is_at_boundary(s, SUPPORTS["S"].L, SUPPORTS["S"].U):
        allow_events = False
        reasons["rule_5_S"] = "anchor_at_boundary"

    return TrustDecision(True, allow_events, reasons, delta_iqr=delta, n_success=mixture.n_success(), min_required_success=min_required_success)
