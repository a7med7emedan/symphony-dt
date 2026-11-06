from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ConvergenceCriteria:
    """
    Methods-required convergence checks for scaffold posterior inference (offline):
      - Rhat <= 1.01
      - bulk ESS >= 400
      - divergent transitions after warm-up == 0
    """
    rhat_max: float = 1.01
    ess_bulk_min: int = 400
    divergences_max: int = 0


@dataclass(frozen=True)
class DiagnosticsReport:
    passed: bool
    failures: List[str]
    summary: Dict[str, Any]


def _collect_scalar_metrics(diagnostics: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    rhat = diagnostics.get("rhat", {}) if isinstance(diagnostics.get("rhat", {}), dict) else {}
    ess = diagnostics.get("ess_bulk", {}) if isinstance(diagnostics.get("ess_bulk", {}), dict) else {}
    rhat_f = {k: float(v) for k, v in rhat.items() if v is not None and np.isfinite(float(v))}
    ess_f = {k: float(v) for k, v in ess.items() if v is not None and np.isfinite(float(v))}
    return rhat_f, ess_f


def evaluate_convergence(
    diagnostics: Mapping[str, Any],
    criteria: ConvergenceCriteria = ConvergenceCriteria(),
    required_params: Optional[List[str]] = None,
) -> DiagnosticsReport:
    failures: List[str] = []
    summary: Dict[str, Any] = {}

    div = diagnostics.get("divergences", 0)
    try:
        div_i = int(div)
    except Exception:  # noqa: BLE001
        div_i = 0
        failures.append("diagnostics.divergences not parseable as int; treating as 0")
    summary["divergences"] = div_i
    if div_i > criteria.divergences_max:
        failures.append(f"divergences {div_i} > {criteria.divergences_max}")

    rhat, ess = _collect_scalar_metrics(diagnostics)
    summary["n_rhat"] = len(rhat)
    summary["n_ess_bulk"] = len(ess)

    params = required_params if required_params is not None else sorted(set(rhat.keys()) | set(ess.keys()))
    if not params:
        failures.append("no scalar diagnostics provided (rhat/ess_bulk empty)")

    bad_rhat = []
    bad_ess = []
    missing_rhat = []
    missing_ess = []

    for p in params:
        if p not in rhat:
            missing_rhat.append(p)
        else:
            if rhat[p] > criteria.rhat_max:
                bad_rhat.append((p, rhat[p]))
        if p not in ess:
            missing_ess.append(p)
        else:
            if ess[p] < criteria.ess_bulk_min:
                bad_ess.append((p, ess[p]))

    if missing_rhat:
        failures.append(f"missing rhat for {len(missing_rhat)} params (first 10): {missing_rhat[:10]}")
    if missing_ess:
        failures.append(f"missing ess_bulk for {len(missing_ess)} params (first 10): {missing_ess[:10]}")
    if bad_rhat:
        failures.append(f"rhat > {criteria.rhat_max} for {len(bad_rhat)} params (first 10): {bad_rhat[:10]}")
    if bad_ess:
        failures.append(f"ess_bulk < {criteria.ess_bulk_min} for {len(bad_ess)} params (first 10): {bad_ess[:10]}")

    ap = diagnostics.get("accept_prob_mean", None)
    if ap is not None:
        try:
            summary["accept_prob_mean"] = float(ap)
        except Exception:  # noqa: BLE001
            failures.append("accept_prob_mean not parseable")

    return DiagnosticsReport(passed=(len(failures) == 0), failures=failures, summary=summary)
