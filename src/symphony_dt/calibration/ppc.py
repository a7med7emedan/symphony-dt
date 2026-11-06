"""Posterior predictive checks: PIT values, boundary statistics and interval coverage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from symphony_dt import constants as C
from symphony_dt.labels import validation

__all__ = ["boundary_statistics", "pit_values", "ppc_coverage", "ppc_summary"]


@validation("TESTED")
def pit_values(draws: ArrayLike, y: ArrayLike, *, seed: int = 0) -> np.ndarray:
    """Return randomised probability integral transform values, one per observation.

    PIT = P(X < y) + U x P(X = y), with U uniform, so ties at a scale boundary do not pile up.
    `draws` is (draws, observations). Uniform PIT values indicate calibration.

    Validation: TESTED.
    """
    x = np.asarray(draws, float)
    obs = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    below = (x < obs).mean(axis=0)
    equal = (x == obs).mean(axis=0)
    return np.asarray(below + rng.uniform(size=obs.size) * equal, float)


@validation("TESTED")
def boundary_statistics(
    draws: ArrayLike, y: ArrayLike, lower: float, upper: float, *, round_to_integer: bool = False
) -> dict[str, Any]:
    """Compare the observed share of values at each scale boundary with the replicated share.

    Returns the observed share, the replicated mean share and its 95% interval across draws, at the
    floor and at the ceiling.

    Validation: TESTED.
    """
    x = np.asarray(draws, float)
    obs = np.asarray(y, float)
    if round_to_integer:
        x = np.round(x)
    out: dict[str, Any] = {}
    for name, bound, below in (("floor", lower, True), ("ceiling", upper, False)):
        rep = (x <= bound).mean(axis=1) if below else (x >= bound).mean(axis=1)
        seen = float((obs <= bound).mean() if below else (obs >= bound).mean())
        lo, hi = np.quantile(rep, [0.025, 0.975])
        out[name] = {
            "observed": seen,
            "replicated_mean": float(rep.mean()),
            "replicated_95": [float(lo), float(hi)],
            "inside": bool(lo <= seen <= hi),
        }
    return out


@validation("TESTED")
def ppc_coverage(draws: ArrayLike, y: ArrayLike, levels: Sequence[float] = (0.5, 0.95)) -> dict[str, float]:
    """Return the share of observations inside the inclusive central predictive interval per level.

    Validation: TESTED.
    """
    x = np.asarray(draws, float)
    obs = np.asarray(y, float)
    out = {}
    for level in levels:
        tail = (1 - level) / 2
        lo, hi = np.quantile(x, [tail, 1 - tail], axis=0)
        out[f"{level:g}"] = float(((obs >= lo) & (obs <= hi)).mean())
    return out


@validation("TESTED")
def ppc_summary(
    draws: ArrayLike,
    stan_data: Mapping[str, Any],
    *,
    round_endpoints: Sequence[int] = (1, 2),
    pit_bins: int = 10,
) -> dict[str, Any]:
    """Summarise posterior predictive draws per endpoint: coverage, boundaries and a PIT histogram.

    `draws` is (draws, n_obs), for example `y_rep_c` from the GQ program run on the fitted data.
    Endpoints in `round_endpoints` are rounded to integers before the boundary comparison, as the
    observed scores are integers. Aggregates only.

    Validation: TESTED.
    """
    x = np.asarray(draws, float)
    y = np.asarray(stan_data["y"], float)
    ks = np.asarray(stan_data["obs_k"], int)
    lower, upper = stan_data["Lk"], stan_data["Uk"]
    out: dict[str, Any] = {}
    for k in (1, 2, 3):
        m = ks == k
        if not m.any():
            continue
        pit = pit_values(x[:, m], y[m])
        hist, _ = np.histogram(pit, bins=np.linspace(0, 1, pit_bins + 1))
        out[C.ENDPOINTS[k - 1]] = {
            "n": int(m.sum()),
            "coverage": ppc_coverage(x[:, m], y[m]),
            "boundaries": boundary_statistics(
                x[:, m], y[m], float(lower[k - 1]), float(upper[k - 1]), round_to_integer=k in round_endpoints
            ),
            "pit_histogram": [int(c) for c in hist],
        }
    return out
