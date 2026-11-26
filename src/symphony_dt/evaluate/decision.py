"""Decision curve analysis for the event forecasts: net benefit across threshold probabilities.

Net benefit follows Vickers and Elkin (2006). At a threshold probability pt, a target is flagged when
its forecast event probability is at least pt. Net benefit is the share of true flags minus the
share of false flags weighted by pt / (1 - pt), the odds at which a false flag is judged as costly
as a missed event. Two reference strategies are always scored: flag every target ("all") and flag
none ("none"). These are the "treat all" and "treat none" curves of the decision curve literature.

Net benefit here describes forecasts against observed events. It does not show that acting on a flag
changes any outcome.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from symphony_dt import constants as C
from symphony_dt.evaluate.metrics import EVENT_THRESHOLDS, bootstrap_mean, cluster_bootstrap_weights
from symphony_dt.labels import validation

__all__ = ["DecisionCurve", "decision_curve", "net_benefit"]

#: Column of the per-target table that holds each method's event probability.
PROBABILITY_COLUMN: dict[str, str] = {"twin": "p_tw", "locf": "p_lo", "lmm": "p_lm"}
REFERENCE_STRATEGIES: tuple[str, str] = ("all", "none")
DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(float(x) for x in np.round(np.arange(0.01, 0.51, 0.01), 2))
_COLUMNS = (
    "endpoint",
    "k",
    "suppressed",
    "strategy",
    "threshold",
    "net_benefit",
    "net_benefit_lo",
    "net_benefit_hi",
    "n_targets",
    "n_participants",
    "event_rate",
)


def _weight(threshold: float) -> float:
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must lie in (0, 1), got {threshold}")
    return threshold / (1 - threshold)


def _row_benefit(flag: NDArray[np.bool_], e: NDArray[np.float64], threshold: float) -> NDArray[np.float64]:
    """Return each row's contribution to net benefit, so that net benefit is the row mean."""
    return np.where(flag, e - (1 - e) * _weight(threshold), 0.0)


@validation("TESTED")
def net_benefit(p: ArrayLike, e: ArrayLike, threshold: float) -> float:
    """Return the net benefit of flagging targets whose event probability is at least `threshold`.

    Net benefit is TP / n - FP / n * threshold / (1 - threshold), as in Vickers and Elkin (2006).
    `p` holds forecast event probabilities and `e` the observed events, 0 or 1.

    Validation: TESTED.
    """
    pa, ea = np.asarray(p, float), np.asarray(e, float)
    if pa.shape != ea.shape or pa.ndim != 1:
        raise ValueError("p and e must be one-dimensional and of equal length")
    if pa.size == 0:
        raise ValueError("p and e must not be empty")
    return float(np.mean(_row_benefit(pa >= threshold, ea, threshold)))


@validation("TESTED")
@dataclass(frozen=True)
class DecisionCurve:
    """Net benefit per endpoint, strategy and threshold probability.

    `table` has one row per endpoint, strategy and threshold. It holds the net benefit with its
    participant-cluster bootstrap interval, the target and participant counts and the observed event
    rate. Suppressed endpoints hold no counts or scores.

    Validation: TESTED.
    """

    table: pd.DataFrame
    min_cell: int

    def to_markdown(self, *, thresholds: Sequence[float] = (0.05, 0.1, 0.2, 0.3)) -> str:
        """Return a Markdown table of net benefit at selected thresholds, one column per strategy.

        Validation: TESTED.
        """
        strategies = list(dict.fromkeys(self.table["strategy"]))
        head = ["endpoint", "threshold", *strategies]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for endpoint, cell in self.table.groupby("endpoint", sort=False):
            if bool(cell["suppressed"].iloc[0]):
                lines.append("| " + " | ".join([str(endpoint), "", *["suppressed"] * len(strategies)]) + " |")
                continue
            for pt in thresholds:
                rows = cell[np.isclose(cell["threshold"].to_numpy(float), pt)].set_index("strategy")
                if rows.empty:
                    continue
                vals = [_fmt(rows, s) for s in strategies]
                lines.append("| " + " | ".join([str(endpoint), f"{pt:.2f}", *vals]) + " |")
        return "\n".join(lines) + "\n"

    def plot(self, *, endpoint: str | None = None, ax: Any = None) -> Any:
        """Plot net benefit against threshold probability; see `plots.plot_decision_curve`.

        Validation: TESTED.
        """
        from symphony_dt.plots import plot_decision_curve

        return plot_decision_curve(self, endpoint=endpoint, ax=ax)


def _fmt(rows: pd.DataFrame, strategy: str) -> str:
    if strategy not in rows.index:
        return ""
    est, lo, hi = rows.loc[strategy, ["net_benefit", "net_benefit_lo", "net_benefit_hi"]].to_numpy(float)
    if not np.isfinite(est):
        return ""
    return f"{est:.3f} [{lo:.3f}, {hi:.3f}]"


@validation("TESTED")
def decision_curve(
    target_scores: pd.DataFrame,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    methods: Sequence[str] = ("twin", "locf", "lmm"),
    min_cell: int = 10,
    bootstrap_replicates: int = 1000,
    seed: int = 20261021,
) -> DecisionCurve:
    """Score net benefit of the event forecasts across threshold probabilities, per endpoint.

    `target_scores` is `FoldScores.target_scores` or the pooled per-target table. It needs the
    columns `i`, `k` and `ev`, and one probability column per method: `p_tw`, `p_lo` or `p_lm`.
    Only rows with an observed event and a finite probability for every method are scored, so all
    strategies share the same targets. The references "all" and "none" are always included.

    Intervals come from a participant-cluster bootstrap. The same resamples serve every strategy
    and threshold of an endpoint, so the curves can be compared pointwise. An endpoint is
    suppressed when it has fewer than `min_cell` participants, events or non-events.

    Validation: TESTED.
    """
    frame = target_scores.reset_index(drop=True)
    unknown = [m for m in methods if m not in PROBABILITY_COLUMN]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {list(PROBABILITY_COLUMN)}")
    needed = ["i", "k", "ev", *[PROBABILITY_COLUMN[m] for m in methods]]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"target table needs columns {missing}")
    pts = np.asarray(thresholds, float)
    if pts.ndim != 1 or pts.size == 0 or not np.all((pts > 0) & (pts < 1)):
        raise ValueError("thresholds must be a non-empty list of values in (0, 1)")
    if min_cell < 1:
        raise ValueError("min_cell must be at least 1")

    strategies = [*methods, *REFERENCE_STRATEGIES]
    rows: list[dict[str, Any]] = []
    for k in sorted(EVENT_THRESHOLDS):
        sub = frame[frame["k"] == k].dropna(subset=needed[2:])
        if sub.empty:
            continue
        endpoint = C.ENDPOINTS[k - 1]
        e = sub["ev"].to_numpy(float)
        n_part = int(sub["i"].nunique())
        n_events = int(e.sum())
        small = min(n_part, n_events, len(e) - n_events) < min_cell
        base = {"endpoint": endpoint, "k": k, "suppressed": small}
        if small:
            rows.extend(
                {
                    **base,
                    "strategy": s,
                    "threshold": float(pt),
                    "net_benefit": np.nan,
                    "net_benefit_lo": np.nan,
                    "net_benefit_hi": np.nan,
                    "n_targets": np.nan,
                    "n_participants": np.nan,
                    "event_rate": np.nan,
                }
                for s in strategies
                for pt in pts
            )
            continue
        _, cluster = np.unique(sub["i"].to_numpy(), return_inverse=True)
        W = cluster_bootstrap_weights(n_part, B=bootstrap_replicates, seed=seed)
        counts = {"n_targets": len(e), "n_participants": n_part, "event_rate": round(float(e.mean()), 4)}
        for s in strategies:
            p = sub[PROBABILITY_COLUMN[s]].to_numpy(float) if s in PROBABILITY_COLUMN else None
            for pt in pts:
                if s == "none":
                    est = [0.0, 0.0, 0.0]
                else:
                    flag = np.ones(len(e), bool) if p is None else p >= pt
                    est = bootstrap_mean(_row_benefit(flag, e, float(pt)), cluster, W)
                rows.append(
                    {
                        **base,
                        "strategy": s,
                        "threshold": float(pt),
                        "net_benefit": est[0],
                        "net_benefit_lo": est[1],
                        "net_benefit_hi": est[2],
                        **counts,
                    }
                )
    return DecisionCurve(table=pd.DataFrame(rows, columns=list(_COLUMNS)), min_cell=min_cell)
