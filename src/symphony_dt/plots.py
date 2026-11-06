"""Figures: trajectories with intervals, coverage by window, CRPS differences, ranks, PIT and effects.

Every function returns a matplotlib Axes and draws nothing outside it. Colours come from the
colour-blind-safe Okabe-Ito palette.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from symphony_dt import constants as C
from symphony_dt.labels import validation

__all__ = [
    "PALETTE",
    "plot_coverage_by_window",
    "plot_crps_differences",
    "plot_effects",
    "plot_forecast",
    "plot_pit",
    "plot_rank_histogram",
    "plot_trajectory",
]

#: Okabe-Ito colours, safe for the common forms of colour blindness.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
}
_ENDPOINT_COLOUR = {1: PALETTE["blue"], 2: PALETTE["vermillion"], 3: PALETTE["green"]}
_ENDPOINT_LABEL = {1: "MDS-UPDRS III", 2: "MoCA", 3: "SCOPA-AUT"}


def _axes(ax: Any) -> Any:
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, new = plt.subplots(figsize=(6, 4))
    return new


@validation("TESTED")
def plot_trajectory(
    dataset: Any, participant_id: str, *, forecast: Any = None, endpoint: str = "updrs3", ax: Any = None
) -> Any:
    """Plot one participant's observed scores and, optionally, forecast intervals after them.

    `forecast` is a Forecast whose targets carry `participant_id` (fast route) or `i` and `t`.

    Validation: TESTED.
    """
    ax = _axes(ax)
    k = C.ENDPOINT_INDEX[endpoint]
    obs = dataset.observations
    obs = obs[(obs["participant_id"] == str(participant_id)) & (obs["endpoint"] == endpoint)]
    t = obs["time_days"].to_numpy(float) / C.DAYS_PER_YEAR
    ax.plot(t, obs["value"], "o-", color=_ENDPOINT_COLOUR[k], label="observed")
    if forecast is not None:
        tg = forecast.targets.reset_index(drop=True)
        sel = tg.index[(tg["k"] == k)]
        if "participant_id" in tg.columns:
            sel = tg.index[(tg["k"] == k) & (tg["participant_id"] == str(participant_id))]
        if len(sel):
            last = float(t.max()) if t.size else 0.0
            x = last + tg.loc[sel, "horizon"].to_numpy(float) if "horizon" in tg else tg.loc[sel, "t"]
            d = forecast.draws[:, sel]
            for lo, hi, alpha in ((0.025, 0.975, 0.2), (0.25, 0.75, 0.4)):
                ql, qh = np.quantile(d, [lo, hi], axis=0)
                ax.fill_between(x, ql, qh, color=_ENDPOINT_COLOUR[k], alpha=alpha, linewidth=0)
            ax.plot(x, np.median(d, axis=0), "--", color=_ENDPOINT_COLOUR[k], label="forecast median")
    lo_b, hi_b = C.SCALE_BOUNDS[endpoint]
    ax.set_ylim(lo_b, hi_b)
    ax.set_xlabel("years since baseline")
    ax.set_ylabel(_ENDPOINT_LABEL[k])
    ax.legend(frameon=False)
    return ax


@validation("TESTED")
def plot_forecast(forecast: Any, *, participant: int | None = None, ax: Any = None) -> Any:
    """Plot central 50% and 95% forecast intervals against horizon, one colour per endpoint.

    Validation: TESTED.
    """
    ax = _axes(ax)
    tg = forecast.targets.reset_index(drop=True)
    if participant is not None and "i" in tg.columns:
        tg = tg[tg["i"] == participant]
    for k in sorted(tg["k"].unique()):
        sel = tg.index[tg["k"] == k]
        x = tg.loc[sel, "horizon"].to_numpy(float)
        order = np.argsort(x)
        d = forecast.draws[:, sel][:, order]
        q = np.quantile(d, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
        ax.fill_between(x[order], q[0], q[4], color=_ENDPOINT_COLOUR[int(k)], alpha=0.2, linewidth=0)
        ax.fill_between(x[order], q[1], q[3], color=_ENDPOINT_COLOUR[int(k)], alpha=0.4, linewidth=0)
        ax.plot(x[order], q[2], color=_ENDPOINT_COLOUR[int(k)], label=_ENDPOINT_LABEL[int(k)])
        if "y" in tg.columns:
            ax.plot(
                x[order],
                tg.loc[sel, "y"].to_numpy(float)[order],
                "o",
                color=_ENDPOINT_COLOUR[int(k)],
                markersize=3,
            )
    ax.set_xlabel("years after anchor")
    ax.set_ylabel("score")
    ax.legend(frameon=False)
    return ax


@validation("TESTED")
def plot_coverage_by_window(scores: Mapping[str, Any], *, level: str = "c95", ax: Any = None) -> Any:
    """Plot interval coverage with bootstrap intervals per endpoint, window and method.

    `scores` is FoldScores.scores or PooledScores.scores. The nominal level is drawn as a line.

    Validation: TESTED.
    """
    ax = _axes(ax)
    cells = scores["cells"]
    keys = [k for k in cells if all(m in cells[k] for m in ("twin", "locf", "lmm"))]
    methods = (("twin", PALETTE["blue"]), ("locf", PALETTE["orange"]), ("lmm", PALETTE["green"]))
    for j, (method, colour) in enumerate(methods):
        est = np.array([cells[k][method][level] for k in keys], float)
        x = np.arange(len(keys)) + (j - 1) * 0.25
        ax.errorbar(
            x,
            est[:, 0],
            yerr=[est[:, 0] - est[:, 1], est[:, 2] - est[:, 0]],
            fmt="o",
            color=colour,
            label=method,
            capsize=2,
        )
    ax.axhline(0.95 if level == "c95" else 0.5, color=PALETTE["black"], linewidth=0.8, linestyle=":")
    ax.set_xticks(np.arange(len(keys)), keys, rotation=60, ha="right")
    ax.set_ylabel(f"{'95' if level == 'c95' else '50'}% interval coverage")
    ax.legend(frameon=False)
    return ax


@validation("TESTED")
def plot_crps_differences(scores: Mapping[str, Any], *, ax: Any = None) -> Any:
    """Plot paired CRPS differences (twin minus LOCF, twin minus LMM) with bootstrap intervals.

    Negative values favour the twin.

    Validation: TESTED.
    """
    ax = _axes(ax)
    cells = scores["cells"]
    keys = [k for k in cells if "crps_diff_twin_minus_locf" in cells[k]]
    for j, (field, colour, label) in enumerate(
        (
            ("crps_diff_twin_minus_locf", PALETTE["orange"], "twin minus LOCF"),
            ("crps_diff_twin_minus_lmm", PALETTE["green"], "twin minus LMM"),
        )
    ):
        est = np.array([cells[k][field] for k in keys], float)
        y = np.arange(len(keys)) + (j - 0.5) * 0.3
        ax.errorbar(
            est[:, 0],
            y,
            xerr=[est[:, 0] - est[:, 1], est[:, 2] - est[:, 0]],
            fmt="o",
            color=colour,
            label=label,
            capsize=2,
        )
    ax.axvline(0, color=PALETTE["black"], linewidth=0.8)
    ax.set_yticks(np.arange(len(keys)), keys)
    ax.set_xlabel("CRPS difference (negative favours the twin)")
    ax.legend(frameon=False)
    return ax


@validation("TESTED")
def plot_rank_histogram(
    ranks: Sequence[int],
    n_draws: int,
    *,
    bins: int = 20,
    level: float = 0.99,
    title: str = "",
    ax: Any = None,
) -> Any:
    """Plot an SBC rank histogram with the binomial band of a uniform histogram.

    Validation: TESTED.
    """
    from symphony_dt.calibration.sbc import rank_histogram_bands

    ax = _axes(ax)
    edges = np.linspace(0, n_draws + 1, bins + 1)
    counts, _ = np.histogram(np.asarray(ranks), bins=edges)
    lo, hi = rank_histogram_bands(len(ranks), bins, level)
    ax.axhspan(lo, hi, color=PALETTE["sky"], alpha=0.3, linewidth=0)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=PALETTE["blue"], edgecolor="white")
    ax.axhline(len(ranks) / bins, color=PALETTE["black"], linewidth=0.8)
    ax.set_xlabel("rank")
    ax.set_ylabel("count")
    if title:
        ax.set_title(title)
    return ax


@validation("TESTED")
def plot_pit(pit: Sequence[float], *, bins: int = 10, ax: Any = None) -> Any:
    """Plot a PIT histogram; a flat histogram indicates calibrated predictive distributions.

    Validation: TESTED.
    """
    ax = _axes(ax)
    values = np.asarray(pit, float)
    ax.hist(values, bins=np.linspace(0, 1, bins + 1), color=PALETTE["blue"], edgecolor="white")
    ax.axhline(values.size / bins, color=PALETTE["black"], linewidth=0.8)
    ax.set_xlabel("PIT")
    ax.set_ylabel("count")
    return ax


@validation("TESTED")
def plot_effects(
    draws: pd.DataFrame | Mapping[str, Sequence[float]], *, names: Sequence[str] | None = None, ax: Any = None
) -> Any:
    """Plot posterior medians with 50% and 95% intervals for named parameters (a forest plot).

    Validation: TESTED.
    """
    ax = _axes(ax)
    frame = pd.DataFrame(draws) if not isinstance(draws, pd.DataFrame) else draws
    cols = list(names) if names is not None else [c for c in frame.columns if not c.endswith("__")]
    q = np.quantile(frame[cols].to_numpy(float), [0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
    y = np.arange(len(cols))
    ax.hlines(y, q[0], q[4], color=PALETTE["sky"], linewidth=1.5)
    ax.hlines(y, q[1], q[3], color=PALETTE["blue"], linewidth=4)
    ax.plot(q[2], y, "o", color=PALETTE["black"], markersize=3)
    ax.axvline(0, color=PALETTE["black"], linewidth=0.6, linestyle=":")
    ax.set_yticks(y, cols)
    ax.set_xlabel("posterior value")
    return ax
