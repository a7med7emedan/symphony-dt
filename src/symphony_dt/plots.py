"""Figures: trajectories with intervals, coverage by window, CRPS differences, ranks, PIT and effects.

Every function returns a matplotlib Axes and draws nothing outside it. Colours come from the
colour-blind-safe Okabe-Ito palette.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from symphony_dt import constants as C
from symphony_dt.labels import validation

__all__ = [
    "PALETTE",
    "participant_report",
    "plot_attrition",
    "plot_cohort_trajectories",
    "plot_coverage_by_window",
    "plot_coverage_calibration",
    "plot_crps_differences",
    "plot_effects",
    "plot_energy",
    "plot_forecast",
    "plot_pit",
    "plot_ppc_intervals",
    "plot_profile",
    "plot_rank",
    "plot_rank_histogram",
    "plot_reliability",
    "plot_trace",
    "plot_trajectory",
    "save_figure",
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


# --------------------------------------------------------------------------------------------------
# Modeller plots (2.1)
# --------------------------------------------------------------------------------------------------

_CHAIN_COLOURS = [PALETTE[k] for k in ("blue", "orange", "green", "vermillion", "sky", "purple", "black", "yellow")]


@validation("TESTED")
def save_figure(fig: Any, path: str | Any, formats: Sequence[str] = ("png", "pdf", "svg"), dpi: int = 200) -> list[str]:
    """Save a figure in several formats at once and return the written paths.

    `path` is given without suffix; each format adds its own.

    Validation: TESTED.
    """
    from pathlib import Path

    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        target = base.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        written.append(str(target))
    return written


@validation("TESTED")
def plot_profile(profile: Any) -> Any:
    """Four panels: visits per participant, intervals, endpoint coverage and boundary shares.

    Validation: TESTED.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].hist(
        profile.visits_per_participant,
        bins=range(1, int(profile.visits_per_participant.max()) + 2),
        color=PALETTE["blue"],
        edgecolor="white",
    )
    axes[0, 0].set_xlabel("visits per participant")
    axes[0, 1].hist(profile.intervals_years, bins=30, color=PALETTE["sky"], edgecolor="white")
    axes[0, 1].set_xlabel("years between visits")
    ep = profile.endpoints
    x = np.arange(len(ep))
    axes[1, 0].bar(x, ep["share_of_visits_scored"], color=[_ENDPOINT_COLOUR[i + 1] for i in range(len(ep))])
    axes[1, 0].set_xticks(x, ep["endpoint"])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("share of visits scored")
    axes[1, 1].bar(x - 0.2, ep["share_at_floor"], width=0.4, color=PALETTE["blue"], label="at floor")
    axes[1, 1].bar(x + 0.2, ep["share_at_ceiling"], width=0.4, color=PALETTE["orange"], label="at ceiling")
    axes[1, 1].set_xticks(x, ep["endpoint"])
    axes[1, 1].set_ylabel("share of observations")
    axes[1, 1].legend(frameon=False)
    fig.tight_layout()
    return fig


@validation("TESTED")
def plot_attrition(result: Any, *, min_cell: int = 10, ax: Any = None) -> Any:
    """Draw the eligibility flow: one box per step with the count remaining and removed.

    Validation: TESTED.
    """
    ax = _axes(ax)
    table = result.attrition_table(min_cell=min_cell)
    n = len(table)
    for i, row in enumerate(table.itertuples()):
        y = n - i
        ax.add_patch(
            __import__("matplotlib").patches.FancyBboxPatch(
                (0.05, y - 0.35), 0.55, 0.7, boxstyle="round,pad=0.02", fc="#F3F5F9", ec="#C8CDD6"
            )
        )
        ax.text(0.08, y + 0.1, str(row.rule), fontsize=9, va="center")
        ax.text(
            0.08,
            y - 0.17,
            f"remaining {row.remaining}",
            fontsize=9,
            va="center",
            color=PALETTE["blue"],
            fontweight="bold",
        )
        if i:
            ax.annotate(
                "",
                xy=(0.3, y + 0.37),
                xytext=(0.3, y + 0.63),
                arrowprops={"arrowstyle": "->", "color": "#6B7385"},
            )
            ax.text(0.65, y, f"removed {row.removed}", fontsize=9, va="center", color=PALETTE["vermillion"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, n + 0.6)
    ax.axis("off")
    return ax


def _csv_name(name: str) -> str:
    """Translate a Stan name such as `Gamma[1,2]` into CmdStan's CSV column `Gamma.1.2`."""
    return name.replace("[", ".").replace(",", ".").replace("]", "").replace(" ", "")


def _chain_matrix(csv_files: Sequence[str], name: str) -> NDArray[np.float64]:
    name = _csv_name(name)
    frames = [pd.read_csv(p, comment="#", usecols=[name]) for p in csv_files]
    return np.column_stack([f[name].to_numpy(float) for f in frames])


@validation("TESTED")
def plot_trace(csv_files: Sequence[str], names: Sequence[str], *, axes: Any = None) -> Any:
    """Trace plots, one row per parameter, one colour per chain.

    Validation: TESTED.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(len(names), 1, figsize=(8, 2.2 * len(names)), squeeze=False)
        axes = axes[:, 0]
    for ax, name in zip(axes, names, strict=True):
        draws = _chain_matrix(csv_files, name)
        for c in range(draws.shape[1]):
            ax.plot(draws[:, c], lw=0.6, color=_CHAIN_COLOURS[c % len(_CHAIN_COLOURS)], alpha=0.85)
        ax.set_ylabel(name)
    axes[-1].set_xlabel("iteration")
    return axes


@validation("TESTED")
def plot_rank(csv_files: Sequence[str], names: Sequence[str], *, bins: int = 20, axes: Any = None) -> Any:
    """Rank plots (Vehtari et al. 2021): ranks of pooled draws, one histogram per chain.

    Uniform histograms indicate that the chains mix over the same distribution.

    Validation: TESTED.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata

    if axes is None:
        _, axes = plt.subplots(len(names), 1, figsize=(8, 2.2 * len(names)), squeeze=False)
        axes = axes[:, 0]
    for ax, name in zip(axes, names, strict=True):
        draws = _chain_matrix(csv_files, name)
        ranks = rankdata(draws.ravel()).reshape(draws.shape)
        edges = np.linspace(0, draws.size, bins + 1)
        for c in range(draws.shape[1]):
            counts, _ = np.histogram(ranks[:, c], bins=edges)
            ax.step(edges[:-1], counts, where="post", color=_CHAIN_COLOURS[c % len(_CHAIN_COLOURS)])
        ax.axhline(draws.shape[0] / bins, color=PALETTE["black"], lw=0.8, ls=":")
        ax.set_ylabel(name)
    axes[-1].set_xlabel("rank of pooled draws")
    return axes


@validation("TESTED")
def plot_energy(csv_files: Sequence[str], *, ax: Any = None) -> Any:
    """Energy plot: marginal energy against its first differences, pooled over chains.

    Clearly narrower differences than marginal energy indicate poor exploration (low E-BFMI).

    Validation: TESTED.
    """
    ax = _axes(ax)
    energy = _chain_matrix(csv_files, "energy__")
    e = (energy - energy.mean(axis=0)).ravel()
    de = np.diff(energy, axis=0).ravel()
    bins = np.linspace(float(min(np.min(e), np.min(de))), float(max(np.max(e), np.max(de))), 40)
    ax.hist(e, bins=bins, alpha=0.6, color=PALETTE["blue"], label="marginal energy")
    ax.hist(de, bins=bins, alpha=0.6, color=PALETTE["orange"], label="energy transition")
    ax.legend(frameon=False)
    return ax


@validation("TESTED")
def plot_cohort_trajectories(
    dataset: Any,
    endpoint: str = "updrs3",
    *,
    by: str = "cohort",
    max_participants: int = 200,
    seed: int = 0,
    ax: Any = None,
) -> Any:
    """Spaghetti plot of observed trajectories, coloured by a participant column, with group medians.

    Validation: TESTED.
    """
    ax = _axes(ax)
    obs = dataset.observations[dataset.observations["endpoint"] == endpoint]
    groups = dataset.participants.set_index("participant_id")[by].astype(str)
    ids = sorted(obs["participant_id"].unique())
    rng = np.random.default_rng(seed)
    if len(ids) > max_participants:
        ids = sorted(rng.choice(ids, max_participants, replace=False))
    colours = dict(zip(sorted(groups.unique()), _CHAIN_COLOURS, strict=False))
    for pid in ids:
        sub = obs[obs["participant_id"] == pid].sort_values("time_days")
        ax.plot(sub["time_days"] / C.DAYS_PER_YEAR, sub["value"], lw=0.6, alpha=0.35, color=colours[groups[pid]])
    obs = obs.assign(year=(obs["time_days"] / C.DAYS_PER_YEAR).round(), group=obs["participant_id"].map(groups))
    for g, sub in obs.groupby("group"):
        med = sub.groupby("year")["value"].median()
        ax.plot(med.index, med.to_numpy(), lw=2.5, color=colours[g], label=f"{g} median")
    ax.set_xlabel("years since baseline")
    ax.set_ylabel(_ENDPOINT_LABEL[C.ENDPOINT_INDEX[endpoint]])
    ax.legend(frameon=False)
    return ax


@validation("TESTED")
def plot_ppc_intervals(draws: Any, y: Any, *, level: float = 0.9, max_points: int = 300, ax: Any = None) -> Any:
    """Observed values against predictive intervals, sorted by predictive median (bayesplot style).

    Points outside their interval are drawn in vermillion.

    Validation: TESTED.
    """
    ax = _axes(ax)
    x = np.asarray(draws, float)
    obs = np.asarray(y, float)
    tail = (1 - level) / 2
    lo, med, hi = np.quantile(x, [tail, 0.5, 1 - tail], axis=0)
    order = np.argsort(med)[:max_points]
    idx = np.arange(order.size)
    ax.vlines(idx, lo[order], hi[order], color=PALETTE["sky"], lw=1.2)
    ax.plot(idx, med[order], "_", color=PALETTE["blue"])
    inside = (obs[order] >= lo[order]) & (obs[order] <= hi[order])
    ax.plot(idx[inside], obs[order][inside], ".", color=PALETTE["black"], ms=3)
    ax.plot(idx[~inside], obs[order][~inside], ".", color=PALETTE["vermillion"], ms=4)
    ax.set_xlabel("observations, sorted by predictive median")
    ax.set_ylabel("score")
    ax.set_title(f"{level:.0%} predictive intervals; {inside.mean():.0%} contain the observation")
    return ax


@validation("TESTED")
def plot_coverage_calibration(
    draws: Any,
    y: Any,
    *,
    levels: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    ax: Any = None,
) -> Any:
    """Empirical coverage of central intervals against nominal level; the diagonal is perfect calibration.

    Validation: TESTED.
    """
    ax = _axes(ax)
    x = np.asarray(draws, float)
    obs = np.asarray(y, float)
    emp = []
    for level in levels:
        tail = (1 - level) / 2
        lo, hi = np.quantile(x, [tail, 1 - tail], axis=0)
        emp.append(float(((obs >= lo) & (obs <= hi)).mean()))
    ax.plot([0, 1], [0, 1], color=PALETTE["black"], lw=0.8, ls=":")
    ax.plot(levels, emp, "o-", color=PALETTE["blue"])
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


@validation("TESTED")
def plot_reliability(p: Any, e: Any, *, bins: int = 10, ax: Any = None) -> Any:
    """Reliability diagram for event probabilities: observed frequency against forecast probability per bin.

    Validation: TESTED.
    """
    ax = _axes(ax)
    prob = np.asarray(p, float)
    event = np.asarray(e, float)
    ok = np.isfinite(prob) & np.isfinite(event)
    prob, event = prob[ok], event[ok]
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = which == b
        if m.any():
            xs.append(prob[m].mean())
            ys.append(event[m].mean())
            ns.append(int(m.sum()))
    ax.plot([0, 1], [0, 1], color=PALETTE["black"], lw=0.8, ls=":")
    ax.scatter(xs, ys, s=[20 + 3 * n for n in ns], color=PALETTE["orange"], edgecolor="white", zorder=3)
    ax.plot(xs, ys, color=PALETTE["orange"])
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


@validation("TESTED")
def participant_report(dataset: Any, participant_id: str, forecast: Any = None, *, title: str | None = None) -> Any:
    """A one-page figure for one participant: the three endpoints with any forecast, side by side.

    Validation: TESTED.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, endpoint in zip(axes, C.ENDPOINTS, strict=True):
        plot_trajectory(dataset, participant_id, forecast=forecast, endpoint=endpoint, ax=ax)
    fig.suptitle(title or f"Participant {participant_id}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
