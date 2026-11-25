"""Calibration by subgroup: interval coverage, CRPS and PIT uniformity within each level of a grouping.

A twin can be well calibrated overall and poorly calibrated in a subgroup, for example in older
participants or at one site. `subgroup_calibration` scores the per-target table of `score_fold`
or `pool_folds` within each level of each grouping, for the twin and both baselines. Intervals
come from a participant-cluster bootstrap within the subgroup, as in the scoring code of record.
Cells with fewer participants than the small-cell limit are suppressed in every output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from symphony_dt import constants as C
from symphony_dt.calibration.ppc import pit_values
from symphony_dt.evaluate.metrics import bootstrap_mean, cluster_bootstrap_weights
from symphony_dt.labels import validation

__all__ = ["SubgroupCalibration", "age_band", "subgroup_calibration"]

METHOD_PREFIX: dict[str, str] = {"twin": "tw", "locf": "lo", "lmm": "lm"}
_METRICS = ("c50", "c95", "crps")


@validation("TESTED")
def age_band(age: ArrayLike, edges: Sequence[float] = (55.0, 65.0, 75.0)) -> NDArray[np.str_]:
    """Label each age with its band, such as "55 to 64", for use as a grouping.

    Bands are closed below and open above. A missing age is labelled "unknown".

    Validation: TESTED.
    """
    a = np.asarray(age, float)
    cuts = [float(e) for e in edges]
    if any(b <= x for x, b in pairwise(cuts)):
        raise ValueError("edges must increase strictly")
    labels = [f"under {cuts[0]:g}"]
    labels += [f"{lo:g} to {hi - 1:g}" for lo, hi in pairwise(cuts)]
    labels += [f"{cuts[-1]:g} and over"]
    idx = np.digitize(a, cuts)
    out = np.array([labels[j] for j in idx], dtype=object)
    out[~np.isfinite(a)] = "unknown"
    return out.astype(str)


@validation("TESTED")
@dataclass
class SubgroupCalibration:
    """Calibration scores per grouping, level, endpoint and method.

    `table` has one row per cell with the participant and target counts, each metric with its
    bootstrap interval, and for the twin the PIT mean, standard deviation and Kolmogorov-Smirnov
    p-value. Suppressed cells hold no counts or scores.

    Validation: TESTED.
    """

    table: pd.DataFrame
    min_cell: int
    multiplicity: str

    def to_markdown(self, *, metric: str = "c95") -> str:
        """Return one Markdown table of a metric, with a column per method and the PIT test.

        Validation: TESTED.
        """
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}")
        methods = [m for m in METHOD_PREFIX if m in set(self.table["method"])]
        head = ["grouping", "level", "endpoint", "participants", *methods, "PIT KS p"]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for (g, lev, e), cell in self.table.groupby(["grouping", "level", "endpoint"], sort=False):
            first = cell.iloc[0]
            if bool(first["suppressed"]):
                vals = ["suppressed"] * (len(methods) + 1)
                count = f"<{self.min_cell}"
            else:
                by = cell.set_index("method")
                vals = [_fmt_ci(by, m, metric) if m in by.index else "" for m in methods]
                adjusted = dict(zip(cell["method"], cell["pit_ks_p_adjusted"].to_numpy(float), strict=True))
                p = adjusted.get("twin", np.nan)
                vals.append("" if not np.isfinite(p) else f"{p:.3f}")
                count = str(int(first["n_participants"]))
            lines.append("| " + " | ".join([str(g), str(lev), str(e), count, *vals]) + " |")
        return "\n".join(lines) + "\n"

    def plot(self, *, metric: str = "c95", endpoint: str | None = None, ax: Any = None) -> Any:
        """Plot a metric per subgroup with bootstrap intervals; see `plots.plot_subgroup_calibration`.

        Validation: TESTED.
        """
        from symphony_dt.plots import plot_subgroup_calibration

        return plot_subgroup_calibration(self, metric=metric, endpoint=endpoint, ax=ax)


def _fmt_ci(by_method: pd.DataFrame, method: str, metric: str) -> str:
    est, lo, hi = by_method.loc[method, [metric, f"{metric}_lo", f"{metric}_hi"]].to_numpy(float)
    if not np.isfinite(est):
        return ""
    return f"{est:.3f} [{lo:.3f}, {hi:.3f}]"


def _labels(name: str, spec: str | Mapping[Any, Any] | ArrayLike, frame: pd.DataFrame) -> NDArray[np.object_]:
    """Resolve one grouping to a label per row of the per-target table."""
    if isinstance(spec, str):
        if spec not in frame.columns:
            raise ValueError(f"grouping {name!r}: no column {spec!r} in the target table")
        values = frame[spec].to_numpy()
    elif isinstance(spec, Mapping):
        missing = sorted(set(frame["i"]) - set(spec))
        if missing:
            raise ValueError(f"grouping {name!r}: no label for participants {missing[:5]}")
        values = frame["i"].map(spec).to_numpy()
    else:
        values = np.asarray(spec, dtype=object)
        if values.shape != (len(frame),):
            raise ValueError(f"grouping {name!r}: expected one label per target row, got shape {values.shape}")
    out = np.array(["unknown" if pd.isna(v) else str(v) for v in values], dtype=object)
    return out


def _adjust(p: NDArray[np.float64], how: str) -> NDArray[np.float64]:
    """Adjust p-values for the number of finite tests: Bonferroni, Benjamini-Hochberg or none."""
    out = p.copy()
    ok = np.isfinite(p)
    m = int(ok.sum())
    if how == "none" or m == 0:
        return out
    if how == "bonferroni":
        out[ok] = np.minimum(p[ok] * m, 1.0)
        return out
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


@validation("TESTED")
def subgroup_calibration(
    target_scores: pd.DataFrame,
    groups: Mapping[str, str | Mapping[Any, Any] | ArrayLike],
    *,
    twin_draws: ArrayLike | None = None,
    pit: ArrayLike | None = None,
    methods: Sequence[str] = ("twin", "locf", "lmm"),
    min_cell: int = 10,
    multiplicity: str = "bh",
    bootstrap_replicates: int = 1000,
    seed: int = 20261021,
    pit_seed: int = 0,
) -> SubgroupCalibration:
    """Score interval coverage, CRPS and PIT uniformity within each level of each grouping.

    `target_scores` is `FoldScores.target_scores` or the pooled per-target table, one row per
    target. Each entry of `groups` names a grouping and gives its labels in one of three ways:
    a column of `target_scores`, such as "sex" or "cohort"; a mapping from participant index `i` to
    a label, for site or genotype group; or an array with one label per row, for example from
    `age_band(target_scores["age"])`. An overall row with grouping "all" is always included.

    PIT values come from `pit`, one per row, or from `twin_draws`, shaped (draws, rows) in row
    order, as passed to `score_fold`. Rows score their own target, so PITs of one participant are
    correlated and the Kolmogorov-Smirnov p-values are approximate. `multiplicity` adjusts them
    across all cells by "bh" (Benjamini-Hochberg), "bonferroni" or "none".

    Cells with fewer than `min_cell` participants are suppressed: no counts and no scores.

    Validation: TESTED.
    """
    frame = target_scores.reset_index(drop=True)
    for col in ("i", "k"):
        if col not in frame.columns:
            raise ValueError(f"target table needs column {col!r}")
    unknown = [m for m in methods if m not in METHOD_PREFIX]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {list(METHOD_PREFIX)}")
    if multiplicity not in ("bh", "bonferroni", "none"):
        raise ValueError("multiplicity must be 'bh', 'bonferroni' or 'none'")
    if min_cell < 1:
        raise ValueError("min_cell must be at least 1")
    if pit is not None and twin_draws is not None:
        raise ValueError("pass pit or twin_draws, not both")

    pit_arr: NDArray[np.float64] | None = None
    if pit is not None:
        pit_arr = np.asarray(pit, float)
    elif twin_draws is not None:
        draws = np.asarray(twin_draws, float)
        if draws.ndim != 2 or draws.shape[1] != len(frame):
            raise ValueError(f"twin_draws must be (draws, {len(frame)}), got {draws.shape}")
        pit_arr = pit_values(draws, frame["y"].to_numpy(float), seed=pit_seed)
    if pit_arr is not None and pit_arr.shape != (len(frame),):
        raise ValueError(f"pit must hold one value per target row, got shape {pit_arr.shape}")

    labelled = {"all": np.full(len(frame), "all", dtype=object)}
    for name, spec in groups.items():
        if name == "all":
            raise ValueError("'all' is reserved for the overall row")
        labelled[name] = _labels(name, spec, frame)

    rows: list[dict[str, Any]] = []
    for gname, labels in labelled.items():
        for level in sorted(set(labels), key=str):
            in_level = labels == level
            for k in sorted(frame["k"].unique()):
                sub = frame[in_level & (frame["k"] == k).to_numpy()]
                parts = np.array(sorted(sub["i"].unique()))
                base = {
                    "grouping": gname,
                    "level": level,
                    "endpoint": C.ENDPOINTS[int(k) - 1],
                    "k": int(k),
                }
                suppressed = parts.size < min_cell
                if suppressed:
                    for m in methods:
                        rows.append({**base, "method": m, "suppressed": True})
                    continue
                pos = {p: n for n, p in enumerate(parts)}
                pp = sub["i"].map(pos).to_numpy()
                W = cluster_bootstrap_weights(parts.size, bootstrap_replicates, seed)
                for m in methods:
                    row: dict[str, Any] = {
                        **base,
                        "method": m,
                        "suppressed": False,
                        "n_participants": int(parts.size),
                        "n_targets": int(len(sub)),
                    }
                    for metric in _METRICS:
                        col = f"{METHOD_PREFIX[m]}_{metric}"
                        if col not in sub.columns:
                            continue
                        vals = sub[col].to_numpy(float)
                        keep = np.isfinite(vals)
                        if not keep.any():
                            continue
                        est, lo, hi = bootstrap_mean(vals[keep], pp[keep], W, nan_aware=True)
                        row.update({metric: est, f"{metric}_lo": lo, f"{metric}_hi": hi})
                    if m == "twin" and pit_arr is not None:
                        u = pit_arr[sub.index.to_numpy()]
                        row["pit_mean"] = round(float(u.mean()), 4)
                        row["pit_sd"] = round(float(u.std(ddof=1)), 4) if u.size > 1 else np.nan
                        row["pit_ks_p"] = float(stats.kstest(u, "uniform").pvalue)
                    rows.append(row)

    cols = [
        "grouping",
        "level",
        "endpoint",
        "k",
        "method",
        "suppressed",
        "n_participants",
        "n_targets",
        *[f"{m}{s}" for m in _METRICS for s in ("", "_lo", "_hi")],
        "pit_mean",
        "pit_sd",
        "pit_ks_p",
    ]
    table = pd.DataFrame(rows).reindex(columns=cols)
    table["suppressed"] = table["suppressed"].astype(bool)
    table["pit_ks_p_adjusted"] = _adjust(table["pit_ks_p"].to_numpy(float), multiplicity)
    return SubgroupCalibration(table=table, min_cell=min_cell, multiplicity=multiplicity)
