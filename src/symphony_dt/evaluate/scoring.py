"""Held-out scoring of the twin against LOCF and a linear mixed model, ported from the code of record.

`score_fold` reproduces `five_fold/score_fold.py` and `pool_folds` reproduces `five_fold/pool_folds.py`.
Golden tests run both scripts and this module on the same synthetic folds and require identical
outputs. All three methods are scored on the same targets: interval coverage, CRPS and the absolute
error of the predictive median, by horizon window, with a participant-cluster bootstrap.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from symphony_dt.evaluate.metrics import (
    LMMFit,
    auc_with_ties,
    bootstrap_mean,
    bootstrap_ratio_skill,
    cluster_bootstrap_weights,
    crps_ensemble,
    event_probability,
    fit_lmm_em,
    lmm_condition,
    percentile_interval,
)
from symphony_dt.evaluate.spec import DeviationRecord, FrozenSpec, ScoringSpec, require_frozen
from symphony_dt.labels import validation

__all__ = ["FoldScores", "PooledScores", "obs_table", "pool_folds", "score_fold"]

_Z = {
    0.25: -0.6744897501960817,
    0.75: 0.6744897501960817,
    0.025: -1.959963984540054,
    0.975: 1.959963984540054,
}


def _node_times(d: Mapping[str, Any], i: int) -> NDArray[np.float64]:
    s, k = d["step_start"][i], d["K"][i]
    return np.concatenate([[0.0], np.cumsum(np.asarray(d["h"][s : s + k], float))])


@validation("VALIDATED")
def obs_table(d: Mapping[str, Any]) -> pd.DataFrame:
    """Return one row per observation of a Stan data block: participant, endpoint, time, value.

    Validation: VALIDATED.
    """
    rows = []
    for i in range(d["N"]):
        tt = _node_times(d, i)
        a, c = d["obs_start"][i], d["obs_count"][i]
        for j in range(a, a + c):
            rows.append((i, d["obs_k"][j], float(tt[d["obs_node"][j] - 1]), float(d["y"][j])))
    return pd.DataFrame(rows, columns=["i", "k", "t", "y"])


@validation("VALIDATED")
@dataclass
class FoldScores:
    """The scores of one fold: the aggregate cells, the per-target table and forecast quantiles.

    `scores` has the layout of the study's `scores.json`. `target_scores` is participant-level and
    must stay where the data are governed.

    Validation: VALIDATED.
    """

    scores: dict[str, Any]
    target_scores: pd.DataFrame
    forecast_quantiles: pd.DataFrame
    lines: list[str] = field(default_factory=list)
    specification: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Return `scores` serialised exactly as the code of record writes it.

        Validation: VALIDATED.
        """
        return json.dumps(self.scores, indent=1) + "\n"

    def sha256(self) -> str:
        """Return the sha256 of `to_json()`.

        Validation: VALIDATED.
        """
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def write(self, directory: str | Path) -> Path:
        """Write scores.json, scores.txt, target_scores.csv and forecast_quantiles.csv.

        Validation: TESTED.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        (out / "scores.json").write_text(self.to_json())
        (out / "scores.txt").write_text("\n".join(self.lines) + "\n")
        self.target_scores.to_csv(out / "target_scores.csv", index=False)
        self.forecast_quantiles.to_csv(out / "forecast_quantiles.csv", index=False)
        return out


@validation("VALIDATED")
def score_fold(
    fit_data: Mapping[str, Any],
    target_data: Mapping[str, Any],
    targets_meta: pd.DataFrame,
    twin_draws: ArrayLike,
    test_index: set[int] | Sequence[int],
    *,
    spec: ScoringSpec,
    frozen: FrozenSpec | None,
    deviation: DeviationRecord | None = None,
    age: ArrayLike | None = None,
    fold: int = 1,
) -> FoldScores:
    """Score one held-out fold: twin draws against LOCF and the LMM, as the code of record does.

    `twin_draws` is (draws, targets), the `y_rep_c` draws in target order. `targets_meta` has
    columns i, k, t, horizon, y, cohort and sex. `test_index` holds the zero-based indices of test
    participants. `frozen` must be the freeze of `spec`; see `freeze_spec`.

    Validation: VALIDATED.
    """
    prereg = require_frozen(spec, frozen, deviation)
    anchor = spec.anchor_years
    bins = list(spec.bins)
    events = spec.events
    endpoints = spec.endpoints
    windows = spec.windows
    meta = targets_meta.reset_index(drop=True)
    meta_rows: list[dict[str, Any]] = meta.to_dict(orient="records")  # type: ignore[assignment]
    L, U = np.asarray(target_data["Lk"], float), np.asarray(target_data["Uk"], float)
    n_tgt = int(target_data["n_obs"])
    if len(meta) != n_tgt or not np.allclose(meta["y"].to_numpy(), np.asarray(target_data["y"], float)):
        raise ValueError("targets_meta does not match the target data")
    if not np.array_equal(meta["k"].to_numpy(), np.asarray(target_data["obs_k"])):
        raise ValueError("endpoint order of targets_meta differs from the target data")
    tests = set(int(i) for i in test_index)
    if not set(meta["i"]).issubset(tests):
        raise ValueError("a target belongs to a non-test participant")
    age_arr = np.asarray(age, float) if age is not None else np.full(int(fit_data["N"]), np.nan)

    obs = obs_table(fit_data)
    obs["test"] = obs["i"].isin(tests)
    leak = int((obs["test"] & (obs["t"] > anchor)).sum())
    if leak:
        raise ValueError(f"masking violated: {leak} test observations after the anchor remain")

    draws = np.asarray(twin_draws, float)
    if draws.ndim != 2 or draws.shape[1] != n_tgt:
        raise ValueError(f"twin_draws must be (draws, {n_tgt}), got {draws.shape}")
    if not np.all(np.isfinite(draws)):
        raise ValueError("non-finite forecast draws")

    train = obs[~obs["test"]].copy()
    pre = train[train["t"] <= anchor]
    anchor_t = pre.groupby("i")["t"].max()
    last = pre.sort_values("t").groupby(["i", "k"])["y"].last()
    post = train[train["t"] > anchor].copy()
    post["anchor_t"] = post["i"].map(anchor_t).fillna(0.0)
    post["h"] = post["t"] - post["anchor_t"]
    post["last"] = [last.get((i, k), np.nan) for i, k in zip(post["i"], post["k"], strict=True)]
    post = post.dropna(subset=["last"])
    post["res"] = post["y"] - post["last"]
    post["bin"] = np.digitize(post["h"], bins) - 1
    resid: dict[tuple[int, int], NDArray[np.float64]] = {}
    bin_report: dict[str, dict[str, Any]] = {}
    for k in endpoints:
        pk = post[post["k"] == k]
        for b in range(len(bins) - 1):
            rb = pk.loc[pk["bin"] == b, "res"].to_numpy()
            pooled = rb.size < spec.min_bin
            resid[(k, b)] = pk["res"].to_numpy() if pooled else rb
            bin_report["%d_%d" % (k, b)] = {"n": int(rb.size), "pooled": bool(pooled)}

    lmm: dict[int, LMMFit] = {}
    for k in endpoints:
        tk = train[train["k"] == k]
        groups = [(g["t"].to_numpy(), g["y"].to_numpy()) for _, g in tk.groupby("i")]
        lmm[k] = fit_lmm_em(groups)

    test_kept = obs[obs["test"]]
    visits_pre = test_kept.groupby("i")["t"].nunique()
    tk_groups = {key: g for key, g in test_kept.groupby(["i", "k"])}
    rng = np.random.default_rng(spec.seed_lmm)

    rec: list[dict[str, Any]] = []
    qrows: list[list[float]] = []
    for q in range(n_tgt):
        i, k, t, y, h = (
            int(meta_rows[q]["i"]),
            int(meta_rows[q]["k"]),
            float(meta_rows[q]["t"]),
            float(meta_rows[q]["y"]),
            float(meta_rows[q]["horizon"]),
        )
        g = tk_groups.get((i, k))
        row: dict[str, Any] = {
            "q": q,
            "i": i,
            "k": k,
            "t": t,
            "h": h,
            "y": y,
            "cohort": str(meta_rows[q]["cohort"]),
            "sex": int(meta_rows[q]["sex"]),
        }
        x = draws[:, q]
        qs = np.quantile(x, [0.025, 0.25, 0.5, 0.75, 0.975])
        row.update(
            tw_c50=float(qs[1] <= y <= qs[3]),
            tw_c95=float(qs[0] <= y <= qs[4]),
            tw_crps=crps_ensemble(x, y),
            tw_ae=abs(qs[2] - y),
        )
        qrows.append([q, i, k, t, h, *qs.tolist()])
        if g is not None and len(g):
            lv = float(g.sort_values("t")["y"].iloc[-1])
            b = int(np.digitize([h], bins)[0] - 1)
            xs = np.clip(lv + resid[(k, b)], L[k - 1], U[k - 1])
            ql = np.quantile(xs, [0.025, 0.25, 0.75, 0.975])
            row.update(
                locf=1.0,
                lo_c50=float(ql[1] <= y <= ql[2]),
                lo_c95=float(ql[0] <= y <= ql[3]),
                lo_crps=crps_ensemble(xs, y),
                lo_ae=abs(lv - y),
            )
            row["lv"] = lv
            if k in events:
                direction, thr = events[k]
                row["ev"] = float((y - lv) >= thr) if direction == "up" else float((lv - y) >= thr)
                row["p_tw"] = event_probability(x, lv, k, events)
                row["p_lo"] = event_probability(xs, lv, k, events)
            tt, yy = g["t"].to_numpy(), g["y"].to_numpy()
        else:
            row.update(locf=0.0, lo_c50=np.nan, lo_c95=np.nan, lo_crps=np.nan, lo_ae=np.nan)
            tt, yy = np.array([]), np.array([])
        th = lmm[k]
        bh, Vb = lmm_condition(th, tt, yy)
        xv = np.array([1.0, t])
        mu = float(xv @ (th.beta + bh))
        sd = float(np.sqrt(xv @ Vb @ xv + th.s2))
        lo_, hi_ = L[k - 1], U[k - 1]
        ivl = {p: min(max(mu + z * sd, lo_), hi_) for p, z in _Z.items()}
        xs = np.clip(mu + sd * rng.standard_normal(spec.lmm_samples), lo_, hi_)
        row.update(
            lm_c50=float(ivl[0.25] <= y <= ivl[0.75]),
            lm_c95=float(ivl[0.025] <= y <= ivl[0.975]),
            lm_crps=crps_ensemble(xs, y),
            lm_ae=abs(min(max(mu, lo_), hi_) - y),
        )
        if "ev" in row:
            row["p_lm"] = event_probability(xs, row["lv"], k, events)
        row["age"] = float(age_arr[i])
        row["visits_pre"] = int(visits_pre.get(i, 0))
        rec.append(row)
    df = pd.DataFrame(rec)
    first = df.groupby(["i", "k"])["t"].transform("min")
    df["w_next"] = df["t"] == first
    df["last_h"] = df.groupby("i")["h"].transform("max")
    for col in ("lv", "ev", "p_tw", "p_lo", "p_lm"):
        if col not in df:
            df[col] = np.nan
    target_scores = df.assign(fold=fold).drop(columns=["q"])
    quantiles = pd.DataFrame(qrows, columns=["q", "i", "k", "t", "h", "q025", "q25", "q50", "q75", "q975"])

    parts = np.array(sorted(df["i"].unique()))
    pos = {p: n for n, p in enumerate(parts)}
    df["pp"] = df["i"].map(pos)
    W = cluster_bootstrap_weights(parts.size, spec.bootstrap_replicates, spec.seed_bootstrap)

    def boot(sub: pd.DataFrame, col: str) -> list[float]:
        return bootstrap_mean(sub[col].to_numpy(float), sub["pp"].to_numpy(), W)

    out: dict[str, Any] = {
        "spec": spec.name,
        "n_targets": n_tgt,
        "n_participants": int(parts.size),
        "twin_draws": int(draws.shape[0]),
        "locf_bins": bin_report,
        "lmm": {str(k): th.as_dict() for k, th in lmm.items()},
        "cells": {},
        "strata": {},
    }
    lines: list[str] = []
    for k, name in endpoints.items():
        for w, rngw in windows.items():
            sub = df[df["k"] == k]
            if w == "next":
                sub = sub[sub["w_next"]]
            else:
                assert rngw is not None
                sub = sub[(sub["h"] >= rngw[0]) & (sub["h"] <= rngw[1])]
            com = sub[sub["locf"] == 1.0].copy()
            if len(sub) == 0:
                continue
            cell: dict[str, Any] = {
                "n_all": len(sub),
                "n_part_all": int(sub["i"].nunique()),
                "n_common": len(com),
                "twin_all": {m: boot(sub, "tw_" + m) for m in ("c50", "c95", "crps", "ae")},
            }
            if len(com):
                com["d_lo"] = com["tw_crps"] - com["lo_crps"]
                com["d_lm"] = com["tw_crps"] - com["lm_crps"]
                for meth, pre_ in (("twin", "tw_"), ("locf", "lo_"), ("lmm", "lm_")):
                    cell[meth] = {m: boot(com, pre_ + m) for m in ("c50", "c95", "crps", "ae")}
                cell["crps_diff_twin_minus_locf"] = boot(com, "d_lo")
                cell["crps_diff_twin_minus_lmm"] = boot(com, "d_lm")
            out["cells"][f"{name}|{w}"] = cell
            if len(com):
                lines.append(
                    "%-9s %-4s n=%5d/%5d  cov95 twin %.3f locf %.3f lmm %.3f | cov50 twin %.3f locf %.3f "
                    "lmm %.3f | CRPS twin %.3f locf %.3f lmm %.3f | dCRPS-locf %+.3f [%+.3f,%+.3f] "
                    "dCRPS-lmm %+.3f [%+.3f,%+.3f]"
                    % (
                        name,
                        w,
                        len(com),
                        len(sub),
                        cell["twin"]["c95"][0],
                        cell["locf"]["c95"][0],
                        cell["lmm"]["c95"][0],
                        cell["twin"]["c50"][0],
                        cell["locf"]["c50"][0],
                        cell["lmm"]["c50"][0],
                        cell["twin"]["crps"][0],
                        cell["locf"]["crps"][0],
                        cell["lmm"]["crps"][0],
                        *cell["crps_diff_twin_minus_locf"],
                        *cell["crps_diff_twin_minus_lmm"],
                    )
                )
        sub = df[df["k"] == k]
        for col in ("cohort", "sex"):
            for val, s2 in sub.groupby(col):
                key = f"{name}|{col}={val}"
                out["strata"][key] = {
                    "n": len(s2),
                    "twin_c95": boot(s2, "tw_c95"),
                    "twin_c50": boot(s2, "tw_c50"),
                    "lmm_c95": boot(s2, "lm_c95"),
                }
                lines.append(
                    "%-9s %s=%-10s n=%5d cov95 twin %.3f [%.3f,%.3f] cov50 twin %.3f"
                    % (
                        name,
                        col,
                        val,
                        len(s2),
                        *out["strata"][key]["twin_c95"],
                        out["strata"][key]["twin_c50"][0],
                    )
                )
    return FoldScores(
        scores=out,
        target_scores=target_scores,
        forecast_quantiles=quantiles,
        lines=lines,
        specification=prereg,
    )


@validation("VALIDATED")
@dataclass
class PooledScores:
    """Pooled scores over folds, in the layout of the study's `pooled_scores.json`.

    Validation: VALIDATED.
    """

    scores: dict[str, Any]
    lines: list[str] = field(default_factory=list)
    specification: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Return the scores serialised exactly as the code of record writes them.

        Validation: VALIDATED.
        """
        return json.dumps(self.scores, indent=1) + "\n"

    def sha256(self) -> str:
        """Return the sha256 of `to_json()`.

        Validation: VALIDATED.
        """
        return hashlib.sha256(self.to_json().encode()).hexdigest()


def _window(df: pd.DataFrame, w: str, spec: ScoringSpec) -> pd.DataFrame:
    if w == "next":
        return df[df["w_next"]]
    bounds = spec.windows.get(w) if w in spec.windows else spec.event_windows[w]
    assert bounds is not None
    lo, hi = bounds
    return df[(df["h"] >= lo) & (df["h"] <= hi)]


@validation("VALIDATED")
def pool_folds(
    folds: Sequence[tuple[pd.DataFrame | None, Mapping[str, Any] | None]],
    *,
    spec: ScoringSpec,
    frozen: FrozenSpec | None,
    deviation: DeviationRecord | None = None,
) -> PooledScores:
    """Pool fold scores: primary cells, per-fold spread, events, strata and the O-20 decomposition.

    `folds` holds, in fold order, (target_scores, fit_record) per fold. A fold enters only when
    its fit record says `converged`. Every participant may be a test participant in one fold only.
    No stratum or event cell is reported from fewer than `spec.min_cell_participants` people.

    Validation: VALIDATED.
    """
    prereg = require_frozen(spec, frozen, deviation)
    names = spec.endpoints
    min_part = spec.min_cell_participants
    frames, fold_status = [], {}
    for k, (ts, rec) in enumerate(folds, start=1):
        if ts is None or rec is None:
            fold_status[k] = "missing"
            continue
        if not rec["converged"]:
            fold_status[k] = "not converged (unscored)"
            continue
        fold_status[k] = "scored: %d targets, %d participants, max R-hat %.5f, min ESS %.1f" % (
            len(ts),
            ts["i"].nunique(),
            rec["max_rhat"],
            rec["min_ess_bulk"],
        )
        frames.append(ts)
    if not frames:
        raise ValueError("no scored fold")
    df = pd.concat(frames, ignore_index=True)
    per = df.groupby("i")["fold"].nunique()
    if (per > 1).any():
        raise ValueError("a participant is a test participant in more than one fold")
    parts = np.array(sorted(df["i"].unique()))
    df["pp"] = df["i"].map({p: n for n, p in enumerate(parts)})
    W = cluster_bootstrap_weights(parts.size, spec.bootstrap_replicates, spec.seed_bootstrap)

    def boot(sub: pd.DataFrame, col: str) -> list[float]:
        return bootstrap_mean(sub[col].to_numpy(float), sub["pp"].to_numpy(), W, nan_aware=True)

    def boot_auc(sub: pd.DataFrame, pcol: str) -> list[float]:
        p, e = sub[pcol].to_numpy(float), sub["ev"].to_numpy(float)
        pp = sub["pp"].to_numpy()
        est = auc_with_ties(p, e, np.ones_like(p))
        bs = np.array([auc_with_ties(p, e, W[b, pp]) for b in range(W.shape[0])])
        return percentile_interval(est, bs, nan_aware=True)

    out: dict[str, Any] = {
        "spec": spec.name + " + ADDENDUM_2026-10-21",
        "folds": fold_status,
        "n_targets": len(df),
        "n_participants": int(parts.size),
        "cells": {},
        "per_fold": {},
        "events": {},
        "strata": {},
        "post_hoc_o20": {},
    }
    lines = [
        "POOLED over folds %s: %d targets, %d participants"
        % ([k for k, s in fold_status.items() if s.startswith("scored")], len(df), parts.size)
    ]

    for k, name in names.items():
        for w in spec.windows:
            sub = _window(df[df["k"] == k], w, spec)
            com = sub[sub["locf"] == 1.0].copy()
            if not len(sub) or not len(com):
                continue
            com["d_lo"] = com["tw_crps"] - com["lo_crps"]
            com["d_lm"] = com["tw_crps"] - com["lm_crps"]
            cell: dict[str, Any] = {
                "n_all": len(sub),
                "n_common": len(com),
                "n_part": int(sub["i"].nunique()),
                "twin_all": {m: boot(sub, "tw_" + m) for m in ("c50", "c95", "crps", "ae")},
            }
            for meth, pre in (("twin", "tw_"), ("locf", "lo_"), ("lmm", "lm_")):
                cell[meth] = {m: boot(com, pre + m) for m in ("c50", "c95", "crps", "ae")}
            cell["crps_diff_twin_minus_locf"] = boot(com, "d_lo")
            cell["crps_diff_twin_minus_lmm"] = boot(com, "d_lm")
            out["cells"][f"{name}|{w}"] = cell
            lines.append(
                "%-9s %-4s n=%5d/%5d cov95 twin %.3f locf %.3f lmm %.3f | cov50 twin %.3f | CRPS twin %.3f "
                "locf %.3f lmm %.3f | dCRPS-locf %+.3f [%+.3f,%+.3f] dCRPS-lmm %+.3f [%+.3f,%+.3f]"
                % (
                    name,
                    w,
                    len(com),
                    len(sub),
                    cell["twin"]["c95"][0],
                    cell["locf"]["c95"][0],
                    cell["lmm"]["c95"][0],
                    cell["twin"]["c50"][0],
                    cell["twin"]["crps"][0],
                    cell["locf"]["crps"][0],
                    cell["lmm"]["crps"][0],
                    *cell["crps_diff_twin_minus_locf"],
                    *cell["crps_diff_twin_minus_lmm"],
                )
            )

    for f, g in df.groupby("fold"):
        for k, name in names.items():
            s = g[g["k"] == k]
            c = s[s["locf"] == 1.0]
            out["per_fold"][f"{f}|{name}"] = {
                "n": len(s),
                "twin_c95": round(float(s["tw_c95"].mean()), 4),
                "twin_c50": round(float(s["tw_c50"].mean()), 4),
                "d_crps_locf": round(float((c["tw_crps"] - c["lo_crps"]).mean()), 4),
                "d_crps_lmm": round(float((c["tw_crps"] - c["lm_crps"]).mean()), 4),
            }
    for _k, name in names.items():
        vals = [v for key, v in out["per_fold"].items() if key.endswith("|" + name)]
        lines.append("%-9s per-fold cov95 twin %s" % (name, " ".join("%.3f" % v["twin_c95"] for v in vals)))

    for k in spec.events:
        for w in spec.event_windows:
            sub = _window(df[(df["k"] == k) & df["ev"].notna()], w, spec).copy()
            if sub["i"].nunique() < min_part:
                continue
            for m in ("tw", "lo", "lm"):
                sub["b_" + m] = (sub["p_" + m] - sub["ev"]) ** 2
            ev: dict[str, Any] = {
                "n": len(sub),
                "n_part": int(sub["i"].nunique()),
                "prevalence": boot(sub, "ev"),
                "brier": {m: boot(sub, "b_" + m) for m in ("tw", "lo", "lm")},
                "bss_twin_vs_locf": bootstrap_ratio_skill(sub["b_tw"], sub["b_lo"], sub["pp"], W),
                "bss_twin_vs_lmm": bootstrap_ratio_skill(sub["b_tw"], sub["b_lm"], sub["pp"], W),
                "auc": {m: boot_auc(sub, "p_" + m) for m in ("tw", "lo", "lm")},
            }
            out["events"][f"{names[k]}|{w}"] = ev
            lines.append(
                "EVENT %-9s %s n=%d prev %.3f | Brier twin %.4f locf %.4f lmm %.4f | BSS vs locf %+.3f "
                "[%+.3f,%+.3f] vs lmm %+.3f [%+.3f,%+.3f] | AUC twin %.3f locf %.3f lmm %.3f"
                % (
                    names[k],
                    w,
                    len(sub),
                    ev["prevalence"][0],
                    ev["brier"]["tw"][0],
                    ev["brier"]["lo"][0],
                    ev["brier"]["lm"][0],
                    *ev["bss_twin_vs_locf"],
                    *ev["bss_twin_vs_lmm"],
                    ev["auc"]["tw"][0],
                    ev["auc"]["lo"][0],
                    ev["auc"]["lm"][0],
                )
            )

    df["sex_label"] = df["sex"].map({0: "Female", 1: "Male"}).fillna("unknown")
    df["age_band"] = pd.cut(df["age"], [-np.inf, 60, 70, np.inf], right=False, labels=["<60", "60-69", ">=70"]).astype(
        str
    )
    df["visits_band"] = pd.cut(df["visits_pre"], [0, 3, 5, np.inf], right=False, labels=["0-2", "3-4", ">=5"]).astype(
        str
    )
    df["followup"] = np.where(df["last_h"] <= 2.0, "last visit <=2y after anchor", "followed >2y after anchor")
    for k, name in names.items():
        sk = df[df["k"] == k]
        for col in ("sex_label", "cohort", "age_band", "visits_band", "followup"):
            for val, s in sk.groupby(col):
                key = f"{name}|{col}={val}"
                if s["i"].nunique() < min_part:
                    out["strata"][key] = {"n_part": int(s["i"].nunique()), "suppressed": True}
                    continue
                out["strata"][key] = {
                    "n": len(s),
                    "n_part": int(s["i"].nunique()),
                    "twin_c95": boot(s, "tw_c95"),
                    "twin_c50": boot(s, "tw_c50"),
                    "lmm_c95": boot(s, "lm_c95"),
                }
                st = out["strata"][key]
                lines.append(
                    "STRATUM %-9s %-12s %-30s n=%5d cov95 twin %.3f [%.3f,%.3f] lmm %.3f"
                    % (name, col, str(val)[:30], len(s), *st["twin_c95"], st["lmm_c95"][0])
                )

    u = df[df["k"] == 1]
    for coh in ["ALL", *sorted(u["cohort"].unique())]:
        s = u if coh == "ALL" else u[u["cohort"] == coh]
        nz = s[s["y"] != 0]
        out["post_hoc_o20"][coh] = {
            "n": len(s),
            "n_y0": int((s["y"] == 0).sum()),
            "y0_outside95": int(((s["y"] == 0) & (s["tw_c95"] == 0)).sum()),
            "cov95": round(float(s["tw_c95"].mean()), 4),
            "cov95_excl_y0": round(float(nz["tw_c95"].mean()), 4) if len(nz) else None,
        }
        o = out["post_hoc_o20"][coh]
        lines.append(
            "POST_HOC O-20 UPDRS-III %-20s n=%5d y0=%4d y0 outside95=%4d cov95 %.3f excl y0 %.3f"
            % (
                coh[:20],
                o["n"],
                o["n_y0"],
                o["y0_outside95"],
                o["cov95"],
                o["cov95_excl_y0"] if o["cov95_excl_y0"] is not None else float("nan"),
            )
        )
    return PooledScores(scores=out, lines=lines, specification=prereg)
