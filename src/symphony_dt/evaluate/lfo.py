"""Exact leave-future-out log predictive density of the forecasts, over one or more anchors.

Leave-future-out cross-validation scores a forecast only on data later than the history it saw,
as set out by Bürkner, Gabry and Vehtari (2020). At each anchor the model is fitted on every
history up to that time, and each later observation is a target. The exact route refits at every
anchor, so no importance sampling approximation is involved.

The score of a target is its log predictive density, the log of the mean over posterior draws of
the observation density. The density is the one of the model: a normal renormalised to the scale
bounds for a truncated endpoint, or a tobit density with point masses at the bounds for a censored
one. Given a draw of the latent path, the targets of one participant are independent, so the
joint density of a participant's future block is the mean over draws of their product.

Scores describe how well forecasts match later observations on the scale of the data. They are
associational and say nothing about the effect of any intervention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, log_ndtr, logsumexp

from symphony_dt import constants as C
from symphony_dt.labels import validation

__all__ = [
    "AnchorDraws",
    "LFOScores",
    "anchor_draws",
    "leave_future_out",
    "observation_draws",
    "predictive_log_density",
]

_LOG_2PI = float(np.log(2 * np.pi))
#: The six free entries of the coupling matrix, as in `data.simulate.coupling_matrix`.
_COUPLING: dict[tuple[int, int], str] = {
    (0, 0): "c_MM",
    (1, 0): "c_CM",
    (1, 1): "c_CC",
    (1, 2): "c_CN",
    (2, 0): "c_NM",
    (2, 2): "c_NN",
}
_SUMMARY_COLUMNS = ("anchor", "endpoint", "n_targets", "n_participants", "elpd", "se", "elpd_mean")


def _log_mass(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return log(Phi(b) - Phi(a)) for a <= b, using the upper tail where it is more accurate."""
    upper = a > 0
    hi = np.where(upper, -a, b)
    lo = np.where(upper, -b, a)
    log_hi, log_lo = log_ndtr(hi), log_ndtr(lo)
    with np.errstate(divide="ignore"):
        out = log_hi + np.log1p(-np.exp(np.minimum(log_lo - log_hi, 0.0)))
    return np.asarray(np.maximum(out, np.log(1e-300)), float)


@validation("TESTED")
def predictive_log_density(
    y: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    censored: ArrayLike,
) -> NDArray[np.float64]:
    """Return the log density of each observation under each draw of its mean and noise.

    `mu` and `sigma` are (draws, targets). `y`, `lower`, `upper` and `censored` hold one value
    per target, or a scalar for all. The density matches `endpoint_log_density` in pd_v2's
    model.stan. With `censored` 0 it is a normal renormalised to the bounds. With `censored` 1 an
    interior value keeps the normal density, and a value at a bound gets the mass beyond it.

    Validation: TESTED.
    """
    m = np.atleast_2d(np.asarray(mu, float))
    s = np.atleast_2d(np.asarray(sigma, float))
    if m.shape != s.shape:
        raise ValueError(f"mu {m.shape} and sigma {s.shape} differ in shape")
    if np.any(~(s > 0)):
        raise ValueError("sigma must be positive")
    n_target = m.shape[1]
    yy, lo, hi, cens = (np.broadcast_to(np.asarray(v, float), (n_target,)) for v in (y, lower, upper, censored))
    if np.any(lo >= hi):
        raise ValueError("lower must be below upper for every target")
    z = (yy - m) / s
    normal = -0.5 * z * z - np.log(s) - 0.5 * _LOG_2PI
    truncated = normal - _log_mass((lo - m) / s, (hi - m) / s)
    tobit = np.where(yy >= hi, log_ndtr((m - hi) / s), np.where(yy <= lo, log_ndtr((lo - m) / s), normal))
    return np.asarray(np.where(cens.astype(bool), tobit, truncated), float)


def _log_mean_exp(lp: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(logsumexp(lp, axis=0) - np.log(lp.shape[0]), float)


def _mcse(lp: NDArray[np.float64]) -> NDArray[np.float64]:
    """Monte Carlo standard error of the log mean density, by the delta method for independent draws."""
    w = np.exp(lp - lp.max(axis=0))
    mean = w.mean(axis=0)
    return np.asarray(w.std(axis=0, ddof=1) / (np.sqrt(lp.shape[0]) * mean), float)


@validation("TESTED")
@dataclass
class AnchorDraws:
    """Forecast draws of the observation mean and noise for the targets of one anchor.

    `targets` has one row per target with at least `i` (participant), `k` (endpoint index 1 to
    3) and `y` (the observed value). `mu` and `sigma` are (draws, targets). `lower`, `upper` and
    `censored` hold one value per endpoint, in endpoint order.

    Validation: TESTED.
    """

    targets: pd.DataFrame
    mu: NDArray[np.float64]
    sigma: NDArray[np.float64]
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    censored: tuple[int, int, int]

    def __post_init__(self) -> None:
        self.targets = self.targets.reset_index(drop=True)
        self.mu = np.asarray(self.mu, float)
        self.sigma = np.asarray(self.sigma, float)
        missing = [c for c in ("i", "k", "y") if c not in self.targets.columns]
        if missing:
            raise ValueError(f"targets need columns {missing}")
        if self.mu.ndim != 2 or self.mu.shape != self.sigma.shape or self.mu.shape[1] != len(self.targets):
            raise ValueError(f"mu {self.mu.shape} and sigma {self.sigma.shape} must be (draws, {len(self.targets)})")
        if len(self.lower) != 3 or len(self.upper) != 3 or len(self.censored) != 3:
            raise ValueError("lower, upper and censored need one value per endpoint")
        k = self.targets["k"].to_numpy(int)
        if np.any((k < 1) | (k > 3)):
            raise ValueError("k must lie in 1 to 3")

    def log_density(self) -> NDArray[np.float64]:
        """Return the (draws, targets) log density of each target under each draw.

        Validation: TESTED.
        """
        k = self.targets["k"].to_numpy(int) - 1
        return predictive_log_density(
            self.targets["y"].to_numpy(float),
            self.mu,
            self.sigma,
            np.asarray(self.lower, float)[k],
            np.asarray(self.upper, float)[k],
            np.asarray(self.censored, int)[k],
        )


def _csv_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def _vector(frame: pd.DataFrame, name: str, n: int) -> NDArray[np.float64]:
    return frame[[f"{name}.{j + 1}" for j in range(n)]].to_numpy(float)


@validation("TESTED")
def observation_draws(
    fitted_params: str | Path | pd.DataFrame, targets: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (draws, targets) arrays of the observation mean and noise sd for each target.

    `fitted_params` is a CmdStan CSV of a pd_v2 fit, such as the thinned file of
    `forecast_exact`, or that file read as a frame. `targets` is the target data block of a
    `HoldoutSplit`, as StanData or a mapping. The latent path is rebuilt from each draw's `s0`,
    `b_raw` and `eps`, as in forecast_gq.stan. The mean therefore equals that program's `mu_c`,
    and the sd is the noise the program draws `y_rep_c` with.

    Validation: TESTED.
    """
    d = dict(targets.data if hasattr(targets, "data") else targets)
    frame = fitted_params if isinstance(fitted_params, pd.DataFrame) else _csv_frame(fitted_params)
    n_draw, n_obs, p = len(frame), int(d["n_obs"]), int(d["P"])
    if n_draw == 0:
        raise ValueError("fitted_params holds no draws")
    mu_r, sigma_r, sigma_xi = (_vector(frame, n, 3) for n in ("mu_r", "sigma_r", "sigma_xi"))
    a, b, sigma_obs, kappa = (_vector(frame, n, 3) for n in ("a", "b", "sigma_obs", "kappa"))
    lam = frame["lambda_ledd"].to_numpy(float)
    gamma = np.stack([_vector(frame, f"Gamma.{r + 1}", 3) for r in range(3)], axis=1)
    amat = np.stack([_vector(frame, f"Amat.{r + 1}", p) for r in range(3)], axis=1) if p else None
    coupling = np.zeros((n_draw, 3, 3))
    for (r_idx, c_idx), name in _COUPLING.items():
        coupling[:, r_idx, c_idx] = frame[name].to_numpy(float)
    lower, upper = np.asarray(d["Lk"], float), np.asarray(d["Uk"], float)
    xbar = np.asarray(d["sigma_xbar"], float)
    state_dependent, ledd_in_mu = int(d["state_dependent_sigma"]), int(d["ledd_in_mu_U"])
    dmat, h = np.asarray(d["dmat"], float), np.asarray(d["h"], float)
    zmat = np.asarray(d["zmat"], float).reshape(p, int(d["N"]))
    mu, sigma = np.zeros((n_draw, n_obs)), np.zeros((n_draw, n_obs))
    for i in range(int(d["N"])):
        start_obs, count = int(d["obs_start"][i]), int(d["obs_count"][i])
        if count == 0:
            continue
        start, k_steps = int(d["step_start"][i]), int(d["K"][i])
        r = mu_r + np.einsum("srj,j->sr", gamma, dmat[:, i])
        if amat is not None:
            r = r + np.einsum("srp,p->sr", amat, zmat[:, i])
        r = r + sigma_r * frame[[f"b_raw.{j + 1}.{i + 1}" for j in range(3)]].to_numpy(float)
        s = frame[[f"s0.{j + 1}.{i + 1}" for j in range(3)]].to_numpy(float)
        path = [s]
        for kk in range(k_steps):
            t = start + kk
            dt = float(h[t])
            e = frame[[f"eps.{j + 1}.{t + 1}" for j in range(3)]].to_numpy(float)
            drive = np.logaddexp(0.0, r + np.einsum("sjm,sm->sj", coupling, expit(s)))
            xi = np.exp(-0.5 * dt * sigma_xi**2 + np.sqrt(dt) * sigma_xi * e)
            s = s + dt * drive * xi
            path.append(s)
        for q in range(start_obs, start_obs + count):
            k = int(d["obs_k"][q]) - 1
            s_k = path[int(d["obs_node"][q]) - 1][:, k]
            lin = a[:, k] + b[:, k] * s_k
            if ledd_in_mu and k == 0:
                lin = lin + lam * float(d["obs_ledd"][q])
            link = expit(lin)
            mu[:, q] = lower[k] + (upper[k] - lower[k]) * ((1 - link) if k == 1 else link)
            sd = sigma_obs[:, k]
            if state_dependent:
                sd = sd * np.exp(kappa[:, k] * (expit(s_k) - xbar[k]))
            sigma[:, q] = sd
    return mu, sigma


@validation("TESTED")
def anchor_draws(forecast: Any, split: Any) -> AnchorDraws:
    """Return the AnchorDraws of an exact-route forecast and the HoldoutSplit it forecast.

    The mean and noise come from the thinned draws that `forecast_exact` wrote, so they belong
    to the same posterior draws as `forecast.draws`.

    Validation: TESTED.
    """
    thin = forecast.metadata.get("thin_csv") if hasattr(forecast, "metadata") else None
    if not thin:
        raise ValueError("forecast carries no thinned draws; use the exact route or build AnchorDraws directly")
    mu, sigma = observation_draws(thin, split.targets)
    d = split.targets.data
    return AnchorDraws(
        targets=split.meta.copy(),
        mu=mu,
        sigma=sigma,
        lower=tuple(float(v) for v in d["Lk"]),  # type: ignore[arg-type]
        upper=tuple(float(v) for v in d["Uk"]),  # type: ignore[arg-type]
        censored=tuple(int(v) for v in d["censored"]),  # type: ignore[arg-type]
    )


def _cluster_sum(values: NDArray[np.float64], cluster: NDArray[Any]) -> tuple[float, float, int]:
    """Return the sum, its participant-cluster standard error and the number of clusters."""
    sums = pd.Series(values).groupby(cluster).sum().to_numpy(float)
    g = len(sums)
    se = float(np.sqrt(g * np.var(sums, ddof=1))) if g > 1 else float("nan")
    return float(values.sum()), se, g


def _summarise(pointwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    anchors: list[Any] = [*sorted(pointwise["anchor"].unique()), "all"]
    for anchor in anchors:
        sub = pointwise if anchor == "all" else pointwise[pointwise["anchor"] == anchor]
        for endpoint in [*C.ENDPOINTS, "all"]:
            cell = sub if endpoint == "all" else sub[sub["endpoint"] == endpoint]
            if cell.empty:
                continue
            total, se, g = _cluster_sum(cell["elpd"].to_numpy(float), cell["i"].to_numpy())
            rows.append(
                {
                    "anchor": anchor,
                    "endpoint": endpoint,
                    "n_targets": len(cell),
                    "n_participants": g,
                    "elpd": total,
                    "se": se,
                    "elpd_mean": total / len(cell),
                }
            )
    return pd.DataFrame(rows, columns=list(_SUMMARY_COLUMNS))


@validation("TESTED")
@dataclass(frozen=True)
class LFOScores:
    """Leave-future-out log predictive densities: per target, per participant block and summed.

    `pointwise` has one row per anchor and target, with `elpd` (the log predictive density) and
    its Monte Carlo standard error `mcse`. `joint` has one row per anchor and participant: the
    joint log density of all that participant's targets, and the sum of their pointwise values.
    `summary` sums `elpd` by anchor and endpoint, with a participant-cluster standard error.
    Rows with anchor "all" pool every anchor and cluster on participant across anchors.

    Validation: TESTED.
    """

    pointwise: pd.DataFrame
    joint: pd.DataFrame
    summary: pd.DataFrame

    def compare(self, other: LFOScores) -> pd.DataFrame:
        """Return the paired difference in elpd, self minus other, by anchor and endpoint.

        Targets are matched on anchor, participant, endpoint and time, and only shared targets
        are compared. The standard error clusters the differences on participant.

        Validation: TESTED.
        """
        keys = [c for c in ("anchor", "i", "k", "t") if c in self.pointwise and c in other.pointwise]
        merged = self.pointwise.merge(other.pointwise[[*keys, "elpd"]], on=keys, suffixes=("", "_other"))
        if merged.empty:
            raise ValueError("the two scores share no targets")
        diff = merged.assign(elpd=merged["elpd"] - merged["elpd_other"])
        out = _summarise(diff).rename(columns={"elpd": "elpd_diff", "elpd_mean": "elpd_diff_mean"})
        return out

    def to_markdown(self) -> str:
        """Return the summary as a Markdown table.

        Validation: TESTED.
        """
        head = ["anchor", "endpoint", "targets", "participants", "elpd", "se", "mean per target"]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for rec in self.summary.to_dict("records"):
            anchor = "all" if rec["anchor"] == "all" else f"{float(rec['anchor']):g}"
            se = f"{rec['se']:.1f}" if np.isfinite(rec["se"]) else ""
            cells = [anchor, str(rec["endpoint"]), str(rec["n_targets"]), str(rec["n_participants"])]
            values = [f"{rec['elpd']:.1f}", se, f"{rec['elpd_mean']:.3f}"]
            lines.append("| " + " | ".join([*cells, *values]) + " |")
        return "\n".join(lines) + "\n"


@validation("TESTED")
def leave_future_out(anchors: Mapping[float, AnchorDraws] | Sequence[tuple[float, AnchorDraws]]) -> LFOScores:
    """Score exact leave-future-out log predictive density over one or more anchors.

    `anchors` maps each anchor time in years to the AnchorDraws of a fit that saw data up to that
    anchor only. Each target scores log mean_s p(y | mu_s, sigma_s), with the model's truncated or
    censored density. Each participant's block of targets also scores its joint density, which
    keeps the dependence induced by the shared latent path. See Bürkner, Gabry and Vehtari (2020).

    Validation: TESTED.
    """
    items = list(anchors.items()) if isinstance(anchors, Mapping) else list(anchors)
    if not items:
        raise ValueError("give at least one anchor")
    point_frames: list[pd.DataFrame] = []
    joint_rows: list[dict[str, Any]] = []
    for anchor, draws in items:
        lp = draws.log_density()
        elpd = _log_mean_exp(lp)
        tg = draws.targets
        point_frames.append(
            tg.assign(
                anchor=float(anchor),
                endpoint=[C.ENDPOINTS[k - 1] for k in tg["k"].to_numpy(int)],
                elpd=elpd,
                mcse=_mcse(lp),
            )
        )
        part = tg["i"].to_numpy()
        for i in pd.unique(part):
            cols = np.flatnonzero(part == i)
            joint_rows.append(
                {
                    "anchor": float(anchor),
                    "i": i,
                    "n_targets": len(cols),
                    "elpd_joint": float(_log_mean_exp(lp[:, cols].sum(axis=1, keepdims=True))[0]),
                    "elpd_pointwise_sum": float(elpd[cols].sum()),
                }
            )
    pointwise = pd.concat(point_frames, ignore_index=True)
    lead = ["anchor", "i", "k", "endpoint"]
    pointwise = pointwise[[*lead, *[c for c in pointwise.columns if c not in lead]]]
    return LFOScores(pointwise=pointwise, joint=pd.DataFrame(joint_rows), summary=_summarise(pointwise))
