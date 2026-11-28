"""Pareto smoothed importance sampling, PSIS-LOO and stacking of predictive distributions.

Leave-one-out cross-validation needs a refit for every left-out point. Importance sampling avoids
the refits by reweighting the full-data posterior draws with ratios 1 / p(y_i | theta). The raw
ratios can have a heavy right tail, so Pareto smoothing replaces the largest ratios with quantiles
of a generalized Pareto fit. The fitted shape k also diagnoses when the estimate cannot be trusted.
The method follows Vehtari, Gelman and Gabry (2017) and Vehtari and colleagues (2024). The tail
length, the prior on k and the k threshold follow the loo R package and ArviZ.

Stacking finds the weights of a mixture of predictive distributions that maximise its summed log
score, as set out by Yao, Vehtari, Simpson and Gelman (2018). Pseudo-BMA+ weights come from the
same paper, with the Bayesian bootstrap.

All scores are on the scale of the data and compare models fitted to the same observations. They
are associational and say nothing about the effect of any intervention.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from symphony_dt import constants as C
from symphony_dt.evaluate.lfo import LFOScores, observation_draws, predictive_log_density
from symphony_dt.labels import validation

__all__ = [
    "LOOResult",
    "PSISResult",
    "fit_generalized_pareto",
    "fit_log_likelihood",
    "loo_compare",
    "pareto_k_threshold",
    "pseudo_bma_weights",
    "psis",
    "psis_loo",
    "relative_efficiency",
    "stacking_weights",
]

#: Prior on the Pareto shape, as in Vehtari and colleagues (2024): weight 10 at k = 0.5.
_PRIOR_K_WEIGHT = 10.0
_PRIOR_K_MEAN = 0.5


@validation("TESTED")
def fit_generalized_pareto(x: ArrayLike) -> tuple[float, float]:
    """Return the shape k and scale sigma of a generalized Pareto fit to non-negative exceedances.

    The fit is the empirical Bayes estimate of Zhang and Stephens (2009). The shape is then shrunk
    towards 0.5 by a weakly informative prior worth ten observations, as in Vehtari and colleagues
    (2024). The result matches `gpdfit` in the loo R package and ArviZ. A sample whose first
    quartile equals its minimum is far from Pareto, and returns nan for both values.

    Validation: TESTED.
    """
    xs = np.sort(np.asarray(x, float).ravel())
    n = len(xs)
    if n < 5:
        raise ValueError("need at least 5 values to fit a generalized Pareto distribution")
    if xs[0] < 0 or not np.all(np.isfinite(xs)):
        raise ValueError("exceedances must be finite and non-negative")
    m = 30 + int(np.sqrt(n))
    x_star = xs[int(n / 4 + 0.5) - 1]
    if x_star <= xs[0]:
        return float("nan"), float("nan")
    theta = 1 / xs[-1] + (1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))) / (3 * x_star)
    k_grid = np.mean(np.log1p(-theta[:, None] * xs), axis=1)
    log_lik = n * (np.log(-theta / k_grid) - k_grid - 1)
    w = np.exp(log_lik - logsumexp(log_lik))
    keep = w >= 10 * np.finfo(float).eps
    theta_hat = float(np.sum(theta[keep] * w[keep]) / np.sum(w[keep]))
    k = float(np.mean(np.log1p(-theta_hat * xs)))
    sigma = -k / theta_hat
    k = (n * k + _PRIOR_K_WEIGHT * _PRIOR_K_MEAN) / (n + _PRIOR_K_WEIGHT)
    if not np.isfinite(k):
        return float("inf"), float("nan")
    return k, float(sigma)


def _gpd_quantile(p: NDArray[np.float64], k: float, sigma: float) -> NDArray[np.float64]:
    if k == 0:
        return np.asarray(-sigma * np.log1p(-p), float)
    return np.asarray(sigma * np.expm1(-k * np.log1p(-p)) / k, float)


@validation("TESTED")
def pareto_k_threshold(n_draws: int) -> float:
    """Return the sample-size dependent threshold above which a Pareto k is unreliable.

    The threshold is min(1 - 1 / log10(S), 0.7) for S draws, from Vehtari and colleagues (2024).

    Validation: TESTED.
    """
    if n_draws < 2:
        raise ValueError("need at least 2 draws")
    return float(min(1 - 1 / np.log10(n_draws), 0.7))


def _tail_length(n_draws: int, r_eff: float) -> int:
    """Return the number of tail draws: 3 sqrt(S / r_eff), or S / 5 when that is smaller."""
    if n_draws * r_eff > 225:
        return int(np.floor(3 * np.sqrt(n_draws / r_eff)))
    return int(np.floor(n_draws / 5))


def _smooth_column(log_ratio: NDArray[np.float64], r_eff: float) -> tuple[NDArray[np.float64], float, int]:
    lw = log_ratio - float(np.max(log_ratio))
    n = len(lw)
    m = _tail_length(n, r_eff)
    if m < 5:
        raise ValueError("too few draws for Pareto smoothing; need a tail of at least 5")
    order = np.argsort(lw, kind="stable")
    tail = order[n - m :]
    cutoff = lw[order[n - m - 1]]
    tail_w = np.exp(lw[tail])
    if np.ptp(tail_w) < np.finfo(float).tiny:
        k = float("-inf")
    else:
        k, sigma = fit_generalized_pareto(tail_w - np.exp(cutoff))
        if np.isfinite(k):
            p = (np.arange(1, m + 1) - 0.5) / m
            smoothed = np.minimum(np.exp(cutoff) + _gpd_quantile(p, k, sigma), tail_w.max())
            lw[tail] = np.log(smoothed)
    return lw - logsumexp(lw), k, m


@validation("TESTED")
@dataclass(frozen=True)
class PSISResult:
    """Pareto smoothed log weights with their diagnostics, one column per set of ratios.

    `log_weights` is (draws, columns), normalised so each column's weights sum to one. `pareto_k`
    is the fitted tail shape of each column. `n_eff` is the effective sample size of the weights,
    1 / sum(w^2) times the relative efficiency. `tail_length` is the number of smoothed draws.

    Validation: TESTED.
    """

    log_weights: NDArray[np.float64]
    pareto_k: NDArray[np.float64]
    n_eff: NDArray[np.float64]
    tail_length: NDArray[np.int64]
    k_threshold: float

    @property
    def weights(self) -> NDArray[np.float64]:
        """Return the normalised weights.

        Validation: TESTED.
        """
        return np.asarray(np.exp(self.log_weights), float)


def _r_eff_vector(r_eff: ArrayLike, n: int) -> NDArray[np.float64]:
    r = np.broadcast_to(np.asarray(r_eff, float), (n,)).copy()
    if np.any(~(r > 0)):
        raise ValueError("r_eff must be positive")
    return r


@validation("TESTED")
def psis(log_ratios: ArrayLike, r_eff: ArrayLike = 1.0) -> PSISResult:
    """Pareto smooth the importance ratios of each column of a (draws, columns) array.

    A 1D input is one column. Within a column the largest 3 sqrt(S / r_eff) ratios, or S / 5 when
    that is smaller, are replaced by expected order statistics of a generalized Pareto fit. Smoothed
    values never exceed the largest raw ratio. `r_eff` is the relative efficiency of the draws, one
    value or one per column. See Vehtari and colleagues (2024).

    Validation: TESTED.
    """
    lr = np.asarray(log_ratios, float)
    if lr.ndim == 1:
        lr = lr[:, None]
    if lr.ndim != 2:
        raise ValueError("log_ratios must be (draws,) or (draws, columns)")
    if not np.all(np.isfinite(lr)):
        raise ValueError("log_ratios must be finite")
    n_draw, n_col = lr.shape
    r = _r_eff_vector(r_eff, n_col)
    lw = np.empty_like(lr)
    k = np.empty(n_col)
    m = np.empty(n_col, dtype=np.int64)
    for j in range(n_col):
        lw[:, j], k[j], m[j] = _smooth_column(lr[:, j].copy(), float(r[j]))
    n_eff = r / np.exp(logsumexp(2 * lw, axis=0))
    return PSISResult(log_weights=lw, pareto_k=k, n_eff=n_eff, tail_length=m, k_threshold=pareto_k_threshold(n_draw))


@validation("TESTED")
def relative_efficiency(log_lik: ArrayLike) -> NDArray[np.float64]:
    """Return the relative efficiency of exp(log_lik) for each observation of (chains, draws, n).

    It is the effective sample size of the likelihood values over the total number of draws, as in
    `relative_eff` of the loo R package. The effective sample size comes from ArviZ.

    Validation: TESTED.
    """
    import arviz as az
    import xarray as xr

    ll = np.asarray(log_lik, float)
    if ll.ndim != 3:
        raise ValueError("log_lik must be (chains, draws, observations)")
    lik = np.exp(ll - ll.max(axis=(0, 1), keepdims=True))
    da = xr.DataArray(lik, dims=("chain", "draw", "obs"))
    ess = np.asarray(az.ess(da, method="mean"), float).reshape(-1)
    return np.asarray(ess / (ll.shape[0] * ll.shape[1]), float)


def _group_sum(ll: NDArray[np.float64], groups: ArrayLike | None) -> tuple[NDArray[np.float64], NDArray[Any]]:
    if groups is None:
        return ll, np.arange(ll.shape[1])
    g = np.asarray(groups)
    if g.shape != (ll.shape[1],):
        raise ValueError(f"groups needs one label per observation, {ll.shape[1]} in all")
    labels, codes = np.unique(g, return_inverse=True)
    summed = np.zeros((ll.shape[0], len(labels)))
    np.add.at(summed.T, codes, ll.T)
    return summed, labels


@validation("TESTED")
@dataclass(frozen=True)
class LOOResult:
    """PSIS-LOO estimates: totals, their standard errors and one row per left-out unit.

    `pointwise` has one row per left-out observation or group, with `unit`, `elpd_loo`, `p_loo`,
    `lpd` (the in-sample log predictive density), `pareto_k`, `n_eff` and `reliable`. A unit is
    reliable when its k is at most `k_threshold`. `elpd_loo` is the summed estimate and `se` its
    standard error. `p_loo` is the effective number of parameters, lpd minus elpd_loo.

    Validation: TESTED.
    """

    pointwise: pd.DataFrame
    elpd_loo: float
    se: float
    p_loo: float
    p_loo_se: float
    n_draws: int
    k_threshold: float

    @property
    def looic(self) -> float:
        """Return -2 times elpd_loo, the information criterion scale.

        Validation: TESTED.
        """
        return -2 * self.elpd_loo

    @property
    def n_unreliable(self) -> int:
        """Return the number of units whose Pareto k exceeds the threshold.

        Validation: TESTED.
        """
        return int((~self.pointwise["reliable"]).sum())

    def k_table(self) -> pd.DataFrame:
        """Return counts and shares of units by Pareto k band: good, at most 1, and above 1.

        Validation: TESTED.
        """
        k = self.pointwise["pareto_k"].to_numpy(float)
        bands = [
            (f"(-inf, {self.k_threshold:.2f}]", k <= self.k_threshold),
            (f"({self.k_threshold:.2f}, 1]", (k > self.k_threshold) & (k <= 1)),
            ("(1, inf)", k > 1),
        ]
        rows = [{"band": b, "count": int(m.sum()), "share": float(m.mean())} for b, m in bands]
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        """Return the estimates and the Pareto k table as Markdown.

        Validation: TESTED.
        """
        lines = [
            f"PSIS-LOO from {self.n_draws} draws and {len(self.pointwise)} units.",
            "",
            "| quantity | estimate | se |",
            "|---|---|---|",
            f"| elpd_loo | {self.elpd_loo:.2f} | {self.se:.2f} |",
            f"| p_loo | {self.p_loo:.2f} | {self.p_loo_se:.2f} |",
            f"| looic | {self.looic:.2f} | {2 * self.se:.2f} |",
            "",
            "| Pareto k | count | share |",
            "|---|---|---|",
        ]
        for rec in self.k_table().to_dict("records"):
            lines.append(f"| {rec['band']} | {rec['count']} | {100 * rec['share']:.1f}% |")
        return "\n".join(lines) + "\n"


def _total(values: NDArray[np.float64]) -> tuple[float, float]:
    n = len(values)
    se = float(np.sqrt(n * np.var(values, ddof=1))) if n > 1 else float("nan")
    return float(values.sum()), se


@validation("TESTED")
def psis_loo(log_lik: ArrayLike, r_eff: ArrayLike = 1.0, groups: ArrayLike | None = None) -> LOOResult:
    """Estimate leave-one-out expected log predictive density by Pareto smoothed importance sampling.

    `log_lik` is (draws, observations), or (chains, draws, observations) with chains pooled. With
    `groups`, the log likelihoods of each group are summed first, so a whole group is left out at
    once. Grouping by participant gives leave-one-participant-out. Each unit's ratios are
    1 / p(y_i | theta_s), smoothed by `psis`. See Vehtari, Gelman and Gabry (2017).

    Validation: TESTED.
    """
    ll = np.asarray(log_lik, float)
    if ll.ndim == 3:
        ll = ll.reshape(-1, ll.shape[2])
    if ll.ndim != 2:
        raise ValueError("log_lik must be (draws, observations) or (chains, draws, observations)")
    ll, units = _group_sum(ll, groups)
    if groups is not None and np.ndim(r_eff) > 0:
        raise ValueError("give one r_eff for grouped units")
    smoothed = psis(-ll, r_eff=r_eff)
    elpd_i = logsumexp(smoothed.log_weights + ll, axis=0)
    lpd_i = logsumexp(ll, axis=0) - np.log(ll.shape[0])
    p_i = lpd_i - elpd_i
    pointwise = pd.DataFrame(
        {
            "unit": units,
            "elpd_loo": elpd_i,
            "p_loo": p_i,
            "lpd": lpd_i,
            "pareto_k": smoothed.pareto_k,
            "n_eff": smoothed.n_eff,
            "reliable": smoothed.pareto_k <= smoothed.k_threshold,
        }
    )
    elpd, se = _total(np.asarray(elpd_i, float))
    p_loo, p_se = _total(np.asarray(p_i, float))
    return LOOResult(
        pointwise=pointwise,
        elpd_loo=elpd,
        se=se,
        p_loo=p_loo,
        p_loo_se=p_se,
        n_draws=ll.shape[0],
        k_threshold=smoothed.k_threshold,
    )


@validation("TESTED")
def fit_log_likelihood(fitted_params: str | Path | pd.DataFrame, data: Any) -> tuple[NDArray[np.float64], pd.DataFrame]:
    """Return the (draws, observations) log likelihood of a pd_v2 fit at its own data, and a table.

    `fitted_params` is a CmdStan CSV of the fit, or that file read as a frame. Concatenate the
    chains first for a multi-chain fit. `data` is the StanData or mapping the model was fitted
    to. The density is the model's truncated or censored density, as in `predictive_log_density`.
    The table has one row per observation with `i` (participant), `k` (endpoint), `endpoint` and
    `y`. Pass `groups=table["i"]` to `psis_loo` for leave-one-participant-out.

    Validation: TESTED.
    """
    d = dict(data.data if hasattr(data, "data") else data)
    mu, sigma = observation_draws(fitted_params, d)
    k = np.asarray(d["obs_k"], int)
    y = np.asarray(d["y"], float)
    lower, upper = np.asarray(d["Lk"], float), np.asarray(d["Uk"], float)
    censored = np.asarray(d["censored"], int)
    i = np.full(len(y), -1, dtype=int)
    for p, (start, count) in enumerate(zip(d["obs_start"], d["obs_count"])):
        i[int(start) : int(start) + int(count)] = p
    ll = predictive_log_density(y, mu, sigma, lower[k - 1], upper[k - 1], censored[k - 1])
    table = pd.DataFrame({"i": i, "k": k, "endpoint": [C.ENDPOINTS[j - 1] for j in k], "y": y})
    return ll, table


def _pointwise_frame(scores: Mapping[str, Any]) -> pd.DataFrame:
    """Return a (units, models) frame of pointwise log scores, aligned on shared units."""
    if len(scores) < 2:
        raise ValueError("give at least two models")
    columns: dict[str, pd.Series] = {}
    for name, sc in scores.items():
        if isinstance(sc, LOOResult):
            columns[name] = sc.pointwise.set_index("unit")["elpd_loo"]
        elif isinstance(sc, LFOScores):
            pw = sc.pointwise
            keys = [c for c in ("anchor", "i", "k", "t") if c in pw.columns]
            columns[name] = pw.set_index(keys)["elpd"]
        else:
            arr = np.asarray(sc, float)
            if arr.ndim != 1:
                raise ValueError(f"pointwise scores of {name!r} must be one dimensional")
            columns[name] = pd.Series(arr)
    frame = pd.concat(columns, axis=1, join="inner")
    if frame.empty:
        raise ValueError("the models share no scored units")
    if not np.all(np.isfinite(frame.to_numpy(float))):
        raise ValueError("pointwise scores must be finite")
    return frame


@validation("TESTED")
def stacking_weights(scores: Mapping[str, Any]) -> pd.Series:
    """Return stacking weights over models from their pointwise log predictive densities.

    `scores` maps each model name to a LOOResult, an LFOScores or a 1D array of pointwise log
    densities. Units are matched by name, and only shared units count. The weights maximise
    sum_i log(sum_k w_k exp(lpd_ik)) over the simplex, following Yao and colleagues (2018).

    Validation: TESTED.
    """
    frame = _pointwise_frame(scores)
    lpd = frame.to_numpy(float)
    shift = lpd.max(axis=1, keepdims=True)
    dens = np.exp(lpd - shift)
    n_model = dens.shape[1]

    def objective(z: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        w = softmax(np.append(z, 0.0))
        mix = dens @ w
        grad_w = -(dens / mix[:, None]).sum(axis=0)
        grad_z = w * (grad_w - grad_w @ w)
        return float(-np.log(mix).sum()), np.asarray(grad_z[:-1], float)

    res = minimize(objective, np.zeros(n_model - 1), jac=True, method="BFGS", options={"gtol": 1e-10})
    w = softmax(np.append(res.x, 0.0))
    w[w < 1e-12] = 0.0
    return pd.Series(w / w.sum(), index=list(frame.columns), name="weight")


@validation("TESTED")
def pseudo_bma_weights(scores: Mapping[str, Any], n_boot: int = 1000, seed: int = 0) -> pd.Series:
    """Return pseudo-BMA+ weights over models, stabilised by the Bayesian bootstrap.

    Each bootstrap replicate reweights the shared units by a flat Dirichlet draw and scales the
    total to the number of units. The weights are the mean over replicates of the softmax of those
    totals. Set `n_boot` to 0 for plain pseudo-BMA. See Yao and colleagues (2018).

    Validation: TESTED.
    """
    frame = _pointwise_frame(scores)
    lpd = frame.to_numpy(float)
    n = lpd.shape[0]
    if n_boot == 0:
        w = softmax(lpd.sum(axis=0))
    else:
        rng = np.random.default_rng(seed)
        alpha = rng.dirichlet(np.ones(n), size=n_boot)
        w = softmax(n * alpha @ lpd, axis=1).mean(axis=0)
    return pd.Series(w, index=list(frame.columns), name="weight")


@validation("TESTED")
def loo_compare(scores: Mapping[str, Any]) -> pd.DataFrame:
    """Rank models by summed pointwise score, with paired differences from the best.

    `scores` takes the same inputs as `stacking_weights`. The table has one row per model, best
    first, with `elpd`, `se`, `elpd_diff`, `se_diff` and the stacking `weight`. The difference
    standard error uses the paired pointwise differences on the shared units.

    Validation: TESTED.
    """
    frame = _pointwise_frame(scores)
    totals = frame.sum(axis=0).sort_values(ascending=False)
    best = frame[totals.index[0]]
    weights = stacking_weights(scores)
    rows = []
    for name in totals.index:
        elpd, se = _total(frame[name].to_numpy(float))
        diff = (frame[name] - best).to_numpy(float)
        d_total, d_se = _total(diff)
        rows.append(
            {
                "model": name,
                "elpd": elpd,
                "se": se,
                "elpd_diff": d_total,
                "se_diff": 0.0 if name == totals.index[0] else d_se,
                "weight": float(weights[name]),
            }
        )
    return pd.DataFrame(rows).set_index("model")
