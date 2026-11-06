"""Scoring metrics of the SYMPHONY v2 code of record.

Each function reproduces the study's scoring code exactly. The CRPS estimator is the
empirical-distribution form mean|X - y| - 0.5 mean|X - X'| with pairs including i = j, which equals
`properscoring.crps_ensemble`. Central intervals are inclusive of their end points.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from symphony_dt.labels import validation

__all__ = [
    "EVENT_THRESHOLDS",
    "LMMFit",
    "auc_with_ties",
    "bootstrap_mean",
    "bootstrap_ratio_skill",
    "brier",
    "brier_skill",
    "central_interval",
    "cluster_bootstrap_weights",
    "crps_ensemble",
    "event_indicator",
    "event_probability",
    "fit_lmm_em",
    "interval_coverage",
    "lmm_condition",
    "mae_median",
    "percentile_interval",
]

#: Event rules per endpoint index: UPDRS-III worsening of 5 or more, MoCA decline of 2 or more.
EVENT_THRESHOLDS: Mapping[int, tuple[str, float]] = {1: ("up", 5.0), 2: ("down", 2.0)}


@validation("VALIDATED")
def crps_ensemble(samples: ArrayLike, y: float) -> float:
    """Return the CRPS of an ensemble against one observation.

    Empirical-distribution estimator, mean|X - y| - 0.5 mean|X - X'| over all pairs including
    i = j. Equal to `properscoring.crps_ensemble`.

    Validation: VALIDATED.
    """
    x = np.sort(np.asarray(samples, float))
    m = x.size
    term1 = np.mean(np.abs(x - y))
    w = 2 * np.arange(1, m + 1) - m - 1
    term2 = 2.0 * np.sum(w * x) / (m * m)
    return float(term1 - 0.5 * term2)


@validation("VALIDATED")
def central_interval(samples: ArrayLike, level: float) -> tuple[float, float]:
    """Return the central interval of the samples at `level`, by numpy's default quantiles.

    Validation: VALIDATED.
    """
    if not 0 < level < 1:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    tail = (1 - level) / 2
    lo, hi = np.quantile(np.asarray(samples, float), [tail, 1 - tail])
    return float(lo), float(hi)


@validation("VALIDATED")
def interval_coverage(samples: ArrayLike, y: float, level: float) -> float:
    """Return 1.0 when `y` lies inside the inclusive central interval at `level`, else 0.0.

    Validation: VALIDATED.
    """
    lo, hi = central_interval(samples, level)
    return float(lo <= y <= hi)


@validation("VALIDATED")
def mae_median(samples: ArrayLike, y: float) -> float:
    """Return the absolute error of the predictive median.

    Validation: VALIDATED.
    """
    return float(abs(np.quantile(np.asarray(samples, float), 0.5) - y))


@validation("VALIDATED")
def event_indicator(
    y: float, last_value: float, k: int, thresholds: Mapping[int, tuple[str, float]] = EVENT_THRESHOLDS
) -> float:
    """Return 1.0 when the observed change from `last_value` meets the event rule of endpoint `k`.

    Validation: VALIDATED.
    """
    direction, thr = thresholds[k]
    return float((y - last_value) >= thr) if direction == "up" else float((last_value - y) >= thr)


@validation("VALIDATED")
def event_probability(
    samples: ArrayLike,
    last_value: float,
    k: int,
    thresholds: Mapping[int, tuple[str, float]] = EVENT_THRESHOLDS,
) -> float:
    """Return the share of samples, rounded to the integer scale, that meet the event rule.

    Validation: VALIDATED.
    """
    direction, thr = thresholds[k]
    d = np.round(np.asarray(samples, float)) - last_value
    return float(np.mean(d >= thr)) if direction == "up" else float(np.mean(-d >= thr))


@validation("VALIDATED")
def brier(p: ArrayLike, e: ArrayLike) -> float:
    """Return the Brier score, the mean of (p - e) squared.

    Validation: VALIDATED.
    """
    return float(np.mean((np.asarray(p, float) - np.asarray(e, float)) ** 2))


@validation("VALIDATED")
def brier_skill(brier_model: float, brier_reference: float) -> float:
    """Return the Brier skill score 1 - Brier_model / Brier_reference.

    Validation: VALIDATED.
    """
    return 1 - brier_model / brier_reference


@validation("VALIDATED")
def auc_with_ties(p: ArrayLike, e: ArrayLike, w: ArrayLike | None = None) -> float:
    """Return the weighted AUC, Mann-Whitney with ties counted one half.

    `w` holds target weights, one per row; None gives equal weights. Returns NaN when either class
    has no weight.

    Validation: VALIDATED.
    """
    p = np.asarray(p, float)
    e = np.asarray(e, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    pos, neg = e == 1, e == 0
    wp, wn = (w * pos).sum(), (w * neg).sum()
    if wp <= 0 or wn <= 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps, es, ws = p[order], e[order], w[order]
    _, start = np.unique(ps, return_index=True)
    end = np.append(start[1:], len(ps))
    below, num = 0.0, 0.0
    for a, b in zip(start, end, strict=True):
        wneg = (ws[a:b] * (es[a:b] == 0)).sum()
        wpos = (ws[a:b] * (es[a:b] == 1)).sum()
        num += wpos * (below + 0.5 * wneg)
        below += wneg
    return float(num / (wp * wn))


@validation("VALIDATED")
@dataclass
class LMMFit:
    """A fitted linear mixed model: fixed effects, random-effect covariance, residual variance.

    Validation: VALIDATED.
    """

    beta: NDArray[np.float64]
    G: NDArray[np.float64]
    s2: float
    iters: int
    loglik: float

    def as_dict(self) -> dict[str, Any]:
        """Return the fit in the layout of the study's scores file.

        Validation: VALIDATED.
        """
        return {"beta": self.beta.tolist(), "G": self.G.tolist(), "s2": self.s2, "iters": self.iters}


@validation("VALIDATED")
def fit_lmm_em(
    groups: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]],
    max_iter: int = 20000,
    tol: float = 1e-12,
) -> LMMFit:
    """Fit y = (b0 + u0) + (b1 + u1) t + e by maximum likelihood with EM.

    `groups` holds one (t, y) pair of arrays per participant. u ~ N(0, G) and e ~ N(0, s2).

    Validation: VALIDATED.
    """
    XtX = np.array([[[len(t), t.sum()], [t.sum(), (t * t).sum()]] for t, _ in groups])
    Xty = np.array([[y.sum(), (t * y).sum()] for t, y in groups])
    yty = np.array([(y * y).sum() for _, y in groups])
    n = np.array([len(t) for t, _ in groups], float)
    beta = np.linalg.solve(XtX.sum(0), Xty.sum(0))
    r2 = yty - 2 * Xty @ beta + np.einsum("i,nij,j->n", beta, XtX, beta)
    s2 = max(r2.sum() / n.sum() / 2, 1e-6)
    G = np.diag([max(r2.sum() / n.sum() / 2, 1e-6), 0.1])
    ll_old = -np.inf
    ll = -np.inf
    it = 0
    for it in range(max_iter):  # noqa: B007
        Gi = np.linalg.inv(G)
        Vb = np.linalg.inv(XtX / s2 + Gi)
        Xr = Xty - XtX @ beta
        bh = np.einsum("nij,nj->ni", Vb, Xr) / s2
        M = np.linalg.inv(s2 * Gi + XtX)
        rr = yty - 2 * Xty @ beta + np.einsum("i,nij,j->n", beta, XtX, beta)
        quad = (rr - np.einsum("ni,nij,nj->n", Xr, M, Xr)) / s2
        ld = n * np.log(s2) + np.log(np.linalg.det(np.eye(2) + G @ XtX / s2))
        ll = -0.5 * np.sum(ld + quad + n * np.log(2 * np.pi))
        if abs(ll - ll_old) < tol * (1 + abs(ll)):
            break
        ll_old = ll
        beta = np.linalg.solve(XtX.sum(0), (Xty - np.einsum("nij,nj->ni", XtX, bh)).sum(0))
        G = (np.einsum("ni,nj->nij", bh, bh) + Vb).mean(0)
        c = beta[None, :] + bh
        rss = yty - 2 * np.einsum("ni,ni->n", c, Xty) + np.einsum("ni,nij,nj->n", c, XtX, c)
        s2 = (rss + np.einsum("nij,nji->n", XtX, Vb)).sum() / n.sum()
    return LMMFit(beta=beta, G=G, s2=float(s2), iters=it + 1, loglik=float(ll))


@validation("VALIDATED")
def lmm_condition(
    fit: LMMFit, t: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Condition one participant's random effects on its observations at the fitted parameters.

    Returns the posterior mean and covariance of (u0, u1). With no observations, the prior.

    Validation: VALIDATED.
    """
    Gi = np.linalg.inv(fit.G)
    if len(t) == 0:
        return np.zeros(2), fit.G
    X = np.column_stack([np.ones_like(t), t])
    Vb = np.linalg.inv(X.T @ X / fit.s2 + Gi)
    bh = Vb @ X.T @ (y - X @ fit.beta) / fit.s2
    return bh, Vb


@validation("VALIDATED")
def cluster_bootstrap_weights(n_clusters: int, B: int = 1000, seed: int = 20261021) -> NDArray[np.float64]:
    """Return a (B, n_clusters) matrix of resampling counts for a participant-cluster bootstrap.

    Validation: VALIDATED.
    """
    rng = np.random.default_rng(seed)
    return np.vstack(
        [np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters) for _ in range(B)]
    ).astype(float)


@validation("VALIDATED")
def percentile_interval(estimate: float, replicates: ArrayLike, *, nan_aware: bool = False) -> list[float]:
    """Return [estimate, 2.5% quantile, 97.5% quantile], each rounded to four decimals.

    Validation: VALIDATED.
    """
    q = np.nanquantile if nan_aware else np.quantile
    lo, hi = q(np.asarray(replicates, float), [0.025, 0.975])
    return [round(float(estimate), 4), round(float(lo), 4), round(float(hi), 4)]


@validation("VALIDATED")
def bootstrap_mean(
    values: ArrayLike, cluster: ArrayLike, W: NDArray[np.float64], *, nan_aware: bool = False
) -> list[float]:
    """Return the mean of `values` with a participant-cluster bootstrap percentile interval.

    `cluster` holds each row's cluster position, 0 to n - 1; `W` comes from
    `cluster_bootstrap_weights`.

    Validation: VALIDATED.
    """
    v = np.asarray(values, float)
    pp = np.asarray(cluster)
    n = W.shape[1]
    sums = np.bincount(pp, weights=v, minlength=n)
    cnt = np.bincount(pp, minlength=n).astype(float)
    est = sums.sum() / cnt.sum()
    bs = (W @ sums) / np.maximum(W @ cnt, 1e-12)
    return percentile_interval(est, bs, nan_aware=nan_aware)


@validation("VALIDATED")
def bootstrap_ratio_skill(
    numerator: ArrayLike, denominator: ArrayLike, cluster: ArrayLike, W: NDArray[np.float64]
) -> list[float]:
    """Return a Brier skill 1 - mean(num) / mean(den) with the same participant resampling.

    Validation: VALIDATED.
    """
    pp = np.asarray(cluster)
    n = W.shape[1]
    a = np.bincount(pp, weights=np.asarray(numerator, float), minlength=n)
    b = np.bincount(pp, weights=np.asarray(denominator, float), minlength=n)
    return percentile_interval(1 - a.sum() / b.sum(), 1 - (W @ a) / np.maximum(W @ b, 1e-12), nan_aware=True)
