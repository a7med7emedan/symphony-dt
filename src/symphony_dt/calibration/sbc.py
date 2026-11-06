"""Simulation-based calibration (SBC): ranks, uniformity tests and a judge with multiplicity control.

Talts et al. (2018) take the rank of the true value among L posterior draws; under a correctly
calibrated sampler it is uniform on 0 to L. The study judged each global parameter by a chi-squared
test over 20 equal bins and corrected for the number of parameters with Bonferroni, because the
acceptance rule fails on any single rejection. `judge` makes that choice explicit, offers
Benjamini-Hochberg and no correction, and always reports the family-wise false-rejection risk of an
uncorrected judge, 1 - (1 - alpha)^m.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from symphony_dt.labels import validation

__all__ = [
    "CHI2_ALPHA",
    "N_BINS",
    "RANK_DRAWS",
    "SBCDesign",
    "SBCResult",
    "SBCVerdict",
    "chi_squared_uniformity",
    "judge",
    "ks_uniformity",
    "pd_v2_prior_design",
    "rank_histogram_bands",
    "run_sbc",
    "sbc_rank",
    "thin_for_rank",
]

#: Family-wise level of the SBC judge.
CHI2_ALPHA = 0.01
#: Number of equal rank bins.
N_BINS = 20
#: Draws kept per replicate for a rank; (RANK_DRAWS + 1) is a multiple of N_BINS.
RANK_DRAWS = 199


@validation("VALIDATED")
def sbc_rank(draws: ArrayLike, truth: float) -> int:
    """Return the SBC rank: how many posterior draws fall below the truth.

    Validation: VALIDATED.
    """
    return int(np.sum(np.asarray(draws) < truth))


@validation("VALIDATED")
def thin_for_rank(draws: ArrayLike, target: int = RANK_DRAWS) -> NDArray[Any]:
    """Thin to `target` draws at uniform spacing, so the rank uses near-independent values.

    Validation: VALIDATED.
    """
    arr = np.asarray(draws).ravel()
    if arr.size <= target:
        return np.asarray(arr)
    index = np.linspace(0, arr.size - 1, target).round().astype(int)
    return np.asarray(arr[index])


@validation("VALIDATED")
def chi_squared_uniformity(
    ranks: ArrayLike, n_draws: int, bins: int = N_BINS, alpha: float | None = None
) -> dict[str, Any]:
    """Test rank uniformity by chi-squared over `bins` equal bins.

    `alpha` is the per-parameter threshold; None uses the family level, which is right only for a
    single test. The uncorrected p-value is always returned.

    Validation: VALIDATED.
    """
    from scipy.stats import chisquare

    r = np.asarray(ranks)
    if (n_draws + 1) % bins != 0:
        raise ValueError(f"{n_draws + 1} possible ranks do not divide into {bins} equal bins")
    edges = np.linspace(0, n_draws + 1, bins + 1)
    counts, _ = np.histogram(r, bins=edges)
    expected = np.full(bins, len(r) / bins)
    stat, p = chisquare(counts, expected)
    threshold = CHI2_ALPHA if alpha is None else float(alpha)
    return {
        "chi2": float(stat),
        "p": float(p),
        "bins": bins,
        "n_ranks": len(r),
        "uniform": bool(p >= threshold),
        "alpha_applied": threshold,
        "uniform_uncorrected": bool(p >= CHI2_ALPHA),
        "counts": [int(c) for c in counts],
    }


@validation("TESTED")
def ks_uniformity(ranks: ArrayLike, n_draws: int) -> dict[str, float]:
    """Kolmogorov-Smirnov test of ranks against the uniform distribution on 0 to n_draws.

    Ranks are mapped to (rank + 0.5) / (n_draws + 1), the midpoints of their cells, which makes the
    test slightly conservative for discrete ranks.

    Validation: TESTED.
    """
    from scipy.stats import kstest

    u = (np.asarray(ranks, float) + 0.5) / (n_draws + 1)
    res = kstest(u, "uniform")
    return {"ks": float(res.statistic), "p": float(res.pvalue), "n_ranks": int(u.size)}


@validation("TESTED")
def rank_histogram_bands(n_ranks: int, bins: int = N_BINS, level: float = 0.99) -> tuple[float, float]:
    """Return the binomial band for one bin count of a uniform rank histogram.

    Each bin count is Binomial(n_ranks, 1 / bins); the band holds its central `level` mass.

    Validation: TESTED.
    """
    from scipy.stats import binom

    tail = (1 - level) / 2
    return (float(binom.ppf(tail, n_ranks, 1 / bins)), float(binom.ppf(1 - tail, n_ranks, 1 / bins)))


@validation("TESTED")
@dataclass
class SBCVerdict:
    """The judge's verdict on a family of uniformity tests.

    Validation: TESTED.
    """

    method: str
    alpha: float
    m: int
    rejected: list[str]
    adjusted_threshold: dict[str, float]
    fwer_uncorrected: float
    p_values: dict[str, float]

    @property
    def passed(self) -> bool:
        """True when no parameter is rejected."""
        return not self.rejected

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record.

        Validation: TESTED.
        """
        return {
            "method": self.method,
            "alpha": self.alpha,
            "m": self.m,
            "passed": self.passed,
            "rejected": self.rejected,
            "fwer_uncorrected": self.fwer_uncorrected,
            "adjusted_threshold": self.adjusted_threshold,
            "p_values": self.p_values,
        }


@validation("TESTED")
def judge(
    per_parameter_p: Mapping[str, float],
    *,
    alpha: float = CHI2_ALPHA,
    method: Literal["bonferroni", "bh", "none"] = "bonferroni",
) -> SBCVerdict:
    """Judge a family of per-parameter uniformity p-values with explicit multiplicity control.

    `bonferroni` controls the family-wise error rate at `alpha` and is right when any single
    rejection fails the campaign. `bh` (Benjamini-Hochberg) controls the false discovery rate.
    `none` applies `alpha` to each test; its family-wise false-rejection risk under independence,
    1 - (1 - alpha)^m, is reported whatever the method.

    Validation: TESTED.
    """
    names = sorted(per_parameter_p)
    m = len(names)
    if m == 0:
        raise ValueError("no p-values to judge")
    p = np.array([per_parameter_p[n] for n in names], float)
    if method == "bonferroni":
        thr = dict.fromkeys(names, alpha / m)
        rejected = [n for n, v in zip(names, p, strict=True) if v < alpha / m]
    elif method == "none":
        thr = dict.fromkeys(names, alpha)
        rejected = [n for n, v in zip(names, p, strict=True) if v < alpha]
    elif method == "bh":
        order = np.argsort(p, kind="mergesort")
        crit = alpha * np.arange(1, m + 1) / m
        below = np.nonzero(p[order] <= crit)[0]
        cut = int(np.max(below)) + 1 if below.size else 0
        rejected = sorted(names[i] for i in order[:cut])
        thr = {names[i]: float(crit[rank]) for rank, i in enumerate(order)}
    else:
        raise ValueError(f"unknown method {method!r}")
    return SBCVerdict(
        method=method,
        alpha=alpha,
        m=m,
        rejected=rejected,
        adjusted_threshold=thr,
        fwer_uncorrected=float(1 - (1 - alpha) ** m),
        p_values={n: float(v) for n, v in zip(names, p, strict=True)},
    )


@validation("TESTED")
@dataclass
class SBCDesign:
    """An SBC design: how to draw a truth and data, and how to fit.

    `generator(rng)` returns (data, truth) with truth a dict of named scalars. `fitter(data, seed)`
    returns a dict of named posterior draws (one array per parameter) or None for a failed fit.
    For the posterior regime, the fitting prior must equal the generating distribution; set
    `fitting_prior_matches_generator=True` only when that is true by construction.

    Validation: TESTED. The posterior regime is EXPERIMENTAL.
    """

    generator: Callable[[np.random.Generator], tuple[Any, dict[str, float]]]
    fitter: Callable[[Any, int], dict[str, NDArray[Any]] | None]
    regime: Literal["prior", "posterior"] = "prior"
    fitting_prior_matches_generator: bool = True
    rank_draws: int = RANK_DRAWS
    bins: int = N_BINS

    def __post_init__(self) -> None:
        if self.regime == "posterior" and not self.fitting_prior_matches_generator:
            raise ValueError(
                "posterior-regime SBC is valid only when the fitting prior equals the "
                "generating distribution; set fitting_prior_matches_generator=True "
                "once that holds"
            )
        if (self.rank_draws + 1) % self.bins:
            raise ValueError("rank_draws + 1 must be a multiple of bins")


@validation("TESTED")
@dataclass
class SBCResult:
    """Ranks per replicate and parameter, failures, uniformity tests and the verdict.

    Validation: TESTED.
    """

    ranks: pd.DataFrame
    failed: list[int]
    tests: dict[str, dict[str, Any]]
    verdict: SBCVerdict
    ks: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary.

        Validation: TESTED.
        """
        return {
            "n_replicates": len(self.ranks),
            "failed_replicates": self.failed,
            "verdict": self.verdict.to_dict(),
            "tests": self.tests,
            "ks": self.ks,
        }


@validation("TESTED")
def run_sbc(
    design: SBCDesign,
    *,
    n_replicates: int,
    seed: int = 0,
    method: Literal["bonferroni", "bh", "none"] = "bonferroni",
    alpha: float = CHI2_ALPHA,
) -> SBCResult:
    """Run an SBC campaign: draw, fit, rank, then judge every parameter.

    A failed fit is recorded by index and excluded from the ranks; it is never refitted until it
    passes, which would select on the outcome. The share of failures belongs in any report.

    Validation: TESTED.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    failed: list[int] = []
    for rep in range(n_replicates):
        data, truth = design.generator(rng)
        draws = design.fitter(data, seed + rep + 1)
        if draws is None:
            failed.append(rep)
            continue
        row: dict[str, Any] = {"replicate": rep}
        for name, value in truth.items():
            if name in draws:
                row[name] = sbc_rank(thin_for_rank(draws[name], design.rank_draws), value)
        rows.append(row)
    ranks = pd.DataFrame(rows)
    names = [c for c in ranks.columns if c != "replicate"]
    if not names:
        raise ValueError("no replicate produced a rank")
    tests = {n: chi_squared_uniformity(ranks[n].to_numpy(), design.rank_draws, design.bins) for n in names}
    ks = {n: ks_uniformity(ranks[n].to_numpy(), design.rank_draws) for n in names}
    verdict = judge({n: t["p"] for n, t in tests.items()}, alpha=alpha, method=method)
    return SBCResult(ranks=ranks, failed=failed, tests=tests, verdict=verdict, ks=ks)


@validation("TESTED")
def pd_v2_prior_design(
    n_participants: int = 60,
    *,
    chains: int = 4,
    iter_warmup: int = 300,
    iter_sampling: int = 300,
    patterns: Sequence[Any] | None = None,
    config: Any = None,
    output_root: str | None = None,
) -> SBCDesign:
    """Return a prior-regime SBC design for pd_v2 on synthetic visit patterns.

    The fitter runs `symphony_dt.fit.fit` and returns None when the convergence gates fail.

    Validation: TESTED.
    """
    import tempfile
    from pathlib import Path

    from symphony_dt.config import ModelConfig
    from symphony_dt.data.simulate import (
        draw_prior,
        flatten_theta,
        simulate_replicate,
        synthetic_visit_patterns,
    )
    from symphony_dt.data.stan_data import StanData

    cfg = config or ModelConfig.primary()
    pats = list(patterns) if patterns is not None else synthetic_visit_patterns(200, np.random.default_rng(1))

    def generator(rng: np.random.Generator) -> tuple[Any, dict[str, float]]:
        theta = draw_prior(rng, 0, cfg.hyperparameters())
        data = simulate_replicate(
            pats,
            n_participants,
            theta,
            rng,
            censored=cfg.censored,
            state_dependent_sigma=cfg.state_dependent_sigma,
            ledd_in_mu_U=cfg.ledd_in_mu_U,
            sigma_xbar=cfg.sigma_xbar,
        )
        flat = {
            k.replace("[", ".").replace("]", "").replace(",", "."): v
            for k, v in flatten_theta({k: v for k, v in theta.items() if k != "Amat"}).items()
        }
        return StanData(data), flat

    def fitter(data: Any, seed: int) -> dict[str, NDArray[Any]] | None:
        from symphony_dt.fit import fit

        root = Path(output_root or tempfile.mkdtemp(prefix="sdt-sbc-"))
        result = fit(
            data,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            output_dir=root / f"rep-{seed}",
        )
        if not result.gates().meets_min_retained:
            return None
        frames = [pd.read_csv(f, comment="#") for f in result.csv_files]
        frame = pd.concat(frames, ignore_index=True)
        return {c: frame[c].to_numpy() for c in frame.columns if not c.endswith("__")}

    return SBCDesign(generator=generator, fitter=fitter, regime="prior")
