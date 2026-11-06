"""The pd_v2 generative model in numpy, ported from the SYMPHONY v2 simulator.

The latent severity s_i(t) in R^3 moves forward over a step of length h (years) as
s(t + h) = s(t) + h * softplus(r_i + C x(t)) * xi, with x = inv_logit(s) and a log-normal
multiplicative innovation xi of mean one, so every increment is non-negative. Each endpoint is
observed through a logistic link scaled to the instrument range, with truncated or censored
Gaussian noise. `simulate_replicate` reproduces the study simulator draw for draw.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import expit, ndtr, ndtri

from symphony_dt import constants as C
from symphony_dt.config import ModelConfig
from symphony_dt.data.dataset import TwinDataset
from symphony_dt.data.stan_data import StanData, substeps_days
from symphony_dt.labels import validation

__all__ = [
    "SimulatedCohort",
    "VisitPattern",
    "coupling_matrix",
    "draw_prior",
    "flatten_theta",
    "initial_values",
    "simulate_cohort",
    "simulate_replicate",
    "synthetic_visit_patterns",
    "visit_patterns",
]


@validation("TESTED")
class VisitPattern(TypedDict):
    """The shape of one participant record: interval lengths and the endpoint mask.

    Validation: TESTED.
    """

    dt_days: NDArray[np.float64]
    dt_years: NDArray[np.float64]
    observed: NDArray[np.bool_]


@validation("TESTED")
def synthetic_visit_patterns(
    n: int,
    rng: np.random.Generator,
    *,
    mean_extra_visits: float = 4.0,
    max_visits: int = 14,
    interval_days: Sequence[int] = (182, 365, 365, 365, 730),
    jitter_days: int = 30,
    observed_probability: Sequence[float] = (0.85, 0.6, 0.6),
) -> list[VisitPattern]:
    """Generate synthetic visit patterns: visit counts, whole-day intervals and endpoint masks.

    The defaults are design choices for testing, not estimates from any cohort. The baseline visit
    always scores every endpoint.

    Validation: TESTED.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    probs = np.asarray(observed_probability, float)
    patterns: list[VisitPattern] = []
    for _ in range(n):
        visits = int(min(1 + rng.poisson(mean_extra_visits), max_visits))
        base = rng.choice(np.asarray(interval_days), size=visits - 1)
        days = np.maximum(base + rng.integers(-jitter_days, jitter_days + 1, size=visits - 1), 1)
        mask = rng.uniform(size=(visits, 3)) < probs
        mask[0] = True
        dt_days = days.astype(float)
        patterns.append({"dt_days": dt_days, "dt_years": dt_days / C.DAYS_PER_YEAR, "observed": mask})
    return patterns


@validation("TESTED")
def visit_patterns(dataset: TwinDataset) -> list[VisitPattern]:
    """Borrow the visit shape of a real dataset: when each participant was seen and what was scored.

    No outcome value is used. Simulating on these patterns keeps the interval distribution, the
    visit counts and the missingness pattern jointly real.

    Validation: TESTED.
    """
    wide = dataset.wide_observations()
    patterns: list[VisitPattern] = []
    for _, block in wide.groupby("participant_id", sort=True):
        times = block["time_days"].to_numpy(float)
        days = np.diff(times)
        mask = np.column_stack([block[name].notna().to_numpy() for name in C.ENDPOINTS])
        patterns.append({"dt_days": days, "dt_years": days / C.DAYS_PER_YEAR, "observed": mask})
    return patterns


@validation("VALIDATED")
def draw_prior(
    rng: np.random.Generator, n_pathway: int, hyper: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Draw the global parameters from the pd_v2 prior.

    Half-normals are drawn as the absolute value of a normal, which is what the lower-bound
    constraint plus a normal statement means in the Stan program.

    Validation: VALIDATED.
    """
    hyper = hyper or C.PRIOR_CORRECTED
    lower = np.array([C.SCALE_BOUNDS[k][0] for k in C.ENDPOINTS], float)
    upper = np.array([C.SCALE_BOUNDS[k][1] for k in C.ENDPOINTS], float)
    return {
        "mu_r": rng.normal(hyper["mu_r_mean"], hyper["mu_r_sd"], size=3),
        "Gamma": rng.normal(0.0, 0.5, size=(3, 3)),
        "Amat": rng.normal(0.0, 0.25, size=(3, n_pathway)),
        "c_MM": float(rng.normal(0, 0.25)),
        "c_CM": float(rng.normal(0, 0.25)),
        "c_CC": float(rng.normal(0, 0.25)),
        "c_CN": float(rng.normal(0, 0.25)),
        "c_NM": float(rng.normal(0, 0.25)),
        "c_NN": float(rng.normal(0, 0.25)),
        "sigma_r": np.abs(rng.normal(0.0, 0.5, size=3)),
        "sigma_xi": np.abs(rng.normal(0.0, 0.15, size=3)),
        "a": np.asarray(hyper["a_mean"], float) + rng.normal(0.0, hyper["a_sd"], size=3),
        "b": rng.lognormal(0.0, 0.5, size=3),
        "sigma_obs": np.abs(rng.normal(0.0, 0.1 * (upper - lower), size=3)),
        "kappa": rng.normal(0.0, 0.5, size=3),
        "lambda_ledd": float(rng.normal(0.0, 0.5)),
    }


@validation("VALIDATED")
def coupling_matrix(theta: Mapping[str, Any]) -> NDArray[np.float64]:
    """Return the sparse 3 x 3 coupling matrix: six free entries, three structural zeros.

    Validation: VALIDATED.
    """
    Cm = np.zeros((3, 3))
    Cm[0, 0] = theta["c_MM"]
    Cm[1, 0] = theta["c_CM"]
    Cm[1, 1] = theta["c_CC"]
    Cm[1, 2] = theta["c_CN"]
    Cm[2, 0] = theta["c_NM"]
    Cm[2, 2] = theta["c_NN"]
    return Cm


@validation("VALIDATED")
def simulate_replicate(
    patterns: Sequence[VisitPattern],
    n: int,
    theta: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    censored: Sequence[int],
    state_dependent_sigma: int,
    ledd_in_mu_U: int,
    sigma_xbar: Sequence[float] | None,
    h_max: float = C.H_MAX_YEARS,
    grainsize: int = 1,
    latents: dict[str, Any] | None = None,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate one replicate and return it as a pd_v2 Stan data block.

    The four observation-model switches are required and keyword-only, so a caller must name the
    arm it simulates. `latents`, when given, is filled with the true `s0`, `b_raw` and `eps` on the
    Stan indexing. `record`, when given, is filled with the chosen patterns and the per-participant
    LEDD level. Neither draws an extra random number, so the output is bit-identical either way.

    Validation: VALIDATED. Draw-for-draw equal to the study simulator (golden test).
    """
    xbar = np.zeros(3) if sigma_xbar is None else np.asarray(sigma_xbar, float)
    chosen = rng.integers(0, len(patterns), size=n)
    Cm = coupling_matrix(theta)
    lower = np.array([C.SCALE_BOUNDS[k][0] for k in C.ENDPOINTS], float)
    upper = np.array([C.SCALE_BOUNDS[k][1] for k in C.ENDPOINTS], float)

    n_pathway = int(np.asarray(theta["Amat"]).shape[1])
    dmat = np.vstack(
        [
            rng.normal(size=n),
            rng.binomial(1, 0.54, size=n).astype(float),
            rng.binomial(1, 0.51, size=n).astype(float),
        ]
    )
    zmat = rng.normal(size=(n_pathway, n)) if n_pathway else np.zeros((0, n))
    ledd_visit_std = rng.normal(size=n)

    K = np.zeros(n, dtype=int)
    step_start = np.zeros(n, dtype=int)
    obs_start = np.zeros(n, dtype=int)
    obs_count = np.zeros(n, dtype=int)
    s0_true: list[NDArray[np.float64]] = []
    b_raw_true: list[NDArray[np.float64]] = []
    eps_true: list[NDArray[np.float64]] = []
    h: list[float] = []
    obs_node: list[int] = []
    obs_k: list[int] = []
    y: list[float] = []
    obs_ledd: list[float] = []
    obs_visit: list[tuple[int, int]] = []

    mu_r = np.asarray(theta["mu_r"], float)
    Gamma = np.asarray(theta["Gamma"], float)
    Amat = np.asarray(theta["Amat"], float)
    sigma_r = np.asarray(theta["sigma_r"], float)
    sigma_xi = np.asarray(theta["sigma_xi"], float)
    a = np.asarray(theta["a"], float)
    b = np.asarray(theta["b"], float)
    sigma_obs = np.asarray(theta["sigma_obs"], float)
    kappa = np.asarray(theta["kappa"], float)

    for i in range(n):
        pattern = patterns[chosen[i]]
        dt = pattern["dt_years"]
        counts = substeps_days(pattern["dt_days"], h_max)
        mask = pattern["observed"]
        step_start[i] = len(h)
        obs_start[i] = len(y)

        b_raw_i = rng.normal(size=3)
        r = mu_r + Gamma @ dmat[:, i] + (Amat @ zmat[:, i] if n_pathway else 0.0) + sigma_r * b_raw_i

        s = rng.normal(size=3)
        s0_true.append(s.copy())
        b_raw_true.append(b_raw_i)
        node = 1
        states = [s.copy()]
        for interval_index, interval in enumerate(dt):
            count = int(counts[interval_index])
            if count:
                step = interval / count
                for _ in range(count):
                    x = expit(s)
                    drive = np.logaddexp(0.0, r + Cm @ x)
                    e = rng.normal(size=3)
                    xi = np.exp(-0.5 * step * sigma_xi**2 + np.sqrt(step) * sigma_xi * e)
                    s = s + step * drive * xi
                    eps_true.append(e)
                    h.append(float(step))
                node += count
            states.append(s.copy())

        for visit, state in enumerate(states):
            if visit >= mask.shape[0]:
                break
            node_index = 1 + int(counts[:visit].sum())
            for k_name in C.ENDPOINTS:
                k = C.ENDPOINT_INDEX[k_name]
                if not mask[visit, k - 1]:
                    continue
                linpred = a[k - 1] + b[k - 1] * state[k - 1]
                if ledd_in_mu_U and k == 1:
                    linpred += float(theta["lambda_ledd"]) * ledd_visit_std[i]
                link = expit(linpred)
                mu = lower[k - 1] + (upper[k - 1] - lower[k - 1]) * ((1 - link) if k == 2 else link)
                sd = sigma_obs[k - 1]
                if state_dependent_sigma:
                    sd = sd * np.exp(kappa[k - 1] * (expit(state[k - 1]) - xbar[k - 1]))
                sd = max(sd, 1e-6)
                if censored[k - 1]:
                    value = float(np.clip(rng.normal(mu, sd), lower[k - 1], upper[k - 1]))
                else:
                    lo = ndtr((lower[k - 1] - mu) / sd)
                    hi = ndtr((upper[k - 1] - mu) / sd)
                    u = float(np.clip(lo + rng.uniform() * (hi - lo), 1e-12, 1 - 1e-12))
                    value = float(mu + sd * ndtri(u))
                obs_node.append(node_index)
                obs_k.append(k)
                y.append(value)
                obs_ledd.append(float(ledd_visit_std[i]) if k == 1 else 0.0)
                obs_visit.append((i, visit))

        K[i] = len(h) - step_start[i]
        obs_count[i] = len(y) - obs_start[i]

    if latents is not None:
        latents["s0"] = np.array(s0_true, float).T
        latents["b_raw"] = np.array(b_raw_true, float).T
        latents["eps"] = np.array(eps_true, float).T if eps_true else np.zeros((3, 0), float)
    if record is not None:
        record["chosen"] = chosen.copy()
        record["ledd_std"] = ledd_visit_std.copy()
        record["obs_visit"] = list(obs_visit)

    return {
        "N": int(n),
        "P": int(n_pathway),
        "n_steps": len(h),
        "n_obs": len(y),
        "K": K.tolist(),
        "step_start": step_start.tolist(),
        "obs_start": obs_start.tolist(),
        "obs_count": obs_count.tolist(),
        "h": h,
        "obs_node": obs_node,
        "obs_k": obs_k,
        "y": y,
        "obs_ledd": obs_ledd,
        "dmat": dmat.tolist(),
        "zmat": zmat.tolist(),
        "Lk": lower.tolist(),
        "Uk": upper.tolist(),
        "mu_r_prior_mean": C.PRIOR_CORRECTED["mu_r_mean"],
        "mu_r_prior_sd": C.PRIOR_CORRECTED["mu_r_sd"],
        "a_prior_mean": list(C.PRIOR_CORRECTED["a_mean"]),
        "a_prior_sd": C.PRIOR_CORRECTED["a_sd"],
        "prior_only": 0,
        "grainsize": int(grainsize),
        "censored": list(censored),
        "state_dependent_sigma": int(state_dependent_sigma),
        "ledd_in_mu_U": int(ledd_in_mu_U),
        "sigma_xbar": [float(v) for v in xbar],
    }


@validation("VALIDATED")
def flatten_theta(theta: Mapping[str, Any]) -> dict[str, float]:
    """Return the global parameters as named scalars, the granularity SBC ranks and gates check.

    Names follow Stan's convention: `mu_r[1]`, `Gamma[2,3]`.

    Validation: VALIDATED.
    """
    flat: dict[str, float] = {}
    for name, value in theta.items():
        array = np.atleast_1d(np.asarray(value, float))
        if array.size == 1:
            flat[name] = float(array.ravel()[0])
        elif array.ndim == 1:
            for j, v in enumerate(array, start=1):
                flat[f"{name}[{j}]"] = float(v)
        else:
            for r in range(array.shape[0]):
                for c in range(array.shape[1]):
                    flat[f"{name}[{r + 1},{c + 1}]"] = float(array[r, c])
    return flat


@validation("VALIDATED")
def initial_values(
    theta_like: Mapping[str, Any], rng: np.random.Generator, n: int = 0, n_steps: int = 0, jitter: float = 0.1
) -> dict[str, Any]:
    """Return inits at the prior location with per-chain jitter, not Stan's uniform(-2, 2) box.

    The default box puts the boundary densities at minus infinity. Starting at the prior mean,
    jittered per chain, keeps the chains distinct so R-hat stays meaningful.

    Validation: VALIDATED.
    """
    n_pathway = int(np.asarray(theta_like["Amat"]).shape[1])

    def j(shape: Any) -> NDArray[np.float64]:
        return np.asarray(rng.normal(0.0, jitter, size=shape), float)

    upper = np.array([132.0, 30.0, 69.0])
    init: dict[str, Any] = {
        "mu_r": (C.PRIOR_CORRECTED["mu_r_mean"] + j(3)).tolist(),
        "Gamma": (0.0 + j((3, 3))).tolist(),
        "Amat": (0.0 + j((3, n_pathway))).tolist(),
        "c_MM": float(j(())),
        "c_CM": float(j(())),
        "c_CC": float(j(())),
        "c_CN": float(j(())),
        "c_NM": float(j(())),
        "c_NN": float(j(())),
        "sigma_r": np.abs(0.3 + j(3)).clip(0.05).tolist(),
        "sigma_xi": np.abs(0.1 + 0.2 * j(3)).clip(0.01, 0.3).tolist(),
        "a": (np.asarray(C.PRIOR_CORRECTED["a_mean"], float) + j(3)).tolist(),
        "b": np.abs(1.0 + j(3)).clip(0.5, 2.0).tolist(),
        "sigma_obs": (0.05 * upper * (1.0 + 0.1 * j(3))).clip(0.5).tolist(),
        "kappa": (0.0 + j(3)).tolist(),
        "lambda_ledd": float(j(())),
    }
    if n:
        init["s0"] = j((3, n)).tolist()
        init["b_raw"] = np.zeros((3, n)).tolist()
    if n_steps:
        init["eps"] = np.zeros((3, n_steps)).tolist()
    return init


@validation("TESTED")
@dataclass
class SimulatedCohort:
    """A simulated cohort: the dataset, the generating Stan data, the true parameters and latents.

    Validation: TESTED.
    """

    dataset: TwinDataset
    stan_data: StanData
    theta: dict[str, Any]
    latents: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def truth(self) -> dict[str, float]:
        """Return the true global parameters as named scalars.

        Validation: TESTED.
        """
        return flatten_theta(self.theta)


@validation("TESTED")
def simulate_cohort(
    n_participants: int,
    *,
    seed: int = C.MASTER_SEED,
    config: ModelConfig | None = None,
    patterns: Sequence[VisitPattern] | None = None,
    theta: Mapping[str, Any] | None = None,
    n_pathway: int = 0,
    cohort_share_pd: float = 0.4,
) -> SimulatedCohort:
    """Simulate a synthetic cohort from the pd_v2 generative model.

    Draws the global parameters from the prior unless `theta` is given, then simulates every
    participant on `patterns` (synthetic ones by default). The returned dataset carries age as
    65 + 10 x the simulated centred age in decades, and LEDD as 300 + 200 x the simulated
    standardised level, so a rebuilt data block sits close to the generating one.

    Validation: TESTED.
    """
    cfg = config or ModelConfig.primary()
    rng = np.random.default_rng(seed)
    pats = list(patterns) if patterns is not None else synthetic_visit_patterns(max(n_participants, 50), rng)
    th = dict(theta) if theta is not None else draw_prior(rng, n_pathway, cfg.hyperparameters())
    latents: dict[str, Any] = {}
    record: dict[str, Any] = {}
    data = simulate_replicate(
        pats,
        n_participants,
        th,
        rng,
        censored=cfg.censored,
        state_dependent_sigma=cfg.state_dependent_sigma,
        ledd_in_mu_U=cfg.ledd_in_mu_U,
        sigma_xbar=cfg.sigma_xbar,
        h_max=cfg.h_max_years,
        grainsize=1,
        latents=latents,
        record=record,
    )
    hyper = cfg.hyperparameters()
    data.update(
        {
            "mu_r_prior_mean": hyper["mu_r_mean"],
            "mu_r_prior_sd": hyper["mu_r_sd"],
            "a_prior_mean": list(hyper["a_mean"]),
            "a_prior_sd": hyper["a_sd"],
        }
    )

    width = max(5, len(str(n_participants)))
    ids = [f"{i + 1:0{width}d}" for i in range(n_participants)]
    dmat = np.asarray(data["dmat"], float)
    cohort = np.where(rng.uniform(size=n_participants) < cohort_share_pd, "PD", "Prodromal")
    participants = pd.DataFrame(
        {
            "participant_id": ids,
            "cohort": cohort,
            "age_baseline_years": 65.0 + 10.0 * dmat[0],
            "sex": dmat[1].astype(int),
            "genotype_assay_available": dmat[2].astype(int),
        }
    )
    visit_rows: list[tuple[str, int, float]] = []
    for i, pid in enumerate(ids):
        pattern = pats[int(record["chosen"][i])]
        times = np.concatenate([[0], np.cumsum(pattern["dt_days"])]).astype(int)
        ledd = 300.0 + 200.0 * float(record["ledd_std"][i])
        visit_rows.extend((pid, int(t), ledd) for t in times)
    visits = pd.DataFrame(visit_rows, columns=["participant_id", "time_days", "ledd"])
    obs_rows = []
    for (i, visit), k, value in zip(record["obs_visit"], data["obs_k"], data["y"], strict=True):
        pattern = pats[int(record["chosen"][i])]
        t = int(np.concatenate([[0], np.cumsum(pattern["dt_days"])])[visit])
        obs_rows.append((ids[i], t, C.ENDPOINTS[k - 1], float(value)))
    observations = pd.DataFrame(obs_rows, columns=["participant_id", "time_days", "endpoint", "value"])
    markers = pd.DataFrame(
        [
            (ids[i], f"pathway_{p + 1}", float(data["zmat"][p][i]))
            for p in range(n_pathway)
            for i in range(n_participants)
        ],
        columns=["participant_id", "name", "value"],
    )
    dataset = TwinDataset(
        participants,
        observations,
        visits,
        markers,
        metadata={"source": "symphony_dt.simulate_cohort", "seed": seed, "synthetic": True},
    )
    stan = StanData(data=data, audit={"source": "simulate_cohort", "seed": seed}, participant_ids=ids)
    return SimulatedCohort(dataset=dataset, stan_data=stan, theta=th, latents=latents, seed=seed)
