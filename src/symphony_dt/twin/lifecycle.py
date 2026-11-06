"""A living digital twin of one participant: observe, update, forecast, explain, schedule, persist.

The twin's state is a weighted particle cloud. For each of K population draws theta_k it keeps M particles of
the participant's latent state s (3 domains) and rate r. A new visit is absorbed in four steps:

1. propagate every particle from the last visit to the new one with the pd_v2 dynamics, on the same integer-day
   sub-step grid the model uses (rule D39);
2. weight each particle by the likelihood of the new scores under the pd_v2 observation model;
3. reweight the population draws by the visit's marginal likelihood under each draw, so the twin also learns which
   population parameters suit this person (a nested, SMC-squared-like scheme over a fixed set of draws);
4. resample a draw's particles when their effective sample size falls below half, then move the static rate by
   Liu and West (2001) kernel shrinkage, so the cloud does not collapse.

Before absorbing a visit, the twin computes the one-step-ahead PIT of each new score from its own predictive
distribution. Attached to a `CalibrationMonitor`, these PITs give an anytime-valid test of the twin's calibration;
attached to `AdaptiveConformal`, they keep interval coverage on target.

Forecast uncertainty is split by the law of total variance into a population part (between draws), a person part
(between particles within a draw) and a measurement part (observation noise). Because pd_v2's latent state only
rises and its links are monotone, the expected score is monotone in time, so the probability of crossing a clinical
threshold by time t is the probability that the expected score is past it at t. `next_visit` ranks candidate visit
times by the preposterior expected reduction in forecast variance at a target horizon (Lindley, 1956).

Everything here is associational and EXPERIMENTAL. No output is a treatment recommendation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import expit, logsumexp

from symphony_dt import constants as C
from symphony_dt.config import ModelConfig
from symphony_dt.data.dataset import TwinDataset
from symphony_dt.data.simulate import coupling_matrix
from symphony_dt.data.stan_data import Standardisation, build_stan_data, substeps_days
from symphony_dt.forecast import Forecast, _endpoint_logpdf, _endpoint_rng, global_draws_from_csv
from symphony_dt.labels import validation
from symphony_dt.twin.monitor import AdaptiveConformal, CalibrationMonitor

__all__ = ["ParticipantTwin", "PopulationDraws", "Visit"]

Array = NDArray[np.float64]


@validation("EXPERIMENTAL")
@dataclass
class PopulationDraws:
    """K draws of the pd_v2 global parameters, stacked as arrays for vectorised updates.

    Validation: EXPERIMENTAL.
    """

    mu_r: Array
    Gamma: Array
    Amat: Array
    C: Array
    sigma_r: Array
    sigma_xi: Array
    a: Array
    b: Array
    sigma_obs: Array
    kappa: Array
    lambda_ledd: Array
    source: str = ""

    @property
    def k(self) -> int:
        """Number of draws."""
        return int(self.mu_r.shape[0])

    @classmethod
    def from_thetas(cls, thetas: Sequence[Mapping[str, Any]], source: str = "") -> PopulationDraws:
        """Stack parameter dicts, as returned by `global_draws_from_csv` or `draw_prior`.

        Validation: EXPERIMENTAL.
        """

        def stack(name: str, shape: tuple[int, ...]) -> Array:
            return np.stack([np.asarray(t.get(name, np.zeros(shape)), float).reshape(shape) for t in thetas])

        p = int(np.asarray(thetas[0].get("Amat", np.zeros((3, 0)))).shape[1])
        return cls(
            mu_r=stack("mu_r", (3,)),
            Gamma=stack("Gamma", (3, 3)),
            Amat=stack("Amat", (3, p)),
            C=np.stack([coupling_matrix(t) for t in thetas]),
            sigma_r=stack("sigma_r", (3,)),
            sigma_xi=stack("sigma_xi", (3,)),
            a=stack("a", (3,)),
            b=stack("b", (3,)),
            sigma_obs=stack("sigma_obs", (3,)),
            kappa=stack("kappa", (3,)),
            lambda_ledd=np.asarray([float(t["lambda_ledd"]) for t in thetas]),
            source=source,
        )

    @classmethod
    def from_fit(cls, fit: Any, *, n_draws: int = 200, n_pathway: int = 0) -> PopulationDraws:
        """Read evenly spaced draws from a FitResult or a list of chain CSV files.

        Validation: EXPERIMENTAL.
        """
        files = fit.csv_files if hasattr(fit, "csv_files") else list(fit)
        source = str(getattr(fit, "output_dir", "")) or ",".join(Path(f).name for f in files)
        return cls.from_thetas(global_draws_from_csv(files, n_pathway=n_pathway, max_draws=n_draws), source)

    def to_arrays(self) -> dict[str, Array]:
        """Return every array by name, for saving.

        Validation: EXPERIMENTAL.
        """
        return {
            k: getattr(self, k)
            for k in (
                "mu_r",
                "Gamma",
                "Amat",
                "C",
                "sigma_r",
                "sigma_xi",
                "a",
                "b",
                "sigma_obs",
                "kappa",
                "lambda_ledd",
            )
        }


@validation("EXPERIMENTAL")
@dataclass(frozen=True)
class Visit:
    """One visit: whole days since baseline, the scores measured and the standardised LEDD.

    Validation: EXPERIMENTAL.
    """

    time_days: int
    scores: dict[str, float]
    ledd_std: float | None = None


def _systematic(weights: Array, rng: np.random.Generator) -> NDArray[np.int64]:
    m = weights.size
    positions = (rng.uniform() + np.arange(m)) / m
    return np.minimum(np.searchsorted(np.cumsum(weights), positions), m - 1)


@validation("EXPERIMENTAL")
class ParticipantTwin:
    """A self-updating twin of one participant over a fixed set of population draws.

    Build it with `ParticipantTwin.create(population, history, standardisation=...)`; the history is replayed
    visit by visit through `observe`, so creation and updating share one code path and one audit log.

    Validation: EXPERIMENTAL.
    """

    def __init__(
        self,
        population: PopulationDraws,
        d_i: Array,
        z_i: Array,
        *,
        participant_id: str = "",
        config: ModelConfig | None = None,
        n_particles: int = 1000,
        seed: int = 0,
        monitor: CalibrationMonitor | None = None,
        conformal: AdaptiveConformal | None = None,
        n_moves: int = 4,
    ) -> None:
        self.pop = population
        self.cfg = config or ModelConfig.primary()
        self.participant_id = participant_id
        self.m = int(n_particles)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.monitor = monitor
        self.conformal = conformal
        self.n_moves = int(n_moves)
        self.lower = np.array([C.SCALE_BOUNDS[k][0] for k in C.ENDPOINTS])
        self.upper = np.array([C.SCALE_BOUNDS[k][1] for k in C.ENDPOINTS])
        k = population.k
        base = population.mu_r + np.einsum("kij,j->ki", population.Gamma, d_i)
        if z_i.size:
            base = base + np.einsum("kij,j->ki", population.Amat, z_i)
        self.d_i, self.z_i = d_i, z_i
        self.base = base
        # The person's whole latent state in non-centred form: every entry is N(0, 1) a priori.
        self.S0 = self.rng.normal(size=(k, self.m, 3))
        self.B = self.rng.normal(size=(k, self.m, 3))
        self.E = np.zeros((k, self.m, 3, 0))
        self.h = np.zeros(0)
        self.records: list[tuple[int, int, float, float]] = []  # (node step, endpoint index, value, ledd)
        self.beta = np.full(k, 0.3)  # pCN step size per draw, adapted online
        self.r = self._rate(self.B)
        self.s = self.S0.copy()
        self.logw = np.full((k, self.m), -np.log(self.m))
        self.logv = np.full(k, -np.log(k))
        self.time_days: int | None = None
        self.last_ledd = 0.0
        self.history: list[Visit] = []
        self.audit: list[dict[str, Any]] = []

    # ---------------------------------------------------------------------------------------- creation
    @classmethod
    def create(
        cls,
        population: PopulationDraws | Any,
        history: TwinDataset,
        *,
        standardisation: Standardisation | None = None,
        config: ModelConfig | None = None,
        n_particles: int = 1000,
        seed: int = 0,
        monitor: CalibrationMonitor | None = None,
        conformal: AdaptiveConformal | None = None,
        pathway: Sequence[str] = (),
    ) -> ParticipantTwin:
        """Create a twin for the single participant in `history` and replay their visits.

        `population` is a PopulationDraws or a FitResult. `standardisation` must be the population fit's.

        Validation: EXPERIMENTAL.
        """
        if len(history.participants) != 1:
            raise ValueError("history must hold exactly one participant")
        pop = (
            population
            if isinstance(population, PopulationDraws)
            else PopulationDraws.from_fit(population, n_pathway=len(pathway))
        )
        cfg = config or ModelConfig.primary()
        sd = build_stan_data(history, config=cfg, standardisation=standardisation, pathway=pathway)
        d = sd.data
        twin = cls(
            pop,
            np.asarray(d["dmat"], float)[:, 0],
            np.asarray(d["zmat"], float)[:, 0] if int(d["P"]) else np.zeros(0),
            participant_id=sd.participant_ids[0],
            config=cfg,
            n_particles=n_particles,
            seed=seed,
            monitor=monitor,
            conformal=conformal,
        )
        for visit in _visits_from_stan(d):
            twin.observe(visit, monitor_outcomes=False)
        twin.audit.append(
            {
                "event": "created",
                "at": _now(),
                "visits": len(twin.history),
                "population": pop.source,
                "draws": pop.k,
                "particles": twin.m,
            }
        )
        return twin

    # ---------------------------------------------------------------------------------------- dynamics
    def _propagate(self, s: Array, r: Array, dt_days: int, rng: np.random.Generator) -> Array:
        count = int(substeps_days(np.array([dt_days]), self.cfg.h_max_years)[0]) if dt_days else 0
        if not count:
            return s
        h = dt_days / C.DAYS_PER_YEAR / count
        sig = self.pop.sigma_xi[:, None, :]
        for _ in range(count):
            x = expit(s)
            drive = np.logaddexp(0.0, r + np.einsum("kij,kmj->kmi", self.pop.C, x))
            xi = np.exp(-0.5 * h * sig**2 + np.sqrt(h) * sig * rng.normal(size=s.shape))
            s = s + h * drive * xi
        return s

    def _mean(self, s_k: Array, k: int, ledd: float) -> Array:
        lin = self.pop.a[:, k][:, None] + self.pop.b[:, k][:, None] * s_k
        if self.cfg.ledd_in_mu_U and k == 0:
            lin = lin + self.pop.lambda_ledd[:, None] * ledd
        link = expit(lin)
        span = self.upper[k] - self.lower[k]
        return np.asarray(self.lower[k] + span * ((1 - link) if k == 1 else link), float)

    def _sd(self, s_k: Array, k: int) -> Array:
        sd = np.broadcast_to(self.pop.sigma_obs[:, k][:, None], s_k.shape).copy()
        if self.cfg.state_dependent_sigma:
            sd = sd * np.exp(self.pop.kappa[:, k][:, None] * (expit(s_k) - self.cfg.sigma_xbar[k]))
        return np.maximum(sd, 1e-6)

    def _weights(self) -> tuple[Array, Array]:
        v = np.exp(self.logv - logsumexp(self.logv))
        w = np.exp(self.logw - logsumexp(self.logw, axis=1, keepdims=True))
        return v, w

    # ---------------------------------------------------------------------------------------- update
    def observe(self, visit: Visit | Mapping[str, Any], *, monitor_outcomes: bool = True) -> dict[str, Any]:
        """Absorb one visit and return its audit entry.

        With `monitor_outcomes`, the one-step-ahead PIT of each score is computed before the update and passed to
        the attached monitor and conformal layer.

        Validation: EXPERIMENTAL.
        """
        v = (
            visit
            if isinstance(visit, Visit)
            else Visit(int(visit["time_days"]), dict(visit["scores"]), visit.get("ledd_std"))
        )
        if self.time_days is not None and v.time_days < self.time_days:
            raise ValueError(f"visit at day {v.time_days} precedes the twin's state at day {self.time_days}")
        if v.ledd_std is not None:
            self.last_ledd = float(v.ledd_std)
        dt = 0 if self.time_days is None else v.time_days - self.time_days
        count = int(substeps_days(np.array([dt]), self.cfg.h_max_years)[0]) if dt else 0
        if count:
            new_e = self.rng.normal(size=(*self.E.shape[:3], count))
            self.h = np.concatenate([self.h, np.full(count, dt / C.DAYS_PER_YEAR / count)])
            self.E = np.concatenate([self.E, new_e], axis=3)
            self.s = self._steps(self.s, self.r, self.h[-count:], new_e)
        self.time_days = v.time_days
        entry: dict[str, Any] = {
            "event": "observe",
            "at": _now(),
            "time_days": v.time_days,
            "scores": dict(v.scores),
            "pit": {},
        }
        if monitor_outcomes and (self.monitor or self.conformal):
            for name, y in v.scores.items():
                samples = self._predictive_samples(C.ENDPOINT_INDEX[name] - 1, 400)
                pit = float(np.mean(samples < y)) + self.rng.uniform() * float(np.mean(samples == y))
                entry["pit"][name] = pit
                if self.monitor:
                    self.monitor.update(pit, name)
                if self.conformal:
                    entry.setdefault("covered", {})[name] = self.conformal.update(samples, float(y), name)
        node = len(self.h)
        new_records = [(node, C.ENDPOINT_INDEX[name] - 1, float(y), self.last_ledd) for name, y in v.scores.items()]
        lml, steps, resampled, acceptance = self._smc_update(new_records)
        self.records.extend(new_records)
        _, w = self._weights()
        v_w = np.exp(self.logv - logsumexp(self.logv))
        entry.update(
            particle_ess_median=float(np.median(1.0 / np.sum(w**2, axis=1))),
            draw_ess=float(1.0 / np.sum(v_w**2)),
            draws_resampled=resampled,
            tempering_steps=steps,
            pcn_acceptance=acceptance,
            log_marginal_likelihood=lml,
        )
        self.history.append(v)
        self.audit.append(entry)
        return entry

    # ---------------------------------------------------------------------------------------- SMC sampler
    def _rate(self, B: Array) -> Array:
        return np.asarray(self.base[:, None, :] + self.pop.sigma_r[:, None, :] * B, float)

    def _steps(self, s: Array, r: Array, h: Array, e: Array) -> Array:
        sig = self.pop.sigma_xi[:, None, :]
        for t in range(h.size):
            x = expit(s)
            drive = np.logaddexp(0.0, r + np.einsum("kij,kmj->kmi", self.pop.C, x))
            xi = np.exp(-0.5 * h[t] * sig**2 + np.sqrt(h[t]) * sig * e[..., t])
            s = s + h[t] * drive * xi
        return s

    def _loglik(
        self, S0: Array, B: Array, E: Array, records: Sequence[tuple[int, int, float, float]]
    ) -> tuple[Array, Array]:
        """Return the log-likelihood of `records` for every particle, and the latent state at the last step."""
        r = self._rate(B)
        by_step: dict[int, list[tuple[int, float, float]]] = {}
        for step, k, y, ledd in records:
            by_step.setdefault(step, []).append((k, y, ledd))
        ll = np.zeros(S0.shape[:2])

        def score(state: Array, step: int) -> None:
            nonlocal ll
            for k, y, ledd in by_step.get(step, []):
                s_k = state[:, :, k]
                ll = ll + _endpoint_logpdf(
                    y, self._mean(s_k, k, ledd), self._sd(s_k, k), self.lower[k], self.upper[k], self.cfg.censored[k]
                )

        s = S0.copy()
        score(s, 0)
        sig = self.pop.sigma_xi[:, None, :]
        for t in range(self.h.size):
            x = expit(s)
            drive = np.logaddexp(0.0, r + np.einsum("kij,kmj->kmi", self.pop.C, x))
            xi = np.exp(-0.5 * self.h[t] * sig**2 + np.sqrt(self.h[t]) * sig * E[..., t])
            s = s + self.h[t] * drive * xi
            score(s, t + 1)
        return ll, s

    def _smc_update(
        self, new: Sequence[tuple[int, int, float, float]], max_stages: int = 100
    ) -> tuple[float, int, int, float]:
        """Absorb new records by tempered SMC with pCN moves on the non-centred latent vector.

        The target at power phi is prior x likelihood(old records) x likelihood(new records)^phi. The prior of
        (S0, B, E) is standard normal, so the pCN proposal z' = sqrt(1 - beta^2) z + beta xi keeps it invariant and
        the acceptance ratio is the likelihood ratio alone. Each stage raises phi as far as keeps the joint
        effective sample size at half, resamples draws whose own ESS fell below half, and applies `n_moves` pCN
        steps to every particle with a step size adapted towards 25 percent acceptance. Returns the log marginal
        likelihood of the new records, the number of stages, the number of draw-level resamplings and the mean
        acceptance rate.
        """
        old = list(self.records)
        ll_old, _ = self._loglik(self.S0, self.B, self.E, old) if old else (np.zeros(self.logw.shape), None)
        inc, _ = self._loglik(self.S0, self.B, self.E, new)
        total = self.pop.k * self.m
        phi, lml, stages, resampled, accepts = 0.0, 0.0, 0, 0, []
        while phi < 1.0 - 1e-12 and stages < max_stages:
            remaining = 1.0 - phi

            def joint_ess(delta: float, inc: Array = inc) -> float:
                lw = self.logv[:, None] + self.logw + delta * inc
                lw = lw - logsumexp(lw)
                return float(1.0 / np.exp(logsumexp(2 * lw)))

            if joint_ess(remaining) >= total / 2 or stages == max_stages - 1:
                delta = remaining
            else:
                lo, hi = 0.0, remaining
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if joint_ess(mid) >= total / 2 else (lo, mid)
                delta = max(lo, 1e-4 * remaining)
            before = logsumexp(self.logw, axis=1)
            self.logw = self.logw + delta * inc
            after = logsumexp(self.logw, axis=1)
            self.logv = self.logv + (after - before)
            self.logw = self.logw - after[:, None]
            norm = logsumexp(self.logv)
            lml += float(norm)
            self.logv = self.logv - norm
            phi += delta
            stages += 1
            ess = 1.0 / np.exp(logsumexp(2 * self.logw, axis=1))
            for k in (int(i) for i in np.flatnonzero(ess < self.m / 2)):
                idx = _systematic(np.exp(self.logw[k]), self.rng)
                self.S0[k], self.B[k], self.E[k] = self.S0[k, idx], self.B[k, idx], self.E[k, idx]
                ll_old[k], inc[k] = ll_old[k, idx], inc[k, idx]
                self.logw[k] = -np.log(self.m)
                resampled += 1
            for _ in range(self.n_moves):
                beta = self.beta[:, None, None]
                rho = np.sqrt(1 - beta**2)
                S0p = rho * self.S0 + beta * self.rng.normal(size=self.S0.shape)
                Bp = rho * self.B + beta * self.rng.normal(size=self.B.shape)
                Ep = rho[..., None] * self.E + beta[..., None] * self.rng.normal(size=self.E.shape)
                ll_old_p, _ = self._loglik(S0p, Bp, Ep, old) if old else (np.zeros(self.logw.shape), None)
                inc_p, _ = self._loglik(S0p, Bp, Ep, new)
                log_ratio = (ll_old_p + phi * inc_p) - (ll_old + phi * inc)
                accept = np.log(self.rng.uniform(size=log_ratio.shape)) < log_ratio
                self.S0 = np.where(accept[..., None], S0p, self.S0)
                self.B = np.where(accept[..., None], Bp, self.B)
                self.E = np.where(accept[..., None, None], Ep, self.E)
                ll_old = np.where(accept, ll_old_p, ll_old)
                inc = np.where(accept, inc_p, inc)
                rate = accept.mean(axis=1)
                accepts.append(float(rate.mean()))
                self.beta = np.clip(self.beta * np.exp(rate - 0.25), 0.01, 0.99)
        _, self.s = self._loglik(self.S0, self.B, self.E, [])
        self.r = self._rate(self.B)
        return lml, stages, resampled, float(np.mean(accepts)) if accepts else float("nan")

    def _predictive_samples(self, k: int, n: int) -> Array:
        """Draw `n` scores of endpoint index `k` at the twin's current time from its predictive distribution."""
        v, w = self._weights()
        joint = (v[:, None] * w).ravel()
        idx = self.rng.choice(joint.size, size=n, p=joint / joint.sum())
        kk, mm = np.unravel_index(idx, w.shape)
        s_vals = self.s[kk, mm, k]
        mu = _row_mean(self, s_vals, kk, k)
        sd = _row_sd(self, s_vals, kk, k)
        return _endpoint_rng(mu, sd, self.lower[k], self.upper[k], self.cfg.censored[k], self.rng)

    # ---------------------------------------------------------------------------------------- outputs
    def _future(self, days_ahead: Sequence[int], seed: int) -> list[tuple[Array, Array]]:
        """Propagate copies of the cloud to each future offset; return (s, r) per offset, cumulatively."""
        rng = np.random.default_rng(seed)
        s, out, last = self.s.copy(), [], 0
        for d in sorted(days_ahead):
            s = self._propagate(s, self.r, int(d - last), rng)
            out.append((s.copy(), self.r))
            last = d
        return out

    def forecast(
        self,
        horizons_years: Sequence[float],
        *,
        n_samples: int = 1000,
        seed: int = 1,
        ledd_std: float | None = None,
    ) -> Forecast:
        """Forecast every endpoint at each horizon (years after the latest visit).

        The forecast's metadata carries the uncertainty decomposition per target.

        Validation: EXPERIMENTAL.
        """
        if self.time_days is None:
            raise ValueError("the twin has no visit yet")
        days = [int(round(h * C.DAYS_PER_YEAR)) for h in horizons_years]
        order = np.argsort(days)
        clouds = self._future([days[i] for i in order], seed)
        v, w = self._weights()
        joint = (v[:, None] * w).ravel()
        rng = np.random.default_rng(seed + 7)
        idx = rng.choice(joint.size, size=n_samples, p=joint / joint.sum())
        kk, mm = np.unravel_index(idx, w.shape)
        ledd = self.last_ledd if ledd_std is None else float(ledd_std)
        draws = np.zeros((n_samples, 3 * len(days)))
        rows, decomposition = [], []
        for pos, i in enumerate(order):
            s_h, _ = clouds[pos]
            for k in range(3):
                mu_all = self._mean(s_h[:, :, k], k, ledd)
                sd_all = self._sd(s_h[:, :, k], k)
                col = 3 * int(i) + k
                draws[:, col] = _endpoint_rng(
                    mu_all[kk, mm], sd_all[kk, mm], self.lower[k], self.upper[k], self.cfg.censored[k], rng
                )
                within_mean = np.sum(w * mu_all, axis=1)
                within_var = np.sum(w * (mu_all - within_mean[:, None]) ** 2, axis=1)
                grand = float(np.sum(v * within_mean))
                decomposition.append(
                    {
                        "horizon": float(horizons_years[int(i)]),
                        "endpoint": C.ENDPOINTS[k],
                        "population_var": float(np.sum(v * (within_mean - grand) ** 2)),
                        "person_var": float(np.sum(v * within_var)),
                        "measurement_var": float(np.sum(v * np.sum(w * sd_all**2, axis=1))),
                    }
                )
        for h in horizons_years:
            for k in range(3):
                rows.append(
                    {
                        "participant_id": self.participant_id,
                        "endpoint": C.ENDPOINTS[k],
                        "k": k + 1,
                        "horizon": float(h),
                        "t_days": (self.time_days or 0) + int(round(h * C.DAYS_PER_YEAR)),
                    }
                )
        meta = {"method": "living twin", "uncertainty": decomposition, "draw_ess": self.status()["draw_ess"]}
        return Forecast(draws=draws, targets=pd.DataFrame(rows), method="twin", metadata=meta)

    def milestone_curve(
        self, endpoint: str, threshold: float, *, years: float = 5.0, step_years: float = 0.25, seed: int = 2
    ) -> pd.DataFrame:
        """Probability that the expected score of `endpoint` has crossed `threshold` by each time.

        For MoCA, crossing means falling to or below the threshold; for the others, rising to or above it.

        Validation: EXPERIMENTAL.
        """
        k = C.ENDPOINT_INDEX[endpoint] - 1
        grid = np.arange(step_years, years + 1e-9, step_years)
        clouds = self._future([int(round(t * C.DAYS_PER_YEAR)) for t in grid], seed)
        v, w = self._weights()
        joint = v[:, None] * w
        crossed = np.zeros_like(joint, dtype=bool)
        rows = []
        for t, (s_h, _) in zip(grid, clouds, strict=True):
            mu = self._mean(s_h[:, :, k], k, self.last_ledd)
            crossed |= (mu <= threshold) if k == 1 else (mu >= threshold)
            rows.append({"years_ahead": float(t), "probability": float(np.sum(joint * crossed))})
        return pd.DataFrame(rows)

    def next_visit(
        self,
        candidates_years: Sequence[float],
        *,
        target_years: float = 2.0,
        endpoint: str = "updrs3",
        n_outcomes: int = 32,
        seed: int = 3,
    ) -> pd.DataFrame:
        """Rank candidate visit times by expected reduction in forecast variance at `target_years`.

        For each candidate, outcomes are simulated from the twin's predictive distribution, the cloud is
        reweighted by each, and the posterior variance of the expected score at the target is averaged. The
        result reports prior and expected posterior variance, their ratio, and a Gaussian information gain.

        Validation: EXPERIMENTAL.
        """
        k = C.ENDPOINT_INDEX[endpoint] - 1
        v, w = self._weights()
        joint = v[:, None] * w
        target_day = int(round(target_years * C.DAYS_PER_YEAR))
        rng = np.random.default_rng(seed)
        rows = []
        for cand in candidates_years:
            c_day = int(round(cand * C.DAYS_PER_YEAR))
            if c_day <= 0 or c_day >= target_day:
                continue
            s_c = self._propagate(self.s.copy(), self.r, c_day, rng)
            s_t = self._propagate(s_c.copy(), self.r, target_day - c_day, rng)
            mu_t = self._mean(s_t[:, :, k], k, self.last_ledd)
            prior_mean = float(np.sum(joint * mu_t))
            prior_var = float(np.sum(joint * (mu_t - prior_mean) ** 2))
            mu_c = self._mean(s_c[:, :, k], k, self.last_ledd)
            sd_c = self._sd(s_c[:, :, k], k)
            flat = joint.ravel() / joint.sum()
            picks = rng.choice(flat.size, size=n_outcomes, p=flat)
            post_vars = []
            for p in picks:
                kk, mm = np.unravel_index(p, joint.shape)
                y = float(
                    _endpoint_rng(
                        np.array([mu_c[kk, mm]]),
                        np.array([sd_c[kk, mm]]),
                        self.lower[k],
                        self.upper[k],
                        self.cfg.censored[k],
                        rng,
                    )[0]
                )
                ll = _endpoint_logpdf(y, mu_c, sd_c, self.lower[k], self.upper[k], self.cfg.censored[k])
                lw = np.log(joint + 1e-300) + ll
                post = np.exp(lw - logsumexp(lw))
                m_post = float(np.sum(post * mu_t))
                post_vars.append(float(np.sum(post * (mu_t - m_post) ** 2)))
            expected = float(np.mean(post_vars))
            rows.append(
                {
                    "visit_in_years": float(cand),
                    "target_years": target_years,
                    "endpoint": endpoint,
                    "prior_variance": prior_var,
                    "expected_posterior_variance": expected,
                    "variance_reduction": 1 - expected / prior_var if prior_var > 0 else 0.0,
                    "information_gain_nats": 0.5 * np.log(prior_var / expected) if expected > 0 else 0.0,
                }
            )
        table = pd.DataFrame(rows)
        return table.sort_values("variance_reduction", ascending=False).reset_index(drop=True) if len(table) else table

    def status(self) -> dict[str, Any]:
        """Return the twin's state in brief: visits, time, effective sample sizes and monitors.

        Validation: EXPERIMENTAL.
        """
        v, w = self._weights()
        return {
            "participant_id": self.participant_id,
            "visits": len(self.history),
            "time_days": self.time_days,
            "draws": self.pop.k,
            "particles": self.m,
            "draw_ess": float(1.0 / np.sum(v**2)),
            "particle_ess_median": float(np.median(1.0 / np.sum(w**2, axis=1))),
            "monitor": self.monitor.state() if self.monitor else None,
            "conformal": self.conformal.state() if self.conformal else None,
        }

    # ---------------------------------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        """Save the twin: arrays to `.npz` and the audit, history and settings to a sidecar `.json`.

        Validation: EXPERIMENTAL.
        """
        base = Path(path).with_suffix("")
        arrays: dict[str, Any] = {
            "S0": self.S0,
            "B": self.B,
            "E": self.E,
            "h": self.h,
            "logw": self.logw,
            "logv": self.logv,
            "beta": self.beta,
            "base": self.base,
            "d_i": self.d_i,
            "z_i": self.z_i,
        }
        arrays.update({f"pop_{k}": v for k, v in self.pop.to_arrays().items()})
        np.savez_compressed(base.with_suffix(".npz"), **arrays)
        meta = {
            "participant_id": self.participant_id,
            "time_days": self.time_days,
            "last_ledd": self.last_ledd,
            "seed": self.seed,
            "m": self.m,
            "n_moves": self.n_moves,
            "records": self.records,
            "config": self.cfg.model_dump(),
            "source": self.pop.source,
            "history": [v.__dict__ for v in self.history],
            "audit": self.audit,
            "rng_state": self.rng.bit_generator.state,
        }
        base.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        return base.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path) -> ParticipantTwin:
        """Load a twin saved by `save`.

        Validation: EXPERIMENTAL.
        """
        base = Path(path).with_suffix("")
        arr = np.load(base.with_suffix(".npz"))
        meta = json.loads(base.with_suffix(".json").read_text())
        pop = PopulationDraws(**{k[4:]: arr[k] for k in arr.files if k.startswith("pop_")}, source=meta["source"])
        twin = cls.__new__(cls)
        twin.pop, twin.cfg = pop, ModelConfig(**meta["config"])
        twin.participant_id, twin.m, twin.seed = meta["participant_id"], meta["m"], meta["seed"]
        twin.rng = np.random.default_rng()
        twin.rng.bit_generator.state = meta["rng_state"]
        twin.monitor, twin.conformal, twin.n_moves = None, None, int(meta.get("n_moves", 4))
        twin.lower = np.array([C.SCALE_BOUNDS[k][0] for k in C.ENDPOINTS])
        twin.upper = np.array([C.SCALE_BOUNDS[k][1] for k in C.ENDPOINTS])
        twin.S0, twin.B, twin.E, twin.h = arr["S0"], arr["B"], arr["E"], arr["h"]
        twin.logw, twin.logv, twin.beta, twin.base = arr["logw"], arr["logv"], arr["beta"], arr["base"]
        twin.d_i, twin.z_i = arr["d_i"], arr["z_i"]
        twin.records = [(int(a), int(b), float(c), float(d)) for a, b, c, d in meta["records"]]
        twin.r = twin._rate(twin.B)
        _, twin.s = twin._loglik(twin.S0, twin.B, twin.E, [])
        twin.time_days, twin.last_ledd = meta["time_days"], meta["last_ledd"]
        twin.history = [Visit(int(h["time_days"]), dict(h["scores"]), h.get("ledd_std")) for h in meta["history"]]
        twin.audit = meta["audit"]
        twin.audit.append({"event": "loaded", "at": _now()})
        return twin


def _row_mean(twin: ParticipantTwin, s_vals: Array, kk: NDArray[np.int64], k: int) -> Array:
    lin = twin.pop.a[kk, k] + twin.pop.b[kk, k] * s_vals
    if twin.cfg.ledd_in_mu_U and k == 0:
        lin = lin + twin.pop.lambda_ledd[kk] * twin.last_ledd
    link = expit(lin)
    span = twin.upper[k] - twin.lower[k]
    return np.asarray(twin.lower[k] + span * ((1 - link) if k == 1 else link), float)


def _row_sd(twin: ParticipantTwin, s_vals: Array, kk: NDArray[np.int64], k: int) -> Array:
    sd = twin.pop.sigma_obs[kk, k].copy()
    if twin.cfg.state_dependent_sigma:
        sd = sd * np.exp(twin.pop.kappa[kk, k] * (expit(s_vals) - twin.cfg.sigma_xbar[k]))
    return np.asarray(np.maximum(sd, 1e-6), float)


def _visits_from_stan(d: Mapping[str, Any]) -> list[Visit]:
    start, count = int(d["step_start"][0]), int(d["K"][0])
    times = np.concatenate([[0.0], np.cumsum(np.asarray(d["h"][start : start + count], float))])
    by_node: dict[int, dict[str, Any]] = {}
    for node, k, y, ledd in zip(d["obs_node"], d["obs_k"], d["y"], d["obs_ledd"], strict=True):
        rec = by_node.setdefault(int(node), {"scores": {}, "ledd": None})
        rec["scores"][C.ENDPOINTS[int(k) - 1]] = float(y)
        if int(k) == 1:
            rec["ledd"] = float(ledd)
    out = []
    for node in sorted(by_node):
        days = int(round(times[node - 1] * C.DAYS_PER_YEAR))
        out.append(Visit(days, by_node[node]["scores"], by_node[node]["ledd"]))
    return out


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
