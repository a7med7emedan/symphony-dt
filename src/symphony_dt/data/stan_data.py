"""The Stan data builder for pd_v2, ported from the SYMPHONY v2 analysis code.

Four rules are decisions rather than plumbing, and each is kept exactly:

* Every visit is a node, scored or not. A visit with no endpoint adds a transition and no
  likelihood factor, which keeps the missing-at-random treatment exact.
* A participant with one visit has zero transitions; the latent state is the initial state alone.
* Age is centred on the training mean and expressed in decades, not standardised.
* A missing pathway marker is set to zero after standardisation, the training-fold mean.

The step count on each interval uses integer arithmetic on whole days (rule D39), so no float
comparison can move a step boundary between platforms.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from symphony_dt import constants as C
from symphony_dt.config import ModelConfig
from symphony_dt.data.dataset import TwinDataset, participant_sort_key
from symphony_dt.labels import validation

__all__ = [
    "StanData",
    "Standardisation",
    "build_stan_data",
    "dt_years_from_days",
    "h_max_in_days",
    "participant_sort_key",
    "substeps_days",
]

D29_NUISANCE = "genotype_assay_available"


@validation("VALIDATED")
def h_max_in_days(h_max_years: float = C.H_MAX_YEARS) -> tuple[int, int]:
    """Return `h_max` in days as an exact fraction (numerator, denominator).

    At the pinned constants this is (1461, 8), which is 182.625 days.

    Validation: VALIDATED.
    """
    exact = Fraction(h_max_years) * Fraction(C.DAYS_PER_YEAR)
    if exact <= 0:
        raise ValueError(f"h_max must be positive, got {h_max_years}")
    return exact.numerator, exact.denominator


@validation("VALIDATED")
def substeps_days(
    dt_days: NDArray[Any] | Sequence[float], h_max_years: float = C.H_MAX_YEARS
) -> NDArray[np.int64]:
    """Return the number of equal sub-steps on each interval, from integer days (rule D39).

    Only an exact tie collapses: an interval of zero days takes zero steps. Every interval of one
    day or more takes at least one step. The count is ceil(q x days / p) in integer arithmetic.

    Validation: VALIDATED.
    """
    days = np.asarray(dt_days)
    if days.size == 0:
        return np.zeros(0, dtype=np.int64)
    as_float = days.astype(float)
    if not np.all(np.isfinite(as_float)):
        raise ValueError("the grid path was given a non-finite interval")
    if not np.array_equal(as_float, np.round(as_float)):
        offenders = as_float[as_float != np.round(as_float)][:5]
        raise ValueError(f"dt_days must be whole days for the integer grid rule, got {offenders}")
    if np.any(as_float < 0):
        raise ValueError("a negative interval reached the grid path; visits are not ordered")
    p, q = h_max_in_days(h_max_years)
    out = np.empty(days.size, dtype=np.int64)
    for index, value in enumerate(as_float.ravel()):
        whole = int(value)
        out[index] = 0 if whole == 0 else max(-((-q * whole) // p), 1)
    return out


@validation("VALIDATED")
def dt_years_from_days(dt_days: NDArray[Any] | Sequence[float]) -> NDArray[np.float64]:
    """Convert intervals in days to years. Used for the step size only, never for the count.

    Validation: VALIDATED.
    """
    return np.asarray(dt_days, float) / C.DAYS_PER_YEAR


@validation("TESTED")
@dataclass(frozen=True)
class Standardisation:
    """Training-only centring and scaling constants, with a digest of the training participants.

    `stats` maps a name to (mean, sd). Build it with `Standardisation.fit` on the training
    participants only, so no held-out row enters a constant.

    Validation: TESTED.
    """

    stats: dict[str, tuple[float, float]]
    train_digest: str = ""
    n_train: int = 0

    @classmethod
    def fit(
        cls, dataset: TwinDataset, train_ids: Iterable[str] | None = None, *, pathway: Sequence[str] = ()
    ) -> Standardisation:
        """Compute constants on the training participants (all participants when `train_ids` is None).

        Age: mean and sd over training participants. LEDD: mean and sd over every visit of a training
        participant, zero dose where missing. Each pathway marker: mean and sd over training
        participants who have it. Standard deviations use ddof = 1.

        Validation: TESTED.
        """
        ids = participant_sort_key(train_ids if train_ids is not None else dataset.participant_ids)
        keep = set(ids)
        part = dataset.participants[dataset.participants["participant_id"].isin(keep)]
        stats: dict[str, tuple[float, float]] = {}
        if "age_baseline_years" in part.columns:
            age = pd.to_numeric(part["age_baseline_years"], errors="coerce").dropna()
            stats["age_baseline_years"] = (float(age.mean()), float(age.std(ddof=1)))
        visits = dataset.visit_table()
        visits = visits[visits["participant_id"].isin(keep)]
        ledd = (
            pd.to_numeric(visits["ledd"], errors="coerce").fillna(0.0)
            if "ledd" in visits.columns
            else pd.Series(np.zeros(len(visits)))
        )
        stats["ledd"] = (float(ledd.mean()), float(ledd.std(ddof=1)) if len(ledd) > 1 else 0.0)
        if pathway:
            wide = dataset.marker_matrix(pathway).loc[ids]
            for name in pathway:
                column = wide[name].dropna()
                stats[name] = (float(column.mean()), float(column.std(ddof=1)))
        digest = hashlib.sha256(",".join(ids).encode()).hexdigest()
        return cls(stats=stats, train_digest=digest, n_train=len(ids))

    @classmethod
    def from_table(cls, table: pd.DataFrame) -> Standardisation:
        """Build from a frame with columns `name`, `mean` and `sd`.

        Validation: TESTED.
        """
        stats = {str(r["name"]): (float(r["mean"]), float(r["sd"])) for _, r in table.iterrows()}
        return cls(stats=stats)

    def get(self, name: str) -> tuple[float, float]:
        """Return (mean, sd) for `name`, or raise KeyError naming the known constants.

        Validation: TESTED.
        """
        if name not in self.stats:
            raise KeyError(f"no standardisation constant for {name!r}; known: {sorted(self.stats)}")
        return self.stats[name]


@validation("TESTED")
@dataclass
class StanData:
    """The pd_v2 data block with its audit and the participant order behind it.

    Validation: TESTED.
    """

    data: dict[str, Any]
    audit: dict[str, Any] = field(default_factory=dict)
    participant_ids: list[str] = field(default_factory=list)

    def serialise(self) -> bytes:
        """Return the canonical JSON bytes: sorted keys, compact separators, one trailing newline.

        Validation: TESTED.
        """
        return (json.dumps(self.data, separators=(",", ":"), sort_keys=True) + "\n").encode()

    def sha256(self) -> str:
        """Return the sha256 of the canonical JSON bytes.

        Validation: TESTED.
        """
        return hashlib.sha256(self.serialise()).hexdigest()

    def to_json(self, path: str | Path) -> Path:
        """Write the canonical JSON to `path` and return the path.

        Validation: TESTED.
        """
        target = Path(path)
        target.write_bytes(self.serialise())
        return target

    @classmethod
    def from_json(cls, path: str | Path) -> StanData:
        """Read a data block written by `to_json` or by CmdStan tooling.

        Validation: TESTED.
        """
        return cls(data=json.loads(Path(path).read_text()))

    @property
    def n_participants(self) -> int:
        """Number of participants, N."""
        return int(self.data["N"])

    def node_times(self, i: int) -> NDArray[np.float64]:
        """Return node times in years for participant index `i` (zero-based).

        Validation: TESTED.
        """
        start, k = int(self.data["step_start"][i]), int(self.data["K"][i])
        steps = np.asarray(self.data["h"][start : start + k], float)
        return np.concatenate([[0.0], np.cumsum(steps)])


@validation("VALIDATED")
def build_stan_data(
    dataset: TwinDataset,
    *,
    config: ModelConfig | None = None,
    standardisation: Standardisation | None = None,
    pathway: Sequence[str] = (),
    grainsize: int = C.PRODUCTION_GRAINSIZE,
) -> StanData:
    """Build the pd_v2 Stan data block from a TwinDataset.

    `standardisation` must come from training participants only; when None it is fitted on every
    participant, which is correct for a full-cohort fit and wrong for a held-out evaluation.
    `pathway` names marker columns for the pathway block z, in order.
    The output equals the study builder's output on the same input (golden-tested).

    Validation: VALIDATED.
    """
    cfg = config or ModelConfig.primary()
    std = standardisation or Standardisation.fit(dataset, pathway=pathway)
    missing_endpoints = [e for e in C.ENDPOINTS if e not in dataset.instruments]
    if missing_endpoints:
        raise ValueError(f"pd_v2 needs instruments {missing_endpoints} in the registry")

    participants = participant_sort_key(dataset.participant_ids)
    index_of = {p: i for i, p in enumerate(participants)}
    n = len(participants)

    visits = dataset.wide_observations()
    order = {p: i for i, p in enumerate(participants)}
    visits = visits.assign(_order=visits["participant_id"].map(order))
    visits = visits.sort_values(["_order", "time_days"], kind="mergesort").reset_index(drop=True)
    visits["dt_days_prev"] = visits.groupby("participant_id", sort=False)["time_days"].diff()

    is_interval = visits["dt_days_prev"].notna()
    dt_days = visits.loc[is_interval, "dt_days_prev"].to_numpy()
    n_sub = substeps_days(dt_days, cfg.h_max_years)
    dt_years = dt_years_from_days(dt_days)

    K = np.zeros(n, dtype=int)
    step_start = np.zeros(n, dtype=int)
    obs_start = np.zeros(n, dtype=int)
    obs_count = np.zeros(n, dtype=int)
    h: list[float] = []
    obs_node: list[int] = []
    obs_k: list[int] = []
    y: list[float] = []
    obs_ledd: list[float] = []

    ledd_mean, ledd_sd = std.get("ledd")
    ledd_present = "ledd" in visits.columns
    if ledd_present:
        raw_ledd = pd.to_numeric(visits["ledd"], errors="coerce").fillna(0.0).to_numpy(float)
    else:
        raw_ledd = np.zeros(len(visits))
    if ledd_sd == 0.0:
        if cfg.ledd_in_mu_U and ledd_present and np.any(raw_ledd != raw_ledd[0]):
            raise ValueError("ledd has zero training variance but varies in the data")
        ledd_std = np.zeros(len(visits))
    else:
        ledd_std = (raw_ledd - ledd_mean) / ledd_sd

    endpoint_values = visits[list(C.ENDPOINTS)].to_numpy(float)
    pid_column = visits["participant_id"].to_numpy()
    interval_cursor = 0
    position = 0
    for row_index in range(len(visits)):
        pid = pid_column[row_index]
        i = index_of[pid]
        first = row_index == 0 or pid_column[row_index - 1] != pid
        if first:
            if row_index > 0:
                prev = index_of[pid_column[row_index - 1]]
                K[prev] = len(h) - step_start[prev]
                obs_count[prev] = len(y) - obs_start[prev]
            step_start[i] = len(h)
            obs_start[i] = len(y)
            node = 1
            position = 0
        if position > 0:
            count = int(n_sub[interval_cursor])
            if count > 0:
                step = float(dt_years[interval_cursor]) / count
                h.extend([step] * count)
                node += count
            interval_cursor += 1
        for k, name in enumerate(C.ENDPOINTS):
            value = endpoint_values[row_index, k]
            if not np.isnan(value):
                obs_node.append(node)
                obs_k.append(C.ENDPOINT_INDEX[name])
                y.append(float(value))
                obs_ledd.append(float(ledd_std[row_index]) if name == "updrs3" else 0.0)
        position += 1
    if len(visits):
        last = index_of[pid_column[-1]]
        K[last] = len(h) - step_start[last]
        obs_count[last] = len(y) - obs_start[last]

    base = dataset.participants.set_index("participant_id").reindex(participants)
    age_mean, _age_sd = std.get("age_baseline_years")
    nuisance = base[D29_NUISANCE].to_numpy(float) if D29_NUISANCE in base.columns else np.zeros(n)
    dmat = np.vstack(
        [
            (base["age_baseline_years"].to_numpy(float) - age_mean) / C.AGE_DIVISOR_YEARS,
            base["sex"].to_numpy(float),
            nuisance,
        ]
    )

    zrows = []
    z_missing: dict[str, int] = {}
    if pathway:
        markers = dataset.marker_matrix(pathway)
        for name in pathway:
            mean, sd = std.get(name)
            raw = markers[name].to_numpy(float)
            z_missing[name] = int(np.isnan(raw).sum())
            if sd == 0.0:
                raise ValueError(f"{name} has zero training variance, so it cannot be standardised")
            standardised = (raw - mean) / sd
            zrows.append(np.where(np.isnan(standardised), 0.0, standardised))
    zmat = np.vstack(zrows) if zrows else np.zeros((0, n))

    if not np.isfinite(dmat).all():
        raise ValueError("the demographic block contains a non-finite value; age and sex are required")
    if not np.isfinite(zmat).all():
        raise ValueError("the pathway block contains a non-finite value")

    hyper = cfg.hyperparameters()
    lower = [float(C.SCALE_BOUNDS[k][0]) for k in C.ENDPOINTS]
    upper = [float(C.SCALE_BOUNDS[k][1]) for k in C.ENDPOINTS]
    data: dict[str, Any] = {
        "N": int(n),
        "P": len(pathway),
        "n_steps": len(h),
        "n_obs": len(y),
        "K": K.tolist(),
        "step_start": step_start.tolist(),
        "obs_start": obs_start.tolist(),
        "obs_count": obs_count.tolist(),
        "h": [float(v) for v in h],
        "obs_node": obs_node,
        "obs_k": obs_k,
        "y": y,
        "obs_ledd": obs_ledd,
        "dmat": dmat.tolist(),
        "zmat": zmat.tolist(),
        "Lk": lower,
        "Uk": upper,
        "mu_r_prior_mean": hyper["mu_r_mean"],
        "mu_r_prior_sd": hyper["mu_r_sd"],
        "a_prior_mean": list(hyper["a_mean"]),
        "a_prior_sd": hyper["a_sd"],
        "prior_only": 0,
        "grainsize": int(grainsize),
        **cfg.switches(),
    }
    p_num, p_den = h_max_in_days(cfg.h_max_years)
    audit: dict[str, Any] = {
        "h_max_years": cfg.h_max_years,
        "prior": cfg.prior,
        "shape": {
            "participants": n,
            "visits": len(visits),
            "observed_intervals": int(is_interval.sum()),
            "total_steps": len(h),
            "latent_dimension": 6 * n + 3 * len(h),
            "observations_total": len(y),
            "observations_by_endpoint": {
                name: sum(1 for k in obs_k if k == C.ENDPOINT_INDEX[name]) for name in C.ENDPOINTS
            },
        },
        "substep_grid": {
            "rule": "D39. ceil(q x dt_days / p) in integer arithmetic, h_max = p/q days",
            "h_max_days_numerator": p_num,
            "h_max_days_denominator": p_den,
            "intervals_total": len(dt_days),
            "intervals_collapsed_exact_ties": int((n_sub == 0).sum()),
        },
        "design": {
            "demographic_Gamma_d": ["age_baseline_years", "sex", D29_NUISANCE],
            "d29_column_present": D29_NUISANCE in base.columns,
            "pathway_A_z": list(pathway),
            "age_centre_used": age_mean,
            "age_divisor_used": C.AGE_DIVISOR_YEARS,
            "pathway_missing_set_to_zero": z_missing,
        },
        "additions": {
            "A01_state_dependent_sigma": cfg.state_dependent_sigma,
            "A02_ledd_in_mu_U": cfg.ledd_in_mu_U,
            "A03_censored_by_endpoint": dict(zip(C.ENDPOINTS, cfg.censored, strict=True)),
            "ledd_column_present": ledd_present,
            "ledd_centre_used": ledd_mean,
            "ledd_sd_used": ledd_sd,
        },
        "standardisation": {"train_digest": std.train_digest, "n_train": std.n_train},
        "dataset_fingerprint": dataset.fingerprint(),
    }
    return StanData(data=data, audit=audit, participant_ids=participants)
