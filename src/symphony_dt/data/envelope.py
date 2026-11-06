"""The validity envelope: does a new dataset sit inside the domain pd_v2 was designed for?

The envelope is a design envelope. Its limits come from the model's construction (instrument
ranges, cohort types, the step grid) and from conservative defaults, not from estimates on any
cohort. Each departure is a warning with a code. A warning does not stop a fit; it tells the user
that the model is being used outside the conditions its validation covers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from symphony_dt import constants as C
from symphony_dt.data.dataset import TwinDataset
from symphony_dt.labels import validation

__all__ = ["ENVELOPE_CODES", "PD_V2_ENVELOPE", "EnvelopeReport", "EnvelopeWarning", "check_envelope"]

#: Warning codes and their meaning.
ENVELOPE_CODES: dict[str, str] = {
    "ENV001": "an endpoint the model needs is absent",
    "ENV002": "an instrument version differs from the version the model was built on",
    "ENV003": "a cohort label is outside the cohort types the model was built for",
    "ENV004": "follow-up extends beyond the design horizon",
    "ENV005": "an inter-visit interval is longer than the design maximum",
    "ENV006": "visits closer than one day apart were collapsed (exact ties)",
    "ENV007": "an endpoint is missing at more than the design share of visits",
    "ENV008": "many participants have a single visit and contribute no transition",
    "ENV009": "baseline age lies outside the design range",
    "ENV010": "an exposure (LEDD) is absent, so A02 contributes nothing",
}

#: Design limits of pd_v2. Each is a modelling choice, never an estimate from a cohort.
PD_V2_ENVELOPE: dict[str, Any] = {
    "endpoints": C.ENDPOINTS,
    "instrument_versions": {"updrs3": "MDS-UPDRS 2008", "moca": "8.1", "scopa_aut": "23-item prorated"},
    "cohorts": ("PD", "Prodromal"),
    "max_follow_up_years": 15.0,
    "max_interval_years": 3.0,
    "max_missing_share": 0.8,
    "max_single_visit_share": 0.5,
    "age_range_years": (30.0, 95.0),
}


@validation("TESTED")
@dataclass(frozen=True)
class EnvelopeWarning:
    """One out-of-envelope finding with its code, a message and the share of data affected.

    Validation: TESTED.
    """

    code: str
    message: str
    share: float = 0.0


@validation("TESTED")
@dataclass
class EnvelopeReport:
    """The result of `check_envelope`.

    Validation: TESTED.
    """

    model: str
    warnings: list[EnvelopeWarning] = field(default_factory=list)

    @property
    def inside(self) -> bool:
        """True when no warning was raised."""
        return not self.warnings

    def codes(self) -> list[str]:
        """Return the warning codes in the order they were raised.

        Validation: TESTED.
        """
        return [w.code for w in self.warnings]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary.

        Validation: TESTED.
        """
        return {"model": self.model, "inside": self.inside, "warnings": [w.__dict__ for w in self.warnings]}


@validation("TESTED")
def check_envelope(dataset: TwinDataset, model: str = "pd_v2", limits: dict[str, Any] | None = None) -> EnvelopeReport:
    """Check that `dataset` lies inside the design envelope of `model`.

    `limits` overrides entries of `PD_V2_ENVELOPE` for a stricter or looser check.

    Validation: TESTED.
    """
    if model != "pd_v2":
        raise ValueError(f"no envelope is defined for model {model!r}")
    env = {**PD_V2_ENVELOPE, **(limits or {})}
    report = EnvelopeReport(model=model)

    def warn(code: str, message: str, share: float = 0.0) -> None:
        report.warnings.append(EnvelopeWarning(code, message, round(float(share), 4)))

    present = set(dataset.observations["endpoint"])
    for name in env["endpoints"]:
        if name not in present:
            warn("ENV001", f"endpoint {name} has no observations")
    for name, version in env["instrument_versions"].items():
        if name in dataset.instruments and dataset.instruments.get(name).version != version:
            warn(
                "ENV002",
                f"{name} registered as version {dataset.instruments.get(name).version!r}, "
                f"the model was built on {version!r}",
            )
    if "instrument_version" in dataset.observations.columns:
        for name, version in env["instrument_versions"].items():
            sub = dataset.observations[dataset.observations["endpoint"] == name]
            other = sub["instrument_version"].dropna().astype(str)
            other = other[other != version]
            if len(other):
                warn(
                    "ENV002",
                    f"{len(other)} {name} observations carry another version",
                    len(other) / max(len(sub), 1),
                )

    cohorts = dataset.participants["cohort"].astype(str)
    outside = ~cohorts.isin(env["cohorts"])
    if outside.any():
        labels = sorted(set(cohorts[outside]))
        warn("ENV003", f"cohort labels {labels} are outside {list(env['cohorts'])}", outside.mean())

    visits = dataset.visit_table()
    follow = visits.groupby("participant_id")["time_days"].max() / C.DAYS_PER_YEAR
    beyond = follow > env["max_follow_up_years"]
    if beyond.any():
        warn(
            "ENV004",
            f"{int(beyond.sum())} participants followed beyond {env['max_follow_up_years']} years",
            beyond.mean(),
        )
    gaps = visits.groupby("participant_id")["time_days"].diff().dropna() / C.DAYS_PER_YEAR
    long_gaps = gaps > env["max_interval_years"]
    if long_gaps.any():
        warn(
            "ENV005",
            f"{int(long_gaps.sum())} intervals exceed {env['max_interval_years']} years",
            long_gaps.mean() if len(gaps) else 0.0,
        )
    if dataset.visits is not None:
        raw = dataset.visits.groupby(["participant_id", "time_days"]).size()
        if (raw > 1).any():
            warn("ENV006", f"{int((raw > 1).sum())} visit times appear more than once and were merged")

    wide = dataset.wide_observations()
    for name in env["endpoints"]:
        if name in wide.columns and len(wide):
            share = float(wide[name].isna().mean())
            if share > env["max_missing_share"]:
                warn("ENV007", f"{name} is missing at {share:.0%} of visits", share)
    per = visits.groupby("participant_id").size()
    single = float((per == 1).mean()) if len(per) else 0.0
    if single > env["max_single_visit_share"]:
        warn("ENV008", f"{single:.0%} of participants have one visit", single)

    if "age_baseline_years" in dataset.participants.columns:
        age = pd.to_numeric(dataset.participants["age_baseline_years"], errors="coerce")
        lo, hi = env["age_range_years"]
        off = (age < lo) | (age > hi)
        if off.any():
            warn("ENV009", f"{int(off.sum())} participants aged outside {lo} to {hi} years", off.mean())
    if dataset.visits is None or "ledd" not in dataset.visits.columns:
        warn("ENV010", "no ledd column; the A02 term is constant")
    elif bool(np.all(pd.to_numeric(dataset.visits["ledd"], errors="coerce").fillna(0) == 0)):
        warn("ENV010", "ledd is zero everywhere; the A02 term is constant")
    return report
