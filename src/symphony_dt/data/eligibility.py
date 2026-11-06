"""Eligibility: declarative inclusion rules applied in sequence, with an attrition table.

Each rule decides, per participant, whether the participant meets it. Rules apply in the order given.
The attrition table follows the OHDSI convention: a base count, then the count remaining after each
rule, the number that rule removed in sequence, and its gain, the number it alone removes from the
final cohort. Per-participant decisions stay with the result object; shared tables suppress small cells.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from symphony_dt import constants as C
from symphony_dt.data.dataset import TwinDataset
from symphony_dt.labels import validation

__all__ = [
    "AgeRange",
    "CohortIn",
    "CustomRule",
    "EligibilityCriteria",
    "EligibilityResult",
    "EligibilityRule",
    "MaxGap",
    "MinEndpointObservations",
    "MinFollowUp",
    "MinVisits",
    "RequiredAtBaseline",
    "apply_eligibility",
    "rule_from_dict",
]


@validation("TESTED")
@dataclass(frozen=True)
class EligibilityRule:
    """Base class: a named rule that returns one boolean per participant.

    Validation: TESTED.
    """

    name: str = field(default="rule", kw_only=True)

    def evaluate(self, dataset: TwinDataset) -> pd.Series:
        """Return a boolean Series indexed by participant_id.

        Validation: TESTED.
        """
        raise NotImplementedError

    def describe(self) -> str:
        """Return a one-line description for tables and flow charts.

        Validation: TESTED.
        """
        return self.name


def _visits(dataset: TwinDataset) -> pd.DataFrame:
    return dataset.visit_table()


def _index(dataset: TwinDataset) -> pd.Index:
    return pd.Index(dataset.participant_ids, name="participant_id")


@validation("TESTED")
@dataclass(frozen=True)
class MinVisits(EligibilityRule):
    """At least `n` distinct visit times. With `endpoint`, only visits that score it count.

    Validation: TESTED.
    """

    n: int = 2
    endpoint: str | None = None
    name: str = field(default="min_visits", kw_only=True)

    def evaluate(self, dataset: TwinDataset) -> pd.Series:
        if self.endpoint is None:
            counts = _visits(dataset).groupby("participant_id")["time_days"].nunique()
        else:
            obs = dataset.observations
            counts = obs[obs["endpoint"] == self.endpoint].groupby("participant_id")["time_days"].nunique()
        return (counts.reindex(_index(dataset)).fillna(0) >= self.n).rename(self.name)

    def describe(self) -> str:
        what = f" scoring {self.endpoint}" if self.endpoint else ""
        return f"at least {self.n} visits{what}"


@validation("TESTED")
@dataclass(frozen=True)
class MinFollowUp(EligibilityRule):
    """Last visit at least `years` after baseline.

    Validation: TESTED.
    """

    years: float = 1.0
    name: str = field(default="min_follow_up", kw_only=True)

    def evaluate(self, dataset: TwinDataset) -> pd.Series:
        last = _visits(dataset).groupby("participant_id")["time_days"].max() / C.DAYS_PER_YEAR
        return (last.reindex(_index(dataset)).fillna(0) >= self.years).rename(self.name)

    def describe(self) -> str:
        return f"follow-up of at least {self.years:g} years"


@validation("TESTED")
@dataclass(frozen=True)
class RequiredAtBaseline(EligibilityRule):
    """Every listed endpoint scored at the baseline visit (day 0).

    Validation: TESTED.
    """

    endpoints: tuple[str, ...] = C.ENDPOINTS
    name: str = field(default="required_at_baseline", kw_only=True)

    def evaluate(self, dataset: TwinDataset) -> pd.Series:
        obs = dataset.observations
        base = obs[obs["time_days"] == 0]
        have = base.groupby("participant_id")["endpoint"].agg(set)
        need = set(self.endpoints)
        ok = pd.Series([need <= s for s in have], index=have.index, dtype=bool)
        return ok.reindex(_index(dataset), fill_value=False).rename(self.name)

    def describe(self) -> str:
        return f"{', '.join(self.endpoints)} scored at baseline"


@validation("TESTED")
@dataclass(frozen=True)
class CohortIn(EligibilityRule):
    """Cohort label in the allowed set.

    Validation: TESTED.
    """

    cohorts: tuple[str, ...] = ("PD", "Prodromal")
    name: str = field(default="cohort_in", kw_only=True)

    def evaluate(self, dataset: TwinDataset) -> pd.Series:
        p = dataset.participants.set_index("participant_id")["cohort"].astype(str)
        return p.isin(self.cohorts).reindex(_index(dataset)).fillna(False).astype(bool).rename(self.name)

    def describe(self) -> str:
        return f"cohort in {', '.join(self.cohorts)}"


@validation("TESTED")
@dataclass(frozen=True)
class AgeRange(EligibilityRule):
    """Baseline age within [low, high] years, ends included. Missing age fails.

    Validation: TESTED.
    """

    low: float = 30.0
    high: float = 95.0
    name: str = field(default="age_range", kw_only=True)

    def evaluate(self, dataset: TwinDataset) -> pd.Series:
        p = dataset.participants.set_index("participant_id")
        if "age_baseline_years" not in p.columns:
            return pd.Series(False, index=_index(dataset), name=self.name)
        age = pd.to_numeric(p["age_baseline_years"], errors="coerce")
        return age.between(self.low, self.high).reindex(_index(dataset)).fillna(False).astype(bool).rename(self.name)

    def describe(self) -> str:
        return f"baseline age {self.low:g} to {self.high:g} years"


@validation("TESTED")
@dataclass(frozen=True)
class MaxGap(EligibilityRule):
    """No interval between consecutive visits longer than `years`.

    Validation: TESTED.
    """

    years: float = 3.0
    name: str = field(default="max_gap", kw_only=True)

    def evaluate(self, dataset: TwinDataset) -> pd.Series:
        v = _visits(dataset)
        gaps = v.groupby("participant_id")["time_days"].diff() / C.DAYS_PER_YEAR
        worst = gaps.groupby(v["participant_id"]).max().fillna(0)
        return (worst.reindex(_index(dataset)).fillna(0) <= self.years).rename(self.name)

    def describe(self) -> str:
        return f"no gap between visits over {self.years:g} years"


@validation("TESTED")
@dataclass(frozen=True)
class MinEndpointObservations(EligibilityRule):
    """At least `n` observations of `endpoint`.

    Validation: TESTED.
    """

    endpoint: str = "updrs3"
    n: int = 1
    name: str = field(default="min_endpoint_observations", kw_only=True)

    def evaluate(self, dataset: TwinDataset) -> pd.Series:
        obs = dataset.observations
        counts = obs[obs["endpoint"] == self.endpoint].groupby("participant_id").size()
        return (counts.reindex(_index(dataset)).fillna(0) >= self.n).rename(self.name)

    def describe(self) -> str:
        return f"at least {self.n} {self.endpoint} observations"


@validation("TESTED")
@dataclass(frozen=True)
class CustomRule(EligibilityRule):
    """A user rule: a function from the dataset to a boolean Series by participant_id.

    Validation: TESTED.
    """

    function: Callable[[TwinDataset], pd.Series] | None = None
    text: str = "custom rule"
    name: str = field(default="custom", kw_only=True)

    def evaluate(self, dataset: TwinDataset) -> pd.Series:
        if self.function is None:
            raise ValueError("CustomRule needs a function")
        out = self.function(dataset)
        return pd.Series(out).reindex(_index(dataset)).fillna(False).astype(bool).rename(self.name)

    def describe(self) -> str:
        return self.text


_RULES: dict[str, type[EligibilityRule]] = {
    "min_visits": MinVisits,
    "min_follow_up": MinFollowUp,
    "required_at_baseline": RequiredAtBaseline,
    "cohort_in": CohortIn,
    "age_range": AgeRange,
    "max_gap": MaxGap,
    "min_endpoint_observations": MinEndpointObservations,
}


@validation("TESTED")
def rule_from_dict(spec: Mapping[str, Any]) -> EligibilityRule:
    """Build a rule from a dict such as `{"rule": "min_visits", "n": 3}`, as written in a study file.

    Validation: TESTED.
    """
    data = dict(spec)
    kind = data.pop("rule", None)
    if kind not in _RULES:
        raise ValueError(f"unknown eligibility rule {kind!r}; known: {sorted(_RULES)}")
    for key in ("endpoints", "cohorts"):
        if key in data:
            data[key] = tuple(data[key])
    return _RULES[kind](**data)


@validation("TESTED")
@dataclass
class EligibilityCriteria:
    """An ordered list of rules.

    Validation: TESTED.
    """

    rules: list[EligibilityRule] = field(default_factory=list)

    @classmethod
    def from_list(cls, specs: Iterable[Mapping[str, Any] | EligibilityRule]) -> EligibilityCriteria:
        """Build criteria from rule objects or study-file dicts.

        Validation: TESTED.
        """
        return cls([s if isinstance(s, EligibilityRule) else rule_from_dict(s) for s in specs])

    @classmethod
    def pd_v2_default(cls) -> EligibilityCriteria:
        """A sensible default for pd_v2: PD or prodromal, age 30 to 95, a baseline score of every endpoint.

        These are design defaults, not the study's eligibility ledger.

        Validation: TESTED.
        """
        return cls([CohortIn(), AgeRange(), RequiredAtBaseline()])


@validation("TESTED")
@dataclass
class EligibilityResult:
    """The outcome of `apply_eligibility`.

    `decisions` is participant-level (one column per rule) and must stay where the data are governed.
    `attrition` is aggregate.

    Validation: TESTED.
    """

    eligible: TwinDataset
    decisions: pd.DataFrame
    attrition: pd.DataFrame
    rules: list[str]

    @property
    def n_eligible(self) -> int:
        """Number of eligible participants."""
        return int(self.decisions["eligible"].sum())

    def excluded_ids(self) -> list[str]:
        """Return identifiers of excluded participants, for local use only.

        Validation: TESTED.
        """
        return list(self.decisions.index[~self.decisions["eligible"]])

    def attrition_table(self, *, min_cell: int = 10) -> pd.DataFrame:
        """Return the attrition table with counts below `min_cell` shown as `<min_cell`.

        Validation: TESTED.
        """
        table = self.attrition.copy()
        for col in ("remaining", "removed", "gain"):
            table[col] = [v if (v == 0 or v >= min_cell) else f"<{min_cell}" for v in table[col]]
        return table

    def to_markdown(self, *, min_cell: int = 10) -> str:
        """Return the attrition table as Markdown.

        Validation: TESTED.
        """
        t = self.attrition_table(min_cell=min_cell)
        lines = ["| step | rule | remaining | removed | gain |", "|---|---|---|---|---|"]
        lines += [f"| {r.step} | {r.rule} | {r.remaining} | {r.removed} | {r.gain} |" for r in t.itertuples()]
        return "\n".join(lines) + "\n"


@validation("TESTED")
def apply_eligibility(
    dataset: TwinDataset, criteria: EligibilityCriteria | Sequence[EligibilityRule | Mapping[str, Any]]
) -> EligibilityResult:
    """Apply rules in sequence and return the eligible dataset, decisions and attrition.

    Validation: TESTED.
    """
    crit = criteria if isinstance(criteria, EligibilityCriteria) else EligibilityCriteria.from_list(criteria)
    idx = _index(dataset)
    decisions = pd.DataFrame(index=idx)
    for rule in crit.rules:
        col = rule.name
        while col in decisions.columns:
            col += "_"
        decisions[col] = rule.evaluate(dataset).reindex(idx).fillna(False).astype(bool).to_numpy()
    cols = list(decisions.columns)
    final = decisions[cols].all(axis=1) if cols else pd.Series(True, index=idx)
    decisions["eligible"] = final.to_numpy()
    rows: list[dict[str, Any]] = [
        {"step": 0, "rule": "all participants", "remaining": len(idx), "removed": 0, "gain": 0}
    ]
    alive = np.ones(len(idx), bool)
    for step, (rule, col) in enumerate(zip(crit.rules, cols, strict=True), start=1):
        passed = decisions[col].to_numpy()
        removed = int((alive & ~passed).sum())
        alive &= passed
        others = [c for c in cols if c != col]
        without = decisions[others].all(axis=1).to_numpy() if others else np.ones(len(idx), bool)
        gain = int(without.sum() - final.sum())
        rows.append(
            {
                "step": step,
                "rule": rule.describe(),
                "remaining": int(alive.sum()),
                "removed": removed,
                "gain": gain,
            }
        )
    eligible = dataset.subset(list(idx[final.to_numpy()]))
    eligible.metadata = {**dataset.metadata, "eligibility": [r.describe() for r in crit.rules]}
    return EligibilityResult(
        eligible=eligible,
        decisions=decisions,
        attrition=pd.DataFrame(rows),
        rules=[r.describe() for r in crit.rules],
    )
