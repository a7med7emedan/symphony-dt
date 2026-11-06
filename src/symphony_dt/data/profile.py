"""A data profile: what a modeller checks before fitting, in one object.

Per endpoint: observations, participants, the share of visits that score it, the share at the floor and at the
ceiling, and the observed range. Per cohort: participants, visits and follow-up. Over the cohort: visits per
participant, intervals between visits and single-visit share. Aggregates only, with small cells suppressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from symphony_dt import constants as C
from symphony_dt.data.dataset import TwinDataset
from symphony_dt.labels import validation

__all__ = ["DataProfile", "profile"]


def _q(series: pd.Series, qs: tuple[float, ...] = (0.25, 0.5, 0.75)) -> list[float]:
    if series.empty:
        return [float("nan")] * len(qs)
    return [round(float(v), 2) for v in series.quantile(list(qs))]


@validation("TESTED")
@dataclass
class DataProfile:
    """Aggregate description of a dataset.

    Validation: TESTED.
    """

    overview: dict[str, Any]
    endpoints: pd.DataFrame
    cohorts: pd.DataFrame
    visits_per_participant: pd.Series = field(repr=False)
    intervals_years: pd.Series = field(repr=False)
    min_cell: int = 10

    def to_markdown(self) -> str:
        """Return the profile as Markdown tables.

        Validation: TESTED.
        """
        lines = ["## Overview", "", "| quantity | value |", "|---|---|"]
        lines += [f"| {k.replace('_', ' ')} | {v} |" for k, v in self.overview.items()]
        for title, frame in (("Endpoints", self.endpoints), ("Cohorts", self.cohorts)):
            lines += [
                "",
                f"## {title}",
                "",
                "| " + " | ".join(frame.columns) + " |",
                "|" + "---|" * len(frame.columns),
            ]
            lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
        return "\n".join(lines) + "\n"

    def plot(self) -> Any:
        """Draw a four-panel profile figure and return it.

        Validation: TESTED.
        """
        from symphony_dt.plots import plot_profile

        return plot_profile(self)


@validation("TESTED")
def profile(dataset: TwinDataset, *, min_cell: int = 10) -> DataProfile:
    """Profile a dataset for modelling. Counts of participants below `min_cell` are shown as `<min_cell`.

    Validation: TESTED.
    """

    def cell(n: int) -> int | str:
        return n if (n == 0 or n >= min_cell) else f"<{min_cell}"

    visits = dataset.visit_table()
    per = visits.groupby("participant_id").size()
    gaps = visits.groupby("participant_id")["time_days"].diff().dropna() / C.DAYS_PER_YEAR
    follow = visits.groupby("participant_id")["time_days"].max() / C.DAYS_PER_YEAR
    wide = dataset.wide_observations()
    rows = []
    for inst in dataset.instruments:
        obs = dataset.observations[dataset.observations["endpoint"] == inst.name]["value"]
        n_part = int(
            dataset.observations.loc[dataset.observations["endpoint"] == inst.name, "participant_id"].nunique()
        )
        coverage = float(wide[inst.name].notna().mean()) if inst.name in wide and len(wide) else 0.0
        rows.append(
            {
                "endpoint": inst.name,
                "observations": len(obs),
                "participants": cell(n_part),
                "share_of_visits_scored": round(coverage, 3),
                "share_at_floor": round(float((obs <= inst.lower).mean()), 3) if len(obs) else float("nan"),
                "share_at_ceiling": round(float((obs >= inst.upper).mean()), 3) if len(obs) else float("nan"),
                "observed_min": round(float(obs.min()), 2) if len(obs) else float("nan"),
                "observed_max": round(float(obs.max()), 2) if len(obs) else float("nan"),
                "scale": f"{inst.lower:g} to {inst.upper:g}",
            }
        )
    cohort_rows = []
    p = dataset.participants.set_index("participant_id")
    for name, members in p.groupby("cohort").groups.items():
        ids = set(members)
        sub = per[per.index.isin(ids)]
        fu = follow[follow.index.isin(ids)]
        cohort_rows.append(
            {
                "cohort": str(name),
                "participants": cell(len(ids)),
                "median_visits": float(sub.median()) if len(sub) >= min_cell else "suppressed",
                "median_follow_up_years": round(float(fu.median()), 2) if len(fu) >= min_cell else "suppressed",
            }
        )
    overview = {
        "participants": cell(len(p)),
        "visits": len(visits),
        "observations": len(dataset.observations),
        "visits_per_participant_q25_q50_q75": _q(per.astype(float)),
        "interval_years_q25_q50_q75": _q(gaps),
        "follow_up_years_q25_q50_q75": _q(follow),
        "single_visit_share": round(float((per == 1).mean()), 3) if len(per) else 0.0,
        "endpoints_present": dataset.endpoints,
        "has_ledd": bool(dataset.visits is not None and "ledd" in dataset.visits.columns),
        "markers": sorted(set(dataset.markers["name"])) if len(dataset.markers) else [],
    }
    return DataProfile(
        overview=overview,
        endpoints=pd.DataFrame(rows),
        cohorts=pd.DataFrame(cohort_rows),
        visits_per_participant=per.astype(int).reset_index(drop=True),
        intervals_years=gaps.reset_index(drop=True),
        min_cell=min_cell,
    )
