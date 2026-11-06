"""A cohort of living twins with one calibration monitor and a relearning loop.

`TwinRegistry.ingest` routes a new visit to the participant's twin. The twin scores the visit against its own
forecast first (one-step-ahead PIT) and then updates. The registry's monitor pools these PITs across the cohort;
its `RelearnPolicy` decides when the population model must be refitted. `relearn` calls a fit function on all
data seen so far, rebuilds every twin on the new population draws by replaying its history, resets the monitor,
and appends the step to the lineage, so each population version is traceable to the data and the reason it was
fitted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from symphony_dt.config import ModelConfig
from symphony_dt.data.dataset import TwinDataset
from symphony_dt.data.stan_data import Standardisation
from symphony_dt.labels import validation
from symphony_dt.twin.lifecycle import ParticipantTwin, PopulationDraws, Visit
from symphony_dt.twin.monitor import AdaptiveConformal, CalibrationMonitor, RelearnDecision, RelearnPolicy

__all__ = ["TwinRegistry"]


@validation("EXPERIMENTAL")
@dataclass
class TwinRegistry:
    """Many participant twins over one population, with pooled monitoring and versioned relearning.

    Validation: EXPERIMENTAL.
    """

    population: PopulationDraws
    standardisation: Standardisation
    config: ModelConfig = field(default_factory=ModelConfig.primary)
    monitor: CalibrationMonitor = field(default_factory=CalibrationMonitor)
    conformal: AdaptiveConformal = field(default_factory=AdaptiveConformal)
    policy: RelearnPolicy = field(default_factory=RelearnPolicy)
    n_particles: int = 1000
    twins: dict[str, ParticipantTwin] = field(default_factory=dict)
    new_outcomes: int = 0
    lineage: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.lineage:
            self.lineage.append(
                {
                    "version": 1,
                    "source": self.population.source,
                    "reason": "initial population",
                    "at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            )

    @property
    def version(self) -> int:
        """The current population version."""
        return int(self.lineage[-1]["version"])

    def enrol(self, history: TwinDataset, *, seed: int = 0) -> ParticipantTwin:
        """Create and register a twin from one participant's history.

        Validation: EXPERIMENTAL.
        """
        twin = ParticipantTwin.create(
            self.population,
            history,
            standardisation=self.standardisation,
            config=self.config,
            n_particles=self.n_particles,
            seed=seed,
            monitor=self.monitor,
            conformal=self.conformal,
        )
        self.twins[twin.participant_id] = twin
        return twin

    def enrol_all(self, dataset: TwinDataset, *, seed: int = 0) -> int:
        """Enrol every participant of a dataset; return how many were enrolled.

        Validation: EXPERIMENTAL.
        """
        for n, pid in enumerate(dataset.participant_ids):
            self.enrol(dataset.subset([pid]), seed=seed + n)
        return len(dataset.participant_ids)

    def ingest(self, participant_id: str, visit: Visit | Mapping[str, Any]) -> dict[str, Any]:
        """Score a new visit against the twin's forecast, update the twin, and return the audit entry.

        Validation: EXPERIMENTAL.
        """
        if participant_id not in self.twins:
            raise KeyError(f"no twin for {participant_id!r}; enrol it first")
        entry = self.twins[participant_id].observe(visit)
        self.new_outcomes += len(entry["scores"])
        entry["population_version"] = self.version
        return entry

    def decide(self) -> RelearnDecision:
        """Ask the relearning policy whether to refit now.

        Validation: EXPERIMENTAL.
        """
        return self.policy.decide(self.monitor, self.new_outcomes)

    def relearn(
        self, fit_function: Callable[[TwinDataset], PopulationDraws], data: TwinDataset, *, reason: str = ""
    ) -> int:
        """Refit the population on `data`, rebuild every twin on the new draws, and return the new version.

        `fit_function` maps a dataset to new population draws, for example by running `symphony_dt.fit.fit`
        and `PopulationDraws.from_fit`. Every twin replays its own history on the new population.

        Validation: EXPERIMENTAL.
        """
        decision = self.decide()
        self.population = fit_function(data)
        self.standardisation = Standardisation.fit(data)
        histories = {pid: twin.history for pid, twin in self.twins.items()}
        self.twins = {}
        for n, (pid, visits) in enumerate(histories.items()):
            twin = ParticipantTwin.create(
                self.population,
                data.subset([pid]),
                standardisation=self.standardisation,
                config=self.config,
                n_particles=self.n_particles,
                seed=n,
            )
            twin.monitor, twin.conformal = self.monitor, self.conformal
            for v in visits[len(twin.history) :]:
                twin.observe(v, monitor_outcomes=False)
            self.twins[pid] = twin
        self.monitor.reset()
        self.lineage.append(
            {
                "version": self.version + 1,
                "source": self.population.source,
                "reason": reason or decision.reason,
                "evidence": decision.evidence,
                "new_outcomes": self.new_outcomes,
                "participants": len(self.twins),
                "dataset_fingerprint": data.fingerprint(),
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        self.new_outcomes = 0
        return self.version

    def status(self) -> pd.DataFrame:
        """Return one row per twin with its visits, time and effective sample sizes.

        Validation: EXPERIMENTAL.
        """
        rows = []
        for twin in self.twins.values():
            s = twin.status()
            rows.append({k: s[k] for k in ("participant_id", "visits", "time_days", "draw_ess", "particle_ess_median")})
        return pd.DataFrame(rows)
