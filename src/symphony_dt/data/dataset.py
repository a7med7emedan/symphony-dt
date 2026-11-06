"""TwinDataset, the canonical data model of symphony-dt.

A dataset holds four pandas frames: participants, long-format observations, visits and markers.
Time is integer days since each participant's baseline. Validation raises `DataValidationError`
with a rule code, so a failure can be looked up in the documentation.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from symphony_dt.data.instruments import InstrumentRegistry, default_registry
from symphony_dt.errors import DataValidationError
from symphony_dt.labels import validation

__all__ = ["VALIDATION_RULES", "TwinDataset", "participant_sort_key"]

#: Rule codes raised by `TwinDataset.validate`, with their meaning.
VALIDATION_RULES: dict[str, str] = {
    "DV001": "a required column is missing",
    "DV002": "participant_id is missing or duplicated in participants",
    "DV003": "an observation, visit or marker names a participant not in participants",
    "DV004": "time_days is missing, negative or not a whole number of days",
    "DV005": "an observation names an endpoint not in the instrument registry",
    "DV006": "an observation value is missing or not finite",
    "DV007": "an observation value lies outside its instrument range",
    "DV008": "the same participant, time and endpoint appears more than once",
    "DV009": "sex is coded outside 0 (female) and 1 (male)",
    "DV010": "a marker is duplicated for one participant",
    "DV011": "a participant has no visit at time 0 (baseline)",
}

_PARTICIPANT_COLUMNS = ("participant_id", "cohort")
_OBSERVATION_COLUMNS = ("participant_id", "time_days", "endpoint", "value")
_VISIT_COLUMNS = ("participant_id", "time_days")
_MARKER_COLUMNS = ("participant_id", "name", "value")


@validation("TESTED")
def participant_sort_key(ids: Iterable[str]) -> list[str]:
    """Sort identifiers numerically when all are whole numbers, otherwise lexically.

    The study sorted on a numeric identifier. Sorting numeric strings as numbers keeps that order.

    Validation: TESTED.
    """
    values = [str(i) for i in ids]
    if values and all(v.isdigit() for v in values):
        return sorted(values, key=int)
    return sorted(values)


def _require(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise DataValidationError("DV001", f"{name} is missing column(s) {missing}")


def _rows(mask: pd.Series | np.ndarray, frame: pd.DataFrame, limit: int = 5) -> str:
    index = list(frame.index[np.asarray(mask, bool)][:limit])
    return f"rows {index}" + (" and more" if int(np.asarray(mask, bool).sum()) > limit else "")


@validation("TESTED")
class TwinDataset:
    """A longitudinal cohort in the canonical long format.

    `participants` needs `participant_id` and `cohort`; `age_baseline_years`, `sex` and other
    covariates are optional. `observations` needs `participant_id`, `time_days`, `endpoint` and
    `value`. `visits` lists every visit time, including visits without a score, and may carry
    exposures such as `ledd`. `markers` holds one value per participant and marker name.
    If `visits` is omitted, the visit times are the distinct observation times.

    Validation: TESTED.
    """

    def __init__(
        self,
        participants: pd.DataFrame,
        observations: pd.DataFrame,
        visits: pd.DataFrame | None = None,
        markers: pd.DataFrame | None = None,
        instruments: InstrumentRegistry | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        validate: bool = True,
    ) -> None:
        self.instruments = instruments if instruments is not None else default_registry()
        self.participants = participants.copy()
        self.observations = observations.copy()
        self.visits = visits.copy() if visits is not None else None
        self.markers = (
            markers.copy()
            if markers is not None
            else pd.DataFrame(
                {
                    "participant_id": pd.Series(dtype=str),
                    "name": pd.Series(dtype=str),
                    "value": pd.Series(dtype=float),
                }
            )
        )
        self.metadata: dict[str, Any] = dict(metadata or {})
        self._normalise()
        if validate:
            self.validate()

    # -- construction helpers ---------------------------------------------------------------
    def _normalise(self) -> None:
        for frame in (self.participants, self.observations, self.markers):
            if "participant_id" in frame.columns:
                frame["participant_id"] = frame["participant_id"].astype(str)
        if self.visits is not None and "participant_id" in self.visits.columns:
            self.visits["participant_id"] = self.visits["participant_id"].astype(str)
        if "endpoint" in self.observations.columns:
            self.observations["endpoint"] = self.observations["endpoint"].astype(str)

    # -- validation -------------------------------------------------------------------------
    def validate(self) -> TwinDataset:
        """Check every rule in `VALIDATION_RULES` and return self, or raise DataValidationError.

        Validation: TESTED.
        """
        p, o = self.participants, self.observations
        _require(p, _PARTICIPANT_COLUMNS, "participants")
        _require(o, _OBSERVATION_COLUMNS, "observations")
        _require(self.markers, _MARKER_COLUMNS, "markers")

        ids = p["participant_id"]
        bad = ids.isna() | ids.duplicated(keep=False) | (ids.astype(str).str.len() == 0)
        if bad.any():
            raise DataValidationError("DV002", f"participant_id missing or duplicated at {_rows(bad, p)}")
        known = set(ids)

        frames = [("observations", o), ("markers", self.markers)]
        if self.visits is not None:
            _require(self.visits, _VISIT_COLUMNS, "visits")
            frames.append(("visits", self.visits))
        for name, frame in frames:
            unknown = ~frame["participant_id"].isin(known)
            if unknown.any():
                raise DataValidationError(
                    "DV003", f"{name} names participants not in participants at {_rows(unknown, frame)}"
                )

        timed = [("observations", o)]
        if self.visits is not None:
            timed.append(("visits", self.visits))
        for name, frame in timed:
            t = pd.to_numeric(frame["time_days"], errors="coerce").to_numpy(float)
            bad_t = ~np.isfinite(t) | (t < 0) | (t != np.round(t))
            if bad_t.any():
                raise DataValidationError(
                    "DV004", f"{name}.time_days must be whole, non-negative days; see {_rows(bad_t, frame)}"
                )
            frame["time_days"] = t.astype(np.int64)

        unknown_ep = ~o["endpoint"].isin(self.instruments.names())
        if unknown_ep.any():
            names = sorted(set(o.loc[unknown_ep, "endpoint"]))
            raise DataValidationError("DV005", f"endpoints {names} are not in the instrument registry")

        values = pd.to_numeric(o["value"], errors="coerce").to_numpy(float)
        bad_v = ~np.isfinite(values)
        if bad_v.any():
            raise DataValidationError("DV006", f"observation values are missing or non-finite at {_rows(bad_v, o)}")
        o["value"] = values
        lower = o["endpoint"].map({i.name: i.lower for i in self.instruments}).to_numpy(float)
        upper = o["endpoint"].map({i.name: i.upper for i in self.instruments}).to_numpy(float)
        out = (values < lower) | (values > upper)
        if out.any():
            raise DataValidationError("DV007", f"values outside the instrument range at {_rows(out, o)}")

        dup = o.duplicated(["participant_id", "time_days", "endpoint"], keep=False)
        if dup.any():
            raise DataValidationError("DV008", f"duplicate participant, time and endpoint at {_rows(dup, o)}")

        if "sex" in p.columns:
            sex = pd.to_numeric(p["sex"], errors="coerce")
            bad_s = sex.notna() & ~sex.isin([0, 1])
            if bad_s.any():
                raise DataValidationError("DV009", f"sex must be 0 or 1 at {_rows(bad_s, p)}")

        mdup = self.markers.duplicated(["participant_id", "name"], keep=False)
        if mdup.any():
            raise DataValidationError("DV010", f"duplicated markers at {_rows(mdup, self.markers)}")

        visits = self.visit_table()
        first = visits.groupby("participant_id")["time_days"].min()
        no_base = first[first != 0]
        missing_any = known - set(first.index)
        if len(no_base) or missing_any:
            examples = sorted(list(no_base.index[:3]) + sorted(missing_any)[:3])
            raise DataValidationError(
                "DV011", f"participants without a baseline visit at day 0, for example {examples}"
            )
        return self

    # -- views ------------------------------------------------------------------------------
    @property
    def participant_ids(self) -> list[str]:
        """Participant identifiers in sorted order, the order every builder uses."""
        return participant_sort_key(self.participants["participant_id"].astype(str))

    @property
    def endpoints(self) -> list[str]:
        """Endpoints present in the observations, in registry order."""
        present = set(self.observations["endpoint"])
        return [n for n in self.instruments.names() if n in present]

    def visit_table(self) -> pd.DataFrame:
        """Return one row per participant and visit time, sorted, with any visit-level exposures.

        Visits given in `visits` are kept; observation times not in `visits` are added.

        Validation: TESTED.
        """
        obs_times = self.observations[["participant_id", "time_days"]].drop_duplicates()
        if self.visits is None:
            table = obs_times.copy()
        else:
            table = self.visits.drop_duplicates(["participant_id", "time_days"]).merge(
                obs_times, on=["participant_id", "time_days"], how="outer"
            )
        table["time_days"] = table["time_days"].astype(np.int64)
        return table.sort_values(["participant_id", "time_days"], kind="mergesort").reset_index(drop=True)

    def wide_observations(self) -> pd.DataFrame:
        """Return the visit table with one column per endpoint, missing where not scored.

        Validation: TESTED.
        """
        wide = self.observations.pivot_table(
            index=["participant_id", "time_days"], columns="endpoint", values="value", aggfunc="first"
        ).reset_index()
        wide.columns.name = None
        merged = self.visit_table().merge(wide, on=["participant_id", "time_days"], how="left")
        for name in self.instruments.names():
            if name not in merged.columns:
                merged[name] = np.nan
        return merged

    def marker_matrix(self, names: Iterable[str]) -> pd.DataFrame:
        """Return markers as a participant by name frame, missing where absent.

        Validation: TESTED.
        """
        names = list(names)
        wide = self.markers.pivot_table(index="participant_id", columns="name", values="value", aggfunc="first")
        return wide.reindex(index=self.participant_ids, columns=names)

    def subset(self, participant_ids: Iterable[str]) -> TwinDataset:
        """Return a new dataset holding only the named participants.

        Validation: TESTED.
        """
        keep = {str(i) for i in participant_ids}
        visits = None if self.visits is None else self.visits[self.visits["participant_id"].isin(keep)]
        return TwinDataset(
            self.participants[self.participants["participant_id"].isin(keep)],
            self.observations[self.observations["participant_id"].isin(keep)],
            visits,
            self.markers[self.markers["participant_id"].isin(keep)],
            self.instruments,
            self.metadata,
        )

    def summary(self, *, min_cell: int = 10) -> dict[str, Any]:
        """Return aggregate counts only, suppressing any count below `min_cell` participants.

        Validation: TESTED.
        """

        def cell(n: int) -> int | str:
            return n if n >= min_cell else f"<{min_cell}"

        visits = self.visit_table()
        per = visits.groupby("participant_id").size()
        out: dict[str, Any] = {
            "participants": cell(len(self.participants)),
            "observations": len(self.observations),
            "visits": len(visits),
            "median_visits_per_participant": float(per.median()) if len(per) else 0.0,
            "max_follow_up_years": round(float(visits["time_days"].max()) / 365.25, 2) if len(visits) else 0.0,
            "cohorts": {str(k): cell(int(v)) for k, v in self.participants["cohort"].value_counts().items()},
            "observations_by_endpoint": {
                str(k): int(v) for k, v in self.observations["endpoint"].value_counts().items()
            },
        }
        return out

    # -- identity ---------------------------------------------------------------------------
    def fingerprint(self) -> str:
        """Return the sha256 of a canonical serialisation of the dataset. The data never leaves.

        Frames are sorted and written as CSV with fixed float formatting, so the digest depends on
        the content and not on row order or dtype detail.

        Validation: TESTED.
        """
        digest = hashlib.sha256()
        parts: list[tuple[str, pd.DataFrame]] = [
            ("participants", self.participants),
            ("observations", self.observations),
            ("visits", self.visit_table()),
            ("markers", self.markers),
        ]
        for name, frame in parts:
            ordered = frame.reindex(sorted(frame.columns), axis=1)
            ordered = ordered.sort_values(list(ordered.columns), kind="mergesort").reset_index(drop=True)
            buffer = io.StringIO()
            ordered.to_csv(buffer, index=False, float_format="%.12g", lineterminator="\n")
            digest.update(name.encode())
            digest.update(buffer.getvalue().encode())
        for instrument in self.instruments:
            digest.update(instrument.model_dump_json().encode())
        return digest.hexdigest()

    def __repr__(self) -> str:
        return (
            f"TwinDataset(participants={len(self.participants)}, "
            f"observations={len(self.observations)}, endpoints={self.endpoints})"
        )
