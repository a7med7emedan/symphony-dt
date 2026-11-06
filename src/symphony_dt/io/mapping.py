"""The YAML mapping engine behind every adapter.

A mapping says which table holds participants, which tables hold scores, how time is measured and how
codes translate to endpoints. It never holds data. Shipped mappings live in `symphony_dt/io/mappings/`
and are documented on the Interoperability page; a user's own mapping is a YAML file with the same keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from symphony_dt import constants as C
from symphony_dt.data.dataset import TwinDataset
from symphony_dt.errors import MappingError
from symphony_dt.labels import validation

__all__ = [
    "CohortMap",
    "DatasetMapping",
    "ObservationSource",
    "apply_mapping",
    "list_mappings",
    "load_mapping",
]

_UNIT_DAYS = {"days": 1.0, "weeks": 7.0, "months": C.DAYS_PER_YEAR / 12, "years": C.DAYS_PER_YEAR}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


@validation("TESTED")
class TimeSpec(_Strict):
    """How a source measures time: a numeric column in a unit, or a date column.

    Validation: TESTED.
    """

    column: str
    kind: Literal["numeric", "date"] = "numeric"
    unit: Literal["days", "weeks", "months", "years"] = "days"
    date_format: str | None = None


@validation("TESTED")
class CohortMap(_Strict):
    """A column whose codes map to cohort labels, or one fixed label.

    Validation: TESTED.
    """

    column: str | None = None
    values: dict[str, str] = {}
    default: str | None = None


@validation("TESTED")
class ParticipantSource(_Strict):
    """Where participants and their covariates live.

    Validation: TESTED.
    """

    table: str
    id: str
    cohort: CohortMap = CohortMap(default="unknown")
    age: str | None = None
    sex: str | None = None
    sex_values: dict[str, int] = {}
    covariates: dict[str, str] = {}
    filter: dict[str, Any] = {}


@validation("TESTED")
class ObservationSource(_Strict):
    """One table of scores: a fixed endpoint or an endpoint code column, a value or items to sum.

    Validation: TESTED.
    """

    table: str
    id: str
    time: TimeSpec
    endpoint: str | None = None
    endpoint_column: str | None = None
    endpoint_values: dict[str, str] = {}
    value: str | None = None
    sum_of: list[str] = []
    filter: dict[str, Any] = {}


@validation("TESTED")
class VisitSource(_Strict):
    """A table of visit-level exposures such as LEDD.

    Validation: TESTED.
    """

    table: str
    id: str
    time: TimeSpec
    exposures: dict[str, str] = {}
    filter: dict[str, Any] = {}


@validation("TESTED")
class MarkerSource(_Strict):
    """A wide table of participant-level markers.

    Validation: TESTED.
    """

    table: str
    id: str
    columns: dict[str, str] = {}


@validation("TESTED")
class DatasetMapping(_Strict):
    """A full mapping from source tables to a TwinDataset.

    Validation: TESTED.
    """

    name: str
    description: str = ""
    files: dict[str, str] = {}
    participants: ParticipantSource
    observations: list[ObservationSource]
    visits: list[VisitSource] = []
    markers: MarkerSource | None = None
    origin: Literal["first_visit", "zero"] = "first_visit"


def _mapping_dir() -> Any:
    return resources.files("symphony_dt.io").joinpath("mappings")


@validation("TESTED")
def list_mappings() -> list[str]:
    """Return the names of the mappings shipped with the package.

    Validation: TESTED.
    """
    return sorted(p.name.removesuffix(".yaml") for p in _mapping_dir().iterdir() if p.name.endswith(".yaml"))


@validation("TESTED")
def load_mapping(name_or_path: str | Path | Mapping[str, Any]) -> DatasetMapping:
    """Load a shipped mapping by name, a YAML file by path, or a dict.

    Raises MappingError with the validation detail when the mapping is malformed.

    Validation: TESTED.
    """
    if isinstance(name_or_path, Mapping):
        raw: Any = dict(name_or_path)
    else:
        text: str
        candidate = Path(name_or_path)
        if candidate.suffix in (".yaml", ".yml") and candidate.exists():
            text = candidate.read_text()
        else:
            shipped = _mapping_dir().joinpath(f"{name_or_path}.yaml")
            if not shipped.is_file():
                raise MappingError(f"no mapping {name_or_path!r}; shipped: {list_mappings()}")
            text = shipped.read_text()
        raw = yaml.safe_load(text)
    try:
        return DatasetMapping(**raw)
    except ValidationError as err:
        raise MappingError(f"malformed mapping: {err}") from err


def _filtered(frame: pd.DataFrame, rules: Mapping[str, Any]) -> pd.DataFrame:
    for column, wanted in rules.items():
        if column not in frame.columns:
            raise MappingError(f"filter column {column!r} is not in the table")
        allowed = wanted if isinstance(wanted, list) else [wanted]
        frame = frame[frame[column].astype(str).isin([str(v) for v in allowed])]
    return frame


def _need(frame: pd.DataFrame, columns: list[str], table: str) -> None:
    missing = [c for c in columns if c and c not in frame.columns]
    if missing:
        raise MappingError(f"table {table!r} lacks column(s) {missing}")


def _time_values(frame: pd.DataFrame, spec: TimeSpec) -> pd.Series:
    if spec.kind == "date":
        dates = pd.to_datetime(frame[spec.column], format=spec.date_format, errors="coerce")
        return pd.Series((dates - pd.Timestamp("1970-01-01")).dt.days, index=frame.index, dtype=float)
    return pd.to_numeric(frame[spec.column], errors="coerce") * _UNIT_DAYS[spec.unit]


@validation("TESTED")
def apply_mapping(
    tables: Mapping[str, pd.DataFrame],
    mapping: DatasetMapping | str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> TwinDataset:
    """Build a TwinDataset from named source tables with a mapping.

    Time is re-expressed as whole days since each participant's first record across all sources
    (`origin: first_visit`), or kept as given (`origin: zero`). Rows with a missing value or time are
    dropped and counted in `metadata["mapping_dropped"]`.

    Validation: TESTED.
    """
    m = mapping if isinstance(mapping, DatasetMapping) else load_mapping(mapping)
    for name in {m.participants.table, *(s.table for s in m.observations), *(v.table for v in m.visits)}:
        if name not in tables:
            raise MappingError(f"the mapping needs table {name!r}; given {sorted(tables)}")

    ps = m.participants
    ptab = _filtered(tables[ps.table].copy(), ps.filter)
    _need(
        ptab, [ps.id, ps.age or "", ps.sex or "", ps.cohort.column or "", *ps.covariates.values()], ps.table
    )
    part = pd.DataFrame({"participant_id": ptab[ps.id].astype(str)})
    if ps.cohort.column:
        codes = ptab[ps.cohort.column].astype(str)
        part["cohort"] = codes.map(ps.cohort.values).fillna(ps.cohort.default or "unknown").to_numpy()
        if not ps.cohort.values:
            part["cohort"] = codes.to_numpy()
    else:
        part["cohort"] = ps.cohort.default or "unknown"
    if ps.age:
        part["age_baseline_years"] = pd.to_numeric(ptab[ps.age], errors="coerce").to_numpy()
    if ps.sex:
        raw = ptab[ps.sex].astype(str)
        part["sex"] = (
            raw.map({str(k): v for k, v in ps.sex_values.items()})
            if ps.sex_values
            else pd.to_numeric(raw, errors="coerce")
        ).to_numpy()
    for target, column in ps.covariates.items():
        part[target] = ptab[column].to_numpy()
    part = part.drop_duplicates("participant_id")

    dropped: dict[str, int] = {}
    obs_frames = []
    for src in m.observations:
        tab = _filtered(tables[src.table].copy(), src.filter)
        _need(
            tab, [src.id, src.time.column, src.value or "", src.endpoint_column or "", *src.sum_of], src.table
        )
        if src.sum_of:
            items = tab[src.sum_of].apply(pd.to_numeric, errors="coerce")
            value = items.sum(axis=1, min_count=len(src.sum_of))
        elif src.value:
            value = pd.to_numeric(tab[src.value], errors="coerce")
        else:
            raise MappingError(f"source {src.table!r} needs value or sum_of")
        if src.endpoint:
            endpoint = pd.Series(src.endpoint, index=tab.index)
        elif src.endpoint_column:
            endpoint = tab[src.endpoint_column].astype(str).map(src.endpoint_values)
        else:
            raise MappingError(f"source {src.table!r} needs endpoint or endpoint_column")
        frame = pd.DataFrame(
            {
                "participant_id": tab[src.id].astype(str),
                "time": _time_values(tab, src.time),
                "endpoint": endpoint,
                "value": value,
            }
        )
        before = len(frame)
        frame = frame.dropna()
        dropped[src.table] = dropped.get(src.table, 0) + before - len(frame)
        obs_frames.append(frame)
    obs = (
        pd.concat(obs_frames, ignore_index=True)
        if obs_frames
        else pd.DataFrame(columns=["participant_id", "time", "endpoint", "value"])
    )

    visit_frames = []
    for vs in m.visits:
        tab = _filtered(tables[vs.table].copy(), vs.filter)
        _need(tab, [vs.id, vs.time.column, *vs.exposures.values()], vs.table)
        frame = pd.DataFrame({"participant_id": tab[vs.id].astype(str), "time": _time_values(tab, vs.time)})
        for target, column in vs.exposures.items():
            frame[target] = pd.to_numeric(tab[column], errors="coerce").to_numpy()
        visit_frames.append(frame.dropna(subset=["time"]))
    visits = pd.concat(visit_frames, ignore_index=True) if visit_frames else None

    if m.origin == "first_visit":
        times = [obs[["participant_id", "time"]]]
        if visits is not None:
            times.append(visits[["participant_id", "time"]])
        first = pd.concat(times).groupby("participant_id")["time"].min()
        obs["time"] = obs["time"] - obs["participant_id"].map(first)
        if visits is not None:
            visits["time"] = visits["time"] - visits["participant_id"].map(first)
    obs["time_days"] = np.round(obs.pop("time").to_numpy(float)).astype(np.int64)
    obs = obs.groupby(["participant_id", "time_days", "endpoint"], as_index=False).agg({"value": "mean"})
    if visits is not None:
        visits["time_days"] = np.round(visits.pop("time").to_numpy(float)).astype(np.int64)
        agg = {c: "mean" for c in visits.columns if c not in ("participant_id", "time_days")}
        visits = (
            visits.groupby(["participant_id", "time_days"], as_index=False).agg(agg)
            if agg
            else visits.drop_duplicates(["participant_id", "time_days"])
        )
        visits = visits[visits["participant_id"].isin(set(part["participant_id"]))]
    obs = obs[obs["participant_id"].isin(set(part["participant_id"]))]

    markers = None
    if m.markers is not None:
        mt = tables[m.markers.table]
        _need(mt, [m.markers.id, *m.markers.columns.values()], m.markers.table)
        markers = pd.DataFrame(
            [
                (str(r[m.markers.id]), name, float(r[col]))
                for _, r in mt.iterrows()
                for name, col in m.markers.columns.items()
                if pd.notna(r[col])
            ],
            columns=["participant_id", "name", "value"],
        )
    seen = set(obs["participant_id"]) | (set(visits["participant_id"]) if visits is not None else set())
    part = part[part["participant_id"].isin(seen)]
    meta = {"mapping": m.name, "mapping_dropped": dropped, **(metadata or {})}
    return TwinDataset(part, obs, visits, markers, metadata=meta)
