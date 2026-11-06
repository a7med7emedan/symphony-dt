"""Adapters from common clinical data layouts to TwinDataset.

Each adapter flattens its source into named tables and hands them to the mapping engine with a
shipped YAML mapping, which the user can copy and edit. No adapter ships data, codes the package
cannot verify, or a crosswalk. Every adapter is TESTED on synthetic inputs only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from symphony_dt.data.dataset import TwinDataset
from symphony_dt.errors import MappingError
from symphony_dt.io.mapping import DatasetMapping, apply_mapping, load_mapping
from symphony_dt.io.tabular import read_table, to_pandas
from symphony_dt.labels import validation

__all__ = ["from_fhir", "from_mapping", "from_omop", "from_ppmi_export", "from_sdtm", "from_wide"]


@validation("TESTED")
def from_mapping(tables: Mapping[str, Any], mapping: DatasetMapping | str | Path) -> TwinDataset:
    """Build a dataset from named tables (pandas or polars) with any mapping.

    Validation: TESTED.
    """
    return apply_mapping({k: to_pandas(v) for k, v in tables.items()}, mapping)


@validation("TESTED")
def from_wide(frame: Any, mapping: DatasetMapping | str | Path | Mapping[str, Any] = "wide") -> TwinDataset:
    """Build a dataset from one wide table: one row per visit, one column per score.

    The shipped `wide` mapping expects `participant_id`, `cohort`, `age`, `sex`, `days`, `updrs3`,
    `moca`, `scopa_aut` and `ledd`. Pass another mapping for other column names.

    Validation: TESTED.
    """
    m = mapping if isinstance(mapping, DatasetMapping) else load_mapping(mapping)
    df = to_pandas(frame)
    names = {m.participants.table, *(s.table for s in m.observations), *(v.table for v in m.visits)}
    return apply_mapping(dict.fromkeys(names, df), m)


@validation("TESTED")
def from_ppmi_export(directory: str | Path, mapping: DatasetMapping | str | Path = "ppmi") -> TwinDataset:
    """Build a dataset from a PPMI study-data download that the user obtained from PPMI.

    Reads the files the mapping names (by default the curated MDS-UPDRS Part III, MoCA, SCOPA-AUT,
    participant status and demographics exports) from `directory`. The package ships no PPMI data
    and no derived values. Check file and column names against your download; PPMI layouts change.

    Validation: TESTED. Tested on synthetic files with the documented layout only.
    """
    m = mapping if isinstance(mapping, DatasetMapping) else load_mapping(mapping)
    root = Path(directory)
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in m.files.items():
        path = root / filename
        if not path.exists():
            raise MappingError(f"{filename} was not found in {root}; the mapping names it for table {name!r}")
        tables[name] = read_table(path)
    if "demographics" in tables and "status" in tables:
        tables["participants"] = tables["status"].merge(
            tables["demographics"], on=m.participants.id, how="left", suffixes=("", "_demo")
        )
    return apply_mapping(tables, m, metadata={"source": "PPMI export supplied by the user"})


@validation("TESTED")
def from_omop(tables: Mapping[str, Any], mapping: DatasetMapping | str | Path = "omop") -> TwinDataset:
    """Build a dataset from OMOP CDM tables: PERSON, MEASUREMENT, OBSERVATION and DRUG_EXPOSURE.

    Keys of `tables` are table names in lower case. The shipped mapping's concept identifiers are
    placeholders: copy it, set the concept ids of your vocabulary and pass the edited file. Age is
    taken at the first record from `year_of_birth`.

    Validation: TESTED.
    """
    t = {k.lower(): to_pandas(v) for k, v in tables.items()}
    if "person" not in t:
        raise MappingError("OMOP input needs a person table")
    m = mapping if isinstance(mapping, DatasetMapping) else load_mapping(mapping)
    for src in m.observations:
        if src.endpoint_column and not src.endpoint_values:
            raise MappingError(f"set concept ids for {src.table} in the OMOP mapping before use")
    person = t["person"].copy()
    dates = []
    for name, col in (("measurement", "measurement_date"), ("observation", "observation_date")):
        if name in t:
            dates.append(t[name][["person_id", col]].rename(columns={col: "date"}))
    if dates:
        first = pd.concat(dates)
        first["date"] = pd.to_datetime(first["date"])
        first_year = first.groupby("person_id")["date"].min().dt.year
        person["age_at_first"] = person["person_id"].map(first_year) - person["year_of_birth"]
    t["person"] = person
    return apply_mapping(t, m, metadata={"source": "OMOP CDM"})


def _coding(resource: Mapping[str, Any]) -> str:
    codings = resource.get("code", {}).get("coding", [])
    return "|".join(f"{c.get('system', '')}#{c.get('code', '')}" for c in codings)


@validation("TESTED")
def from_fhir(
    bundles: Iterable[Mapping[str, Any] | str | Path], mapping: DatasetMapping | str | Path = "fhir"
) -> TwinDataset:
    """Build a dataset from HL7 FHIR R4 Bundles of Patient and Observation resources.

    Bundles may be dicts, JSON strings or paths. An Observation's code is matched as
    `system#code` against the mapping's `endpoint_values`; the shipped codes are placeholders to set.
    Age is taken at the first observation from Patient.birthDate; sex from Patient.gender.

    Validation: TESTED.
    """
    patients, observations = [], []
    for bundle in bundles:
        if isinstance(bundle, str | Path) and Path(str(bundle)).exists():
            data = json.loads(Path(str(bundle)).read_text())
        elif isinstance(bundle, str):
            data = json.loads(bundle)
        else:
            data = dict(bundle)  # type: ignore[arg-type]
        if data.get("resourceType") != "Bundle":
            raise MappingError("each FHIR input must be a Bundle")
        for entry in data.get("entry", []):
            res = entry.get("resource", {})
            kind = res.get("resourceType")
            if kind == "Patient":
                patients.append(
                    {
                        "id": res.get("id"),
                        "birthDate": res.get("birthDate"),
                        "gender": res.get("gender"),
                        "cohort": next(
                            (
                                e.get("valueString")
                                for e in res.get("extension", [])
                                if str(e.get("url", "")).endswith("cohort")
                            ),
                            None,
                        ),
                    }
                )
            elif kind == "Observation":
                subject = str(res.get("subject", {}).get("reference", ""))
                observations.append(
                    {
                        "subject": subject.split("/")[-1],
                        "code": _coding(res),
                        "effective": res.get("effectiveDateTime"),
                        "value": res.get("valueQuantity", {}).get("value"),
                    }
                )
    pat = pd.DataFrame(patients)
    obs = pd.DataFrame(observations)
    if pat.empty or obs.empty:
        raise MappingError("the bundles hold no Patient or no Observation resources")
    m = mapping if isinstance(mapping, DatasetMapping) else load_mapping(mapping)
    for src in m.observations:
        if src.endpoint_column == "code":
            lookup: dict[str, str] = {}
            for code in obs["code"].unique():
                for part in str(code).split("|"):
                    if part in src.endpoint_values:
                        lookup[code] = part
            obs["code"] = obs["code"].map(lambda c, lk=lookup: lk.get(c, c))
    first = obs.assign(date=pd.to_datetime(obs["effective"])).groupby("subject")["date"].min()
    ages: list[float | None] = []
    for pid, birth in zip(pat["id"], pat["birthDate"], strict=True):
        start = first.get(pid)
        ages.append((start - pd.Timestamp(birth)).days / 365.25 if start is not None and birth else None)
    pat["age_at_first"] = ages
    return apply_mapping({"patient": pat, "observation": obs}, m, metadata={"source": "FHIR R4"})


@validation("TESTED")
def from_sdtm(qs: Any, dm: Any, mapping: DatasetMapping | str | Path = "sdtm", ex: Any | None = None) -> TwinDataset:
    """Build a dataset from CDISC SDTM domains: QS (questionnaires) and DM (demographics).

    Time is the study day `QSDY`. Test codes are matched on `QSTESTCD`; the shipped values are examples
    to adapt to your study's controlled terminology. An optional exposure domain can carry LEDD.

    Validation: TESTED.
    """
    tables = {"qs": to_pandas(qs), "dm": to_pandas(dm)}
    if ex is not None:
        tables["ex"] = to_pandas(ex)
    return apply_mapping(tables, mapping, metadata={"source": "CDISC SDTM"})
