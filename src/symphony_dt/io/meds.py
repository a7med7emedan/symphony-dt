"""MEDS, the Medical Event Data Standard: a flat event table of subject, time, code and numeric value.

`from_meds` reads MEDS events into a TwinDataset with a code map. `to_meds` writes a TwinDataset as
MEDS events. MEDS needs absolute times, while symphony-dt keeps days since baseline. `to_meds` therefore
places each baseline at a reference date and records that choice in the event metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from symphony_dt.data.dataset import TwinDataset
from symphony_dt.errors import MappingError
from symphony_dt.io.tabular import read_table, to_pandas
from symphony_dt.labels import validation

__all__ = ["MEDS_COLUMNS", "from_meds", "to_meds"]

#: The MEDS core columns.
MEDS_COLUMNS = ("subject_id", "time", "code", "numeric_value")


@validation("TESTED")
def from_meds(
    events: Any,
    endpoint_codes: Mapping[str, str],
    *,
    sex_codes: Mapping[str, int] | None = None,
    birth_code: str = "MEDS_BIRTH",
    cohort_codes: Mapping[str, str] | None = None,
    exposure_codes: Mapping[str, str] | None = None,
) -> TwinDataset:
    """Build a dataset from MEDS events.

    `endpoint_codes` maps a MEDS code to an endpoint, for example `{"LOINC//UPDRS3": "updrs3"}`.
    Static events (no time) supply sex, cohort and birth; age at baseline comes from the birth event.
    `exposure_codes` maps codes to visit-level exposures such as LEDD. Time becomes whole days since each
    subject's first scored event.

    Validation: TESTED.
    """
    frame = to_pandas(read_table(events) if isinstance(events, str | Path) else events).copy()
    missing = [c for c in ("subject_id", "time", "code") if c not in frame.columns]
    if missing:
        raise MappingError(f"MEDS events lack column(s) {missing}")
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    scored = frame[frame["code"].isin(endpoint_codes) & frame["time"].notna()].copy()
    if scored.empty:
        raise MappingError("no MEDS event matches the endpoint codes")
    first = scored.groupby("subject_id")["time"].min()
    scored["time_days"] = ((scored["time"] - scored["subject_id"].map(first)).dt.days).astype(int)
    obs = pd.DataFrame(
        {
            "participant_id": scored["subject_id"],
            "time_days": scored["time_days"],
            "endpoint": scored["code"].map(endpoint_codes),
            "value": pd.to_numeric(scored["numeric_value"], errors="coerce"),
        }
    ).dropna()
    obs = obs.groupby(["participant_id", "time_days", "endpoint"], as_index=False).agg({"value": "mean"})

    ids = sorted(obs["participant_id"].unique())
    part = pd.DataFrame({"participant_id": ids, "cohort": "unknown"}).set_index("participant_id")
    static = frame[frame["subject_id"].isin(ids)]
    if sex_codes:
        sx = static[static["code"].isin(sex_codes)].drop_duplicates("subject_id").set_index("subject_id")
        part["sex"] = sx["code"].map(sex_codes).reindex(part.index)
    if cohort_codes:
        co = static[static["code"].isin(cohort_codes)].drop_duplicates("subject_id").set_index("subject_id")
        part["cohort"] = co["code"].map(cohort_codes).reindex(part.index).fillna("unknown")
    births = static[static["code"] == birth_code].drop_duplicates("subject_id").set_index("subject_id")["time"]
    if len(births):
        part["age_baseline_years"] = ((first - births.reindex(first.index)).dt.days / 365.25).reindex(part.index)
    visits = None
    if exposure_codes:
        ex = frame[frame["code"].isin(exposure_codes) & frame["time"].notna() & frame["subject_id"].isin(ids)].copy()
        if len(ex):
            ex["time_days"] = ((ex["time"] - ex["subject_id"].map(first)).dt.days).astype(int)
            ex = ex[ex["time_days"] >= 0]
            ex["name"] = ex["code"].map(exposure_codes)
            visits = (
                ex.pivot_table(
                    index=["subject_id", "time_days"], columns="name", values="numeric_value", aggfunc="mean"
                )
                .reset_index()
                .rename(columns={"subject_id": "participant_id"})
            )
            visits.columns.name = None
    return TwinDataset(part.reset_index(), obs, visits, metadata={"source": "MEDS"})


@validation("TESTED")
def to_meds(
    dataset: TwinDataset,
    *,
    reference_date: str = "2000-01-01",
    endpoint_codes: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return the dataset as MEDS events, sorted by subject and time.

    Each baseline sits at `reference_date`, so absolute dates carry no real calendar information.
    Endpoint codes default to `SYMPHONY//<endpoint>`.

    Validation: TESTED.
    """
    ref = pd.Timestamp(reference_date)
    codes = dict(endpoint_codes or {e: f"SYMPHONY//{e}" for e in dataset.instruments.names()})
    obs = dataset.observations
    rows = pd.DataFrame(
        {
            "subject_id": obs["participant_id"].astype(str),
            "time": ref + pd.to_timedelta(obs["time_days"], unit="D"),
            "code": obs["endpoint"].map(codes),
            "numeric_value": obs["value"].astype(float),
        }
    )
    static = []
    p = dataset.participants
    if "sex" in p.columns:
        static.append(
            pd.DataFrame(
                {
                    "subject_id": p["participant_id"],
                    "time": pd.NaT,
                    "code": np.where(p["sex"] == 1, "GENDER//M", "GENDER//F"),
                    "numeric_value": np.nan,
                }
            )
        )
    static.append(
        pd.DataFrame(
            {
                "subject_id": p["participant_id"],
                "time": pd.NaT,
                "code": "SYMPHONY//COHORT//" + p["cohort"].astype(str),
                "numeric_value": np.nan,
            }
        )
    )
    if "age_baseline_years" in p.columns:
        birth = ref - pd.to_timedelta(pd.to_numeric(p["age_baseline_years"]) * 365.25, unit="D")
        static.append(
            pd.DataFrame(
                {
                    "subject_id": p["participant_id"],
                    "time": birth.dt.floor("D"),
                    "code": "MEDS_BIRTH",
                    "numeric_value": np.nan,
                }
            )
        )
    if dataset.visits is not None:
        for col in [c for c in dataset.visits.columns if c not in ("participant_id", "time_days")]:
            v = dataset.visits
            static.append(
                pd.DataFrame(
                    {
                        "subject_id": v["participant_id"].astype(str),
                        "time": ref + pd.to_timedelta(v["time_days"], unit="D"),
                        "code": f"SYMPHONY//{col.upper()}",
                        "numeric_value": v[col].astype(float),
                    }
                )
            )
    out = pd.concat([rows, *static], ignore_index=True)
    out["subject_id"] = out["subject_id"].astype(str)
    out = out.sort_values(["subject_id", "time"], na_position="first", kind="mergesort").reset_index(drop=True)
    out.attrs["meds_note"] = f"baseline placed at {reference_date}; times are relative, not calendar dates"
    return out
