"""Readers, writers, posterior storage and every adapter, on synthetic inputs only.

The PPMI-layout test writes synthetic files with the documented column names into a temporary
folder; no PPMI data exists in this repository.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest

from symphony_dt.errors import MappingError
from symphony_dt.io import (
    from_fhir,
    from_mapping,
    from_omop,
    from_ppmi_export,
    from_sdtm,
    from_wide,
    list_mappings,
    load_mapping,
    read_dataset,
    read_posterior,
    read_table,
    to_pandas,
    write_dataset,
    write_posterior,
    write_table,
)


def wide_frame():
    return pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2"],
            "cohort": ["PD", "PD", "Prodromal", "Prodromal"],
            "age": [61.0, 61.0, 70.0, 70.0],
            "sex": [0, 0, 1, 1],
            "days": [0, 365, 0, 380],
            "updrs3": [20, 24, 3, np.nan],
            "moca": [27, 26, np.nan, 29],
            "scopa_aut": [8, 9, 2, 3],
            "ledd": [300, 400, 0, 0],
        }
    )


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".parquet", ".xlsx", ".arrow"])
def test_table_round_trip(tmp_path, suffix):
    df = wide_frame()
    path = write_table(df, tmp_path / f"t{suffix}")
    back = read_table(path)
    assert list(back.columns) == list(df.columns) and len(back) == 4


def test_table_errors_and_polars(tmp_path):
    with pytest.raises(ValueError, match="infer"):
        read_table(tmp_path / "x.unknown")
    with pytest.raises(ValueError, match="unknown format"):
        read_table(tmp_path / "x.csv", format="nope")
    with pytest.raises(ValueError, match="unknown format"):
        write_table(wide_frame(), tmp_path / "x.csv", format="nope")
    assert len(to_pandas(pl.from_pandas(wide_frame()))) == 4
    with pytest.raises(TypeError):
        to_pandas(42)


def test_wide_adapter_and_dataset_round_trip(tmp_path):
    ds = from_wide(pl.from_pandas(wide_frame()))
    assert ds.participant_ids == ["p1", "p2"] and len(ds.observations) == 10
    assert ds.visits is not None and ds.visits["ledd"].tolist()[:2] == [300.0, 400.0]
    for fmt in ("parquet", "csv"):
        out = write_dataset(ds, tmp_path / fmt, format=fmt)
        assert read_dataset(out).fingerprint() == ds.fingerprint()
    meta = json.loads((tmp_path / "csv" / "dataset.json").read_text())
    meta["fingerprint"] = "0" * 64
    (tmp_path / "csv" / "dataset.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="fingerprint"):
        read_dataset(tmp_path / "csv")


def test_mappings_load_and_validate(tmp_path):
    assert {"wide", "ppmi", "omop", "fhir", "sdtm", "generic_long"} <= set(list_mappings())
    with pytest.raises(MappingError, match="no mapping"):
        load_mapping("nope")
    with pytest.raises(MappingError, match="malformed"):
        load_mapping({"name": "x"})
    path = tmp_path / "m.yaml"
    path.write_text(
        (__import__("importlib").resources.files("symphony_dt.io") / "mappings" / "wide.yaml").read_text()
    )
    assert load_mapping(path).name == "wide"


def test_generic_long_with_months():
    parts = pd.DataFrame(
        {"subject": ["s1", "s2"], "group": ["PD", "PROD"], "age_at_baseline": [60, 65], "sex_code": [1, 0]}
    )
    scores = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2", "s2"],
            "months_since_baseline": [0, 12, 0, 6],
            "instrument": ["UPDRS3", "MOCA", "UPDRS3", "OTHER"],
            "score": [10, 28, 4, 1],
        }
    )
    ds = from_mapping({"participants": parts, "scores": scores}, "generic_long")
    assert set(ds.participants["cohort"]) == {"PD", "Prodromal"}
    assert ds.observations["time_days"].max() == 365
    assert ds.metadata["mapping_dropped"]["scores"] == 1
    with pytest.raises(MappingError, match="needs table"):
        from_mapping({"participants": parts}, "generic_long")
    with pytest.raises(MappingError, match="lacks column"):
        from_mapping({"participants": parts.drop(columns="group"), "scores": scores}, "generic_long")


def test_ppmi_layout_adapter(tmp_path):
    status = pd.DataFrame({"PATNO": [3001, 3002], "COHORT": [1, 4], "AGE_AT_BL": [62.0, 66.0]})
    demo = pd.DataFrame({"PATNO": [3001, 3002], "SEX": [1, 0]})
    up = pd.DataFrame(
        {
            "PATNO": [3001, 3001, 3002],
            "EVENT_ID": ["BL", "V04", "BL"],
            "INFODT": ["01/2015", "01/2016", "03/2015"],
            "NP3TOT": [20, 25, 2],
        }
    )
    moca = pd.DataFrame({"PATNO": [3001, 3002], "INFODT": ["01/2015", "03/2015"], "MCATOT": [27, 30]})
    scopa = pd.DataFrame({"PATNO": [3001], "INFODT": ["01/2016"], "SCAU_TOTAL": [10.0]})
    m = load_mapping("ppmi")
    frames = {"status": status, "demographics": demo, "updrs3": up, "moca": moca, "scopa": scopa}
    for name, frame in frames.items():
        frame.to_csv(tmp_path / m.files[name], index=False)
    ds = from_ppmi_export(tmp_path)
    assert ds.participants.set_index("participant_id").loc["3002", "cohort"] == "Prodromal"
    assert ds.observations.query("participant_id == '3001'")["time_days"].max() == 365
    (tmp_path / m.files["moca"]).unlink()
    with pytest.raises(MappingError, match="was not found"):
        from_ppmi_export(tmp_path)


def test_omop_adapter():
    person = pd.DataFrame(
        {"person_id": [1, 2], "year_of_birth": [1950, 1960], "gender_concept_id": [8507, 8532]}
    )
    meas = pd.DataFrame(
        {
            "person_id": [1, 1, 2],
            "measurement_concept_id": [111, 222, 111],
            "measurement_date": ["2015-01-01", "2016-01-01", "2015-06-01"],
            "value_as_number": [20, 27, 5],
        }
    )
    with pytest.raises(MappingError, match="concept ids"):
        from_omop({"PERSON": person, "MEASUREMENT": meas})
    m = load_mapping("omop").model_dump()
    m["observations"][0]["endpoint_values"] = {"111": "updrs3", "222": "moca"}
    ds = from_omop({"PERSON": person, "MEASUREMENT": meas}, load_mapping(m))
    assert ds.participants.set_index("participant_id").loc["1", "sex"] == 1
    assert ds.participants.set_index("participant_id").loc["1", "age_baseline_years"] == 65
    with pytest.raises(MappingError, match="person"):
        from_omop({"measurement": meas})


def _bundle():
    code = "urn:symphony-dt:placeholder#"
    entries = [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "a",
                "birthDate": "1950-01-01",
                "gender": "male",
                "extension": [{"url": "http://example.org/cohort", "valueString": "PD"}],
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "subject": {"reference": "Patient/a"},
                "code": {"coding": [{"system": "urn:symphony-dt:placeholder", "code": "updrs3-total"}]},
                "effectiveDateTime": "2015-01-01",
                "valueQuantity": {"value": 22},
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "subject": {"reference": "Patient/a"},
                "code": {"coding": [{"system": "urn:symphony-dt:placeholder", "code": "moca-total"}]},
                "effectiveDateTime": "2016-01-01",
                "valueQuantity": {"value": 26},
            }
        },
    ]
    assert code
    return {"resourceType": "Bundle", "entry": entries}


def test_fhir_adapter(tmp_path):
    ds = from_fhir([_bundle()])
    assert ds.participants["cohort"].tolist() == ["PD"]
    assert sorted(ds.observations["endpoint"]) == ["moca", "updrs3"]
    path = tmp_path / "b.json"
    path.write_text(json.dumps(_bundle()))
    assert len(from_fhir([path]).observations) == 2
    assert len(from_fhir([json.dumps(_bundle())]).observations) == 2
    with pytest.raises(MappingError, match="Bundle"):
        from_fhir([{"resourceType": "Patient"}])
    with pytest.raises(MappingError, match="no Patient"):
        from_fhir([{"resourceType": "Bundle", "entry": []}])


def test_sdtm_adapter():
    dm = pd.DataFrame({"USUBJID": ["S-1", "S-2"], "AGE": [60, 70], "SEX": ["M", "F"], "ARMCD": ["PD", "PD"]})
    qs = pd.DataFrame(
        {
            "USUBJID": ["S-1", "S-1", "S-2"],
            "QSTESTCD": ["UPD3TOT", "MOCATOT", "UPD3TOT"],
            "QSDY": [1, 366, 1],
            "QSSTRESN": [18, 28, 4],
        }
    )
    ds = from_sdtm(qs, dm)
    assert ds.observations.query("participant_id == 'S-1'")["time_days"].tolist() == [0, 365]
    assert ds.participants.set_index("participant_id").loc["S-2", "sex"] == 0


def test_posterior_round_trip(tmp_path):
    import arviz as az

    mu = {"mu": np.random.default_rng(0).normal(size=(2, 50))}
    try:
        idata = az.from_dict({"posterior": mu})
    except TypeError:  # ArviZ before 1.0
        idata = az.from_dict(posterior=mu)
    path = write_posterior(idata, tmp_path / "p.nc")
    assert read_posterior(path).posterior["mu"].shape == (2, 50)
