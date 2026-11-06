"""TwinDataset, instruments, envelope, harmonisation, Stan data builder and simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from symphony_dt.config import ModelConfig
from symphony_dt.data import (
    Crosswalk,
    Instrument,
    InstrumentRegistry,
    Standardisation,
    StanData,
    TwinDataset,
    UnitConverter,
    build_stan_data,
    check_envelope,
    convert_units,
    default_registry,
    h_max_in_days,
    harmonise,
    simulate_cohort,
    substeps_days,
    synthetic_visit_patterns,
    visit_patterns,
)
from symphony_dt.errors import DataValidationError


def small():
    p = pd.DataFrame(
        {
            "participant_id": ["a", "b"],
            "cohort": ["PD", "Prodromal"],
            "age_baseline_years": [60.0, 70.0],
            "sex": [0, 1],
        }
    )
    o = pd.DataFrame(
        {
            "participant_id": ["a", "a", "b", "b"],
            "time_days": [0, 365, 0, 400],
            "endpoint": ["updrs3", "moca", "updrs3", "scopa_aut"],
            "value": [10.0, 28, 5, 3],
        }
    )
    return p, o


def test_dataset_basics():
    p, o = small()
    ds = TwinDataset(p, o)
    assert ds.participant_ids == ["a", "b"]
    assert ds.endpoints == ["updrs3", "moca", "scopa_aut"]
    assert len(ds.visit_table()) == 4
    assert "updrs3" in ds.wide_observations().columns
    assert ds.summary(min_cell=10)["participants"] == "<10"
    assert ds.summary(min_cell=1)["participants"] == 2
    assert len(ds.fingerprint()) == 64
    assert ds.subset(["a"]).participant_ids == ["a"]
    assert "TwinDataset" in repr(ds)


def test_fingerprint_ignores_row_order():
    p, o = small()
    a = TwinDataset(p, o).fingerprint()
    b = TwinDataset(p.iloc[::-1], o.iloc[::-1]).fingerprint()
    assert a == b
    o2 = o.copy()
    o2.loc[0, "value"] = 11.0
    assert TwinDataset(p, o2).fingerprint() != a


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda p, o: (p.drop(columns="cohort"), o), "DV001"),
        (lambda p, o: (pd.concat([p, p.iloc[:1]]), o), "DV002"),
        (lambda p, o: (p, o.assign(participant_id=["a", "a", "b", "z"])), "DV003"),
        (lambda p, o: (p, o.assign(time_days=[0, 1.5, 0, 3])), "DV004"),
        (lambda p, o: (p, o.assign(time_days=[0, -1, 0, 3])), "DV004"),
        (lambda p, o: (p, o.assign(endpoint=["updrs3", "x", "updrs3", "moca"])), "DV005"),
        (lambda p, o: (p, o.assign(value=[1.0, np.nan, 2, 3])), "DV006"),
        (lambda p, o: (p, o.assign(value=[1.0, 31, 2, 3])), "DV007"),
        (
            lambda p, o: (p, o.assign(time_days=[0, 0, 0, 3], endpoint=["updrs3", "updrs3", "moca", "moca"])),
            "DV008",
        ),
        (lambda p, o: (p.assign(sex=[0, 2]), o), "DV009"),
        (lambda p, o: (p, o.assign(time_days=[5, 365, 0, 400])), "DV011"),
    ],
)
def test_validation_codes(mutate, code):
    p, o = small()
    p2, o2 = mutate(p, o)
    with pytest.raises(DataValidationError) as err:
        TwinDataset(p2, o2)
    assert err.value.code == code


def test_duplicate_marker_rejected():
    p, o = small()
    m = pd.DataFrame({"participant_id": ["a", "a"], "name": ["x", "x"], "value": [1.0, 2.0]})
    with pytest.raises(DataValidationError, match="DV010"):
        TwinDataset(p, o, markers=m)


def test_instruments():
    reg = default_registry()
    assert reg.names() == ("updrs3", "moca", "scopa_aut")
    assert reg.get("moca").worsening == "decrease"
    assert reg.get("moca").contains(30) and not reg.get("moca").contains(31)
    assert "moca" in reg and len(reg) == 3
    with pytest.raises(KeyError, match="registered"):
        reg.get("nope")
    with pytest.raises(ValueError, match="already"):
        reg.add(reg.get("moca"))
    with pytest.raises(ValueError, match="exceed"):
        Instrument(name="x", lower=1, upper=1, worsening="increase")
    assert len(InstrumentRegistry()) == 0


def test_envelope_flags_and_passes(cohort):
    report = check_envelope(cohort.dataset)
    assert report.model == "pd_v2"
    p, o = small()
    p = p.assign(cohort=["Healthy", "PD"], age_baseline_years=[20.0, 70.0])
    o = o.assign(time_days=[0, 365 * 20, 0, 400])
    r = check_envelope(TwinDataset(p, o))
    for code in ("ENV003", "ENV004", "ENV005", "ENV009", "ENV010"):
        assert code in r.codes()
    assert not r.inside and r.to_dict()["warnings"]
    with pytest.raises(ValueError, match="no envelope"):
        check_envelope(cohort.dataset, model="other")


def test_envelope_versions_missing_and_single_visits():
    p = pd.DataFrame({"participant_id": list("abc"), "cohort": "PD", "age_baseline_years": 60.0, "sex": 0})
    o = pd.DataFrame(
        {
            "participant_id": list("abc"),
            "time_days": 0,
            "endpoint": "updrs3",
            "value": 1.0,
            "instrument_version": "old",
        }
    )
    v = pd.DataFrame({"participant_id": ["a", "a", "b", "c"], "time_days": [0, 0, 0, 0], "ledd": 0.0})
    r = check_envelope(TwinDataset(p, o, visits=v))
    assert {"ENV001", "ENV002", "ENV006", "ENV008", "ENV010"} <= set(r.codes())
    reg = InstrumentRegistry(
        [Instrument(name="updrs3", version="v9", lower=0, upper=132, worsening="increase")]
    )
    r2 = check_envelope(TwinDataset(p, o, instruments=reg))
    assert "ENV002" in r2.codes()


def test_harmonise_and_units():
    p, o = small()
    o = o.assign(instrument_version=["old", "8.1", "old", "x"])
    ds = TwinDataset(p, o, visits=pd.DataFrame({"participant_id": ["a"], "time_days": [0], "ledd": [100.0]}))
    cw = Crosswalk("updrs3", "old", "MDS-UPDRS 2008", lambda v: v + 1, citation="Example 2020")
    out = harmonise(ds, [cw])
    assert out.observations.loc[out.observations["endpoint"] == "updrs3", "value"].tolist() == [11.0, 6.0]
    assert out.metadata["harmonisation"][0]["citation"] == "Example 2020"
    with pytest.raises(ValueError, match="citation"):
        Crosswalk("updrs3", "a", "b", lambda v: v, citation=" ")
    conv = convert_units(ds, [UnitConverter("ledd", 2.0, "x", "mg")])
    assert conv.visits["ledd"].tolist() == [200.0]
    with pytest.raises(KeyError):
        convert_units(ds, [UnitConverter("nope", 2.0, "x", "mg")])
    assert convert_units(TwinDataset(p, o), []) is not None


def test_grid_rule():
    assert h_max_in_days() == (1461, 8)
    assert substeps_days([0, 1, 182, 183, 1461]).tolist() == [0, 1, 1, 2, 8]
    assert substeps_days([]).size == 0
    for bad in ([1.5], [-1], [np.inf]):
        with pytest.raises(ValueError):
            substeps_days(bad)
    with pytest.raises(ValueError):
        h_max_in_days(0.0)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=5000), min_size=1, max_size=30))
def test_grid_rule_properties(days):
    counts = substeps_days(days)
    for d, c in zip(days, counts, strict=True):
        assert (c == 0) == (d == 0)
        if d:
            assert (d / c) <= 182.625 + 1e-9
            assert c == 1 or d / (c - 1) > 182.625


def test_builder_round_trip_and_audit(cohort, tmp_path):
    sd = build_stan_data(cohort.dataset, pathway=["pathway_1", "pathway_2"])
    gen = cohort.stan_data.data
    for key in ("N", "n_obs", "n_steps", "K", "obs_k", "obs_node", "step_start", "obs_count"):
        assert sd.data[key] == gen[key], key
    assert np.allclose(sd.data["y"], gen["y"])
    assert np.allclose(sd.data["h"], gen["h"])
    assert sd.audit["shape"]["participants"] == 40
    assert sd.data["censored"] == [0, 1, 0]
    path = sd.to_json(tmp_path / "d.json")
    assert StanData.from_json(path).data == sd.data
    assert len(sd.sha256()) == 64
    assert sd.n_participants == 40
    assert sd.node_times(0)[0] == 0.0


def test_builder_switches_and_errors(cohort):
    sens = build_stan_data(cohort.dataset, config=ModelConfig.a01_sensitivity())
    assert sens.data["state_dependent_sigma"] == 1
    pub = build_stan_data(cohort.dataset, config=ModelConfig(prior="published"))
    assert pub.data["mu_r_prior_mean"] == 0.0
    p, o = small()
    with pytest.raises(ValueError, match="age and sex"):
        build_stan_data(
            TwinDataset(p.assign(age_baseline_years=[np.nan, 1.0]), o),
            standardisation=Standardisation({"age_baseline_years": (60.0, 1.0), "ledd": (0.0, 0.0)}),
        )
    zero = Standardisation({"age_baseline_years": (60.0, 1.0), "ledd": (0.0, 0.0), "m": (0.0, 0.0)})
    ds = TwinDataset(p, o, markers=pd.DataFrame({"participant_id": ["a"], "name": ["m"], "value": [1.0]}))
    with pytest.raises(ValueError, match="zero training variance"):
        build_stan_data(ds, standardisation=zero, pathway=["m"])
    reg = InstrumentRegistry([Instrument(name="updrs3", lower=0, upper=132, worsening="increase")])
    with pytest.raises(ValueError, match="needs instruments"):
        build_stan_data(TwinDataset(p, o[o["endpoint"] == "updrs3"], instruments=reg))


def test_standardisation(cohort):
    std = Standardisation.fit(cohort.dataset, cohort.dataset.participant_ids[:20], pathway=["pathway_1"])
    assert std.n_train == 20 and set(std.stats) >= {"ledd", "age_baseline_years", "pathway_1"}
    tab = pd.DataFrame([{"name": k, "mean": v[0], "sd": v[1]} for k, v in std.stats.items()])
    assert Standardisation.from_table(tab).get("ledd") == std.get("ledd")
    with pytest.raises(KeyError):
        std.get("nope")


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(censored=(0, 2, 0))
    with pytest.raises(ValueError):
        ModelConfig(state_dependent_sigma=3)
    with pytest.raises(ValueError):
        ModelConfig(sigma_xbar=(0.5, 1.5, 0.5))
    with pytest.raises(ValueError):
        ModelConfig(h_max_years=0)
    assert ModelConfig.primary().switches()["ledd_in_mu_U"] == 1


def test_simulation(cohort):
    assert cohort.dataset.metadata["synthetic"] is True
    assert "mu_r[1]" in cohort.truth()
    assert cohort.latents["s0"].shape == (3, 40)
    again = simulate_cohort(40, seed=11, n_pathway=2)
    assert again.stan_data.data == cohort.stan_data.data
    pats = visit_patterns(cohort.dataset)
    assert len(pats) == 40 and pats[0]["observed"].shape[1] == 3
    reuse = simulate_cohort(10, seed=1, patterns=pats, theta=cohort.theta)
    assert reuse.theta is not cohort.theta and reuse.dataset.participant_ids[0] == "00001"
    with pytest.raises(ValueError):
        synthetic_visit_patterns(0, np.random.default_rng(0))
