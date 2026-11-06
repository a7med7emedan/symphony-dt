"""2.1 modeller workflow: eligibility, formats, MEDS, profile, outputs, plots and the study runner."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from symphony_dt.cli import app
from symphony_dt.data import (
    AgeRange,
    CohortIn,
    CustomRule,
    EligibilityCriteria,
    MaxGap,
    MinEndpointObservations,
    MinFollowUp,
    MinVisits,
    RequiredAtBaseline,
    apply_eligibility,
    profile,
    rule_from_dict,
    simulate_cohort,
)
from symphony_dt.io import from_meds, read_query, read_table, to_meds, write_table


@pytest.fixture(scope="module")
def cohort200():
    return simulate_cohort(200, seed=4)


def test_eligibility_attrition_and_gain(cohort200):
    ds = cohort200.dataset
    rules = [
        CohortIn(("PD",)),
        MinVisits(3),
        MinFollowUp(2.0),
        MaxGap(1.5),
        AgeRange(50, 80),
        RequiredAtBaseline(),
        MinVisits(2, endpoint="moca"),
        MinEndpointObservations("updrs3", 2),
    ]
    res = apply_eligibility(ds, rules)
    att = res.attrition
    assert att["remaining"].is_monotonic_decreasing
    assert att["remaining"].iloc[-1] == res.n_eligible == len(res.eligible.participants)
    assert (att["removed"].iloc[1:].to_numpy() == -np.diff(att["remaining"].to_numpy())).all()
    others = res.decisions.drop(columns=["eligible"])
    for col, gain in zip(others.columns, att["gain"].iloc[1:], strict=True):
        assert gain == int(others.drop(columns=[col]).all(axis=1).sum() - res.n_eligible)
    assert set(res.excluded_ids()).isdisjoint(res.eligible.participant_ids)
    small = res.attrition_table(min_cell=1000)
    assert set(small["remaining"].iloc[1:]) <= {"<1000", 0}
    assert "| step |" in res.to_markdown()


def test_rules_from_dicts_and_custom(cohort200):
    crit = EligibilityCriteria.from_list(
        [{"rule": "cohort_in", "cohorts": ["PD", "Prodromal"]}, {"rule": "min_visits", "n": 2}]
    )
    assert len(crit.rules) == 2
    assert EligibilityCriteria.pd_v2_default().rules
    with pytest.raises(ValueError, match="unknown eligibility rule"):
        rule_from_dict({"rule": "nope"})
    keep_first = CustomRule(function=lambda d: pd.Series({d.participant_ids[0]: True}), text="first only")
    assert apply_eligibility(cohort200.dataset, [keep_first]).n_eligible == 1
    with pytest.raises(ValueError, match="needs a function"):
        CustomRule().evaluate(cohort200.dataset)
    assert MaxGap(3).describe() and MinFollowUp(1).describe()
    no_age = cohort200.dataset.subset(cohort200.dataset.participant_ids[:5])
    no_age.participants = no_age.participants.drop(columns="age_baseline_years")
    assert not AgeRange().evaluate(no_age).any()


@pytest.mark.parametrize("suffix", [".dta", ".jsonl", ".json"])
def test_more_table_formats(tmp_path, suffix):
    frame = pd.DataFrame({"participant_id": ["a", "b"], "value": [1.5, 2.0]})
    back = read_table(write_table(frame, tmp_path / f"t{suffix}"))
    assert list(back["value"]) == [1.5, 2.0]


def test_spss_and_sas(tmp_path):
    pyreadstat = pytest.importorskip("pyreadstat")
    frame = pd.DataFrame({"pid": ["a", "b"], "score": [3.0, 4.0]})
    pyreadstat.write_sav(frame, tmp_path / "t.sav")
    pyreadstat.write_xport(frame, tmp_path / "t.xpt", file_format_version=5)
    assert list(read_table(tmp_path / "t.sav")["score"]) == [3.0, 4.0]
    assert list(read_table(tmp_path / "t.xpt")["score"]) == [3.0, 4.0]


def test_read_query_sqlite_and_duckdb(tmp_path):
    db = tmp_path / "c.sqlite"
    with sqlite3.connect(db) as con:
        pd.DataFrame({"x": [1, 2, 3]}).to_sql("t", con, index=False)
    assert read_query("select sum(x) as s from t", db)["s"].iloc[0] == 6
    with sqlite3.connect(db) as con:
        assert len(read_query("select * from t", con)) == 3
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "c.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("create table t as select 1 as x union all select 4")
    assert read_query("select sum(x) as s from t", path)["s"].iloc[0] == 5


def test_meds_round_trip(cohort200):
    ds = cohort200.dataset.subset(cohort200.dataset.participant_ids[:30])
    events = to_meds(ds)
    assert set(events.columns) == {"subject_id", "time", "code", "numeric_value"}
    back = from_meds(
        events,
        {f"SYMPHONY//{e}": e for e in ("updrs3", "moca", "scopa_aut")},
        sex_codes={"GENDER//M": 1, "GENDER//F": 0},
        exposure_codes={"SYMPHONY//LEDD": "ledd"},
        cohort_codes={"SYMPHONY//COHORT//PD": "PD", "SYMPHONY//COHORT//Prodromal": "Prodromal"},
    )
    a = back.observations.sort_values(["participant_id", "time_days", "endpoint"])
    b = ds.observations.sort_values(["participant_id", "time_days", "endpoint"])
    assert np.allclose(a["value"], b["value"]) and (a["time_days"].to_numpy() == b["time_days"].to_numpy()).all()
    assert back.participants.set_index("participant_id")["sex"].equals(
        ds.participants.set_index("participant_id")["sex"].astype(back.participants["sex"].dtype)
    )
    assert np.allclose(back.participants["age_baseline_years"], ds.participants["age_baseline_years"], atol=0.01)
    from symphony_dt.errors import MappingError

    with pytest.raises(MappingError, match="lack"):
        from_meds(events.drop(columns="code"), {})
    with pytest.raises(MappingError, match="no MEDS event"):
        from_meds(events, {"nope": "updrs3"})


def test_profile(cohort200):
    prof = profile(cohort200.dataset)
    assert set(prof.endpoints["endpoint"]) == {"updrs3", "moca", "scopa_aut"}
    assert 0 <= prof.overview["single_visit_share"] <= 1
    assert "## Endpoints" in prof.to_markdown()
    assert prof.plot() is not None
    tiny = profile(cohort200.dataset.subset(cohort200.dataset.participant_ids[:5]))
    assert tiny.overview["participants"] == "<10"


def test_outputs_tables(tmp_path):
    from symphony_dt.forecast import Forecast
    from symphony_dt.outputs import forecast_table, scores_table, tidy_summary, write_tables

    table = pd.DataFrame(
        {
            "Mean": [1.0, 2.0, 0.0],
            "StdDev": [0.1, 0.2, 0.0],
            "5%": [0, 1, 0],
            "50%": [1, 2, 0],
            "95%": [2, 3, 0],
            "ESS_bulk": [500, 600, 1],
            "ESS_tail": [400, 500, 1],
            "R_hat": [1.0, 1.001, 1.0],
            "MCSE": [0.01, 0.01, 0],
        },
        index=["mu_r[1]", "b[2]", "s0[1,1]"],
    )
    tidy = tidy_summary(table)
    assert list(tidy["parameter"]) == ["mu_r[1]", "b[2]"] and "rhat" in tidy.columns
    fc = Forecast(
        np.random.default_rng(0).normal(10, 1, size=(50, 2)),
        pd.DataFrame({"i": [0, 1], "k": [1, 2], "t": [3.0, 3.5], "horizon": [1.0, 1.5], "y": [10.0, 9.0]}),
    )
    ft = forecast_table(fc, ["a", "b"])
    assert list(ft["participant_id"]) == ["a", "b"] and "q5" in ft.columns and "sd" in ft.columns
    scores = {
        "cells": {
            "UPDRS-III|all": {
                "n_all": 5,
                "n_common": 4,
                "twin_all": {"c95": [0.9, 0.8, 1.0]},
                "twin": {"crps": [1, 0.5, 1.5]},
                "crps_diff_twin_minus_locf": [-1, -2, 0],
            }
        },
        "events": {
            "MoCA|2y": {
                "n": 30,
                "brier": {"tw": [0.1, 0.05, 0.2]},
                "auc": {"tw": [0.7, 0.6, 0.8]},
                "prevalence": [0.3, 0.2, 0.4],
            }
        },
    }
    st = scores_table(scores)
    assert {"twin_all", "twin", "paired", "tw", "events"} <= set(st["method"])
    written = write_tables({"forecasts": ft, "scores": st}, tmp_path)
    assert all(len(v) == 3 for v in written.values()) and (tmp_path / "tables.xlsx").exists()


def test_new_plots(tmp_path, cohort200):
    import matplotlib.pyplot as plt

    from symphony_dt import plots

    ds = cohort200.dataset
    assert plots.plot_cohort_trajectories(ds, "moca", max_participants=20) is not None
    rng = np.random.default_rng(1)
    draws, y = rng.normal(size=(300, 120)), rng.normal(size=120)
    assert plots.plot_ppc_intervals(draws, y) is not None
    assert plots.plot_coverage_calibration(draws, y) is not None
    assert plots.plot_reliability(rng.uniform(size=200), rng.integers(0, 2, 200)) is not None
    res = apply_eligibility(ds, [CohortIn(("PD",)), MinVisits(3)])
    assert plots.plot_attrition(res) is not None
    fig = plots.participant_report(ds, ds.participant_ids[0])
    paths = plots.save_figure(fig, tmp_path / "report", ("png", "svg"))
    assert all(p.endswith((".png", ".svg")) for p in paths)
    files = []
    for c in (1, 2):
        p = tmp_path / f"m_{c}.csv"
        pd.DataFrame(
            {"lp__": rng.normal(size=100), "mu_r.1": rng.normal(size=100), "energy__": rng.normal(size=100).cumsum()}
        ).to_csv(p, index=False)
        files.append(str(p))
    assert plots.plot_trace(files, ["mu_r.1"]) is not None
    assert plots.plot_trace(files, ["mu_r[1]"]) is not None
    assert plots.plot_rank(files, ["mu_r.1", "lp__"]) is not None
    assert plots.plot_energy(files) is not None
    plt.close("all")


def test_study_dry_run_and_skip(tmp_path):
    from symphony_dt.study import STAGES, init_study, load_study, run_study

    path = init_study(tmp_path)
    assert init_study(tmp_path) == path
    text = path.read_text().replace("simulate_n: 120", "simulate_n: 60")
    path.write_text(text)
    cfg = load_study(path)
    first = run_study(cfg, dry_run=True)
    assert set(first.summary().values()) == {"ran"}
    second = run_study(cfg, dry_run=True)
    assert set(second.summary().values()) == {"skipped (complete)"}
    root = tmp_path / "symphony_study"
    assert (root / "01_eligibility" / "attrition.csv").exists() and (root / "04_split" / "targets_meta.csv").exists()
    assert json.loads((root / "00_data" / "manifest.json").read_text())["kind"] == "study:data"
    forced = run_study(cfg, stages=["profile"], force=True)
    assert forced.summary() == {"profile": "ran"}
    with pytest.raises(ValueError, match="unknown stages"):
        run_study(cfg, stages=["nope"])
    assert len(STAGES) == 10
    assert cfg.digest("data") != cfg.digest("sampler")


def test_study_errors_and_table_input(tmp_path):
    from symphony_dt.study import StudyConfig, run_study

    with pytest.raises(ValueError, match="names no data"):
        run_study(StudyConfig(output_dir=str(tmp_path / "x")), dry_run=True)
    wide = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2"],
            "cohort": "PD",
            "age": [60, 60, 70, 70],
            "sex": [0, 0, 1, 1],
            "days": [0, 900, 0, 1000],
            "updrs3": [10, 14, 5, 7],
            "moca": [27, 26, 29, 28],
            "scopa_aut": [5, 6, 2, 3],
            "ledd": [100, 200, 0, 50],
        }
    )
    wide.to_csv(tmp_path / "w.csv", index=False)
    cfg = StudyConfig(
        output_dir=str(tmp_path / "out"),
        data={"path": str(tmp_path / "w.csv")},
        split={"test_fraction": 0.5, "anchor_years": 1.0},
        outputs={"plots": False},
    )
    assert run_study(cfg, dry_run=True).summary()["split"] == "ran"


def test_cli_workflow_commands(tmp_path):
    runner = CliRunner()
    res = runner.invoke(app, ["init", str(tmp_path / "s")])
    assert res.exit_code == 0
    study = tmp_path / "s" / "study.yaml"
    study.write_text(
        study.read_text().replace("simulate_n: 120", "simulate_n: 40").replace("plots: true", "plots: false")
    )
    res = runner.invoke(app, ["run", str(study), "--dry-run"])
    assert res.exit_code == 0, res.output
    data_dir = tmp_path / "s" / "symphony_study" / "00_data" / "dataset"
    rules = tmp_path / "rules.yaml"
    rules.write_text("eligibility:\n  - {rule: min_visits, n: 3}\n")
    assert "| step |" in runner.invoke(app, ["eligibility", str(data_dir), "--rules", str(rules)]).output
    assert "## Endpoints" in runner.invoke(app, ["profile", str(data_dir)]).output
    assert '"title": "symphony-dt study file"' in runner.invoke(app, ["schema", "study"]).output
    assert "FROM python" in runner.invoke(app, ["container", "docker"]).output
    assert "Bootstrap: docker" in runner.invoke(app, ["container", "apptainer"]).output
    assert "#SBATCH --array=1-4" in runner.invoke(app, ["slurm", "data.json"]).output
    bundle = tmp_path / "b.tar.gz"
    assert runner.invoke(app, ["bundle", "write", str(study), "--out", str(bundle)]).exit_code == 0
    assert "MATCH" in runner.invoke(app, ["bundle", "verify", str(bundle)]).output
