"""The living twin, the anytime-valid monitor, adaptive conformal, relearning, compute tools and the benchmark."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from symphony_dt import constants as C
from symphony_dt.data import Standardisation, simulate_cohort
from symphony_dt.twin import (
    AdaptiveConformal,
    CalibrationMonitor,
    ParticipantTwin,
    PopulationDraws,
    RelearnPolicy,
    TwinRegistry,
    Visit,
)


@pytest.fixture(scope="module")
def world():
    cohort = simulate_cohort(40, seed=7)
    return cohort, PopulationDraws.from_thetas([cohort.theta], source="oracle"), Standardisation.fit(cohort.dataset)


def _history(ds, pid, until_days):
    full = ds.subset([pid])
    full.observations = full.observations[full.observations["time_days"] <= until_days]
    full.visits = full.visits[full.visits["time_days"] <= until_days]
    return full


def test_monitor_false_alarm_rate_is_controlled():
    rng = np.random.default_rng(0)
    alarms = 0
    for _ in range(300):
        mon = CalibrationMonitor(alpha=0.05)
        for u in rng.uniform(size=300):
            mon.update(float(u))
        alarms += mon.alarm()
    assert alarms / 300 <= 0.08


def test_monitor_detects_miscalibration():
    rng = np.random.default_rng(1)
    for pits in (rng.beta(2, 1, size=300), rng.beta(0.5, 0.5, size=300)):
        mon = CalibrationMonitor(alpha=0.05)
        for u in pits:
            mon.update(float(u), "updrs3")
        assert mon.alarm() and mon.alarm("updrs3")
        assert mon.state()["all"]["alarm_at"] < 300
    mon.reset()
    assert mon.evidence() == pytest.approx(1.0)


def test_adaptive_conformal_long_run_coverage_under_drift():
    rng = np.random.default_rng(2)
    aci = AdaptiveConformal(target=0.1, gamma=0.02)
    for t in range(2000):
        samples = rng.normal(0, 1, size=200)
        y = rng.normal(0.0004 * t, 1 + 0.0003 * t)
        aci.update(samples, float(y), "moca")
    assert abs(aci.state()["moca"]["miss_rate"] - 0.1) < 0.03
    lo, hi = aci.interval(np.arange(100.0), "moca")
    assert lo < hi


def test_relearn_policy():
    mon = CalibrationMonitor()
    policy = RelearnPolicy(max_new_outcomes=10)
    assert not policy.decide(mon, 3).refit
    assert policy.decide(mon, 10).refit
    for _ in range(200):
        mon.update(0.995)
    assert "alarm" in policy.decide(mon, 0).reason


def test_twin_lifecycle(world, tmp_path):
    cohort, pop, std = world
    ds = cohort.dataset
    pid = ds.participant_ids[3]
    obs = ds.subset([pid]).observations
    times = sorted(obs["time_days"].unique())
    twin = ParticipantTwin.create(
        pop,
        _history(ds, pid, times[1]),
        standardisation=std,
        n_particles=300,
        seed=1,
        monitor=CalibrationMonitor(),
        conformal=AdaptiveConformal(),
    )
    assert twin.status()["visits"] == 2
    for t in times[2:4]:
        scores = {r.endpoint: r.value for r in obs[obs["time_days"] == t].itertuples()}
        entry = twin.observe(Visit(int(t), scores))
        assert entry["pit"] and entry["tempering_steps"] >= 1 and 0 < entry["pcn_acceptance"] <= 1
    with pytest.raises(ValueError, match="precedes"):
        twin.observe({"time_days": 0, "scores": {"moca": 25.0}})
    fc = twin.forecast([0.5, 1.0], n_samples=300)
    assert fc.draws.shape == (300, 6)
    assert np.all(fc.draws >= 0) and np.all(fc.draws[:, 1] <= 30)
    parts = pd.DataFrame(fc.metadata["uncertainty"])
    assert (parts[["population_var", "person_var", "measurement_var"]] >= 0).all().all()
    curve = twin.milestone_curve("moca", 26, years=2.0)
    assert curve["probability"].is_monotonic_increasing
    plan = twin.next_visit([0.25, 0.5, 1.0, 3.0], target_years=2.0, n_outcomes=12)
    assert len(plan) == 3 and plan["variance_reduction"].between(-0.5, 1).all()
    path = twin.save(tmp_path / "twin")
    back = ParticipantTwin.load(path)
    assert back.status()["visits"] == twin.status()["visits"] and back.audit[-1]["event"] == "loaded"
    assert back.forecast([1.0], n_samples=100).draws.shape == (100, 3)
    with pytest.raises(ValueError, match="exactly one"):
        ParticipantTwin.create(pop, ds, standardisation=std)


def test_oracle_twin_is_calibrated_on_held_out_visits(world):
    cohort, pop, std = world
    ds = cohort.dataset
    rng = np.random.default_rng(3)
    pits = []
    for pid in ds.participant_ids[:25]:
        obs = ds.subset([pid]).observations
        hist = _history(ds, pid, 730)
        future = obs[obs["time_days"] > 730]
        if hist.observations.empty or future.empty:
            continue
        twin = ParticipantTwin.create(pop, hist, standardisation=std, n_particles=300, seed=2)
        horizons = sorted({(t - twin.time_days) / C.DAYS_PER_YEAR for t in future["time_days"]})
        fc = twin.forecast(horizons, n_samples=400)
        for r in future.itertuples():
            h = (r.time_days - twin.time_days) / C.DAYS_PER_YEAR
            d = fc.draws[:, 3 * horizons.index(h) + C.ENDPOINT_INDEX[r.endpoint] - 1]
            pits.append(np.mean(d < r.value) + rng.uniform() * np.mean(d == r.value))
    pits = np.array(pits)
    assert len(pits) > 80
    assert 0.88 <= np.mean((pits > 0.025) & (pits < 0.975)) <= 1.0
    assert 0.38 <= np.mean((pits > 0.25) & (pits < 0.75)) <= 0.62


def test_registry_ingest_and_relearn(world):
    cohort, pop, std = world
    ds = cohort.dataset.subset(cohort.dataset.participant_ids[:4])
    early = ds.subset(ds.participant_ids)
    early.observations = early.observations[early.observations["time_days"] <= 400]
    early.visits = early.visits[early.visits["time_days"] <= 400]
    reg = TwinRegistry(pop, std, n_particles=150, policy=RelearnPolicy(max_new_outcomes=1))
    assert reg.enrol_all(early) == 4
    late = ds.observations[ds.observations["time_days"] > 400]
    pid = sorted(late["participant_id"].unique())[0]
    later = late[late["participant_id"] == pid]
    t = int(later["time_days"].min())
    entry = reg.ingest(
        pid, {"time_days": t, "scores": {r.endpoint: r.value for r in later[later.time_days == t].itertuples()}}
    )
    assert entry["population_version"] == 1
    assert reg.decide().refit
    version = reg.relearn(lambda data: PopulationDraws.from_thetas([cohort.theta], source="refit"), ds)
    assert version == 2 and reg.lineage[-1]["source"] == "refit" and reg.new_outcomes == 0
    assert len(reg.status()) == 4
    with pytest.raises(KeyError):
        reg.ingest("nobody", {"time_days": 1, "scores": {}})


def test_serve_twin_json_lines(world, tmp_path):
    from symphony_dt.compute import serve_twin

    cohort, pop, std = world
    pid = cohort.dataset.participant_ids[5]
    twin = ParticipantTwin.create(pop, _history(cohort.dataset, pid, 400), standardisation=std, n_particles=150)
    path = twin.save(tmp_path / "t")
    later = cohort.dataset.observations
    later = later[(later.participant_id == pid) & (later.time_days > 400)]
    t = int(later["time_days"].min())
    visit = {"time_days": t, "scores": {r.endpoint: r.value for r in later[later.time_days == t].itertuples()}}
    commands = [
        {"op": "status"},
        {"op": "observe", "visit": visit},
        {"op": "forecast", "horizons": [1.0], "n_samples": 100},
        {"op": "milestone", "endpoint": "updrs3", "threshold": 40, "years": 1},
        {"op": "next_visit", "candidates": [0.5], "target_years": 1.5},
        {"op": "bogus"},
        {"op": "save"},
        {"op": "quit"},
    ]
    out = io.StringIO()
    handled = serve_twin(path, io.StringIO("\n".join(json.dumps(c) for c in commands) + "\n"), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert handled == len(commands)
    assert [r["ok"] for r in replies] == [True, True, True, True, True, False, True, True]


def test_compute_tools(tmp_path):
    from symphony_dt.compute import container_recipe, json_schema, slurm_script, verify_bundle, write_bundle

    a = tmp_path / "a" / "x.json"
    a.parent.mkdir()
    a.write_text("{}")
    b = tmp_path / "b.txt"
    b.write_text("hello")
    bundle = write_bundle([a, b], tmp_path / "out" / "run.tar.gz", note="test")
    assert all(verify_bundle(bundle).values())
    with pytest.raises(ValueError):
        write_bundle([], tmp_path / "none.tar.gz")
    script = slurm_script("d.json", "fit", chains=8, partition="batch", account="acct", modules=["gcc"], cmdstan="/x")
    assert "--array=1-8" in script and "--partition=batch" in script and "module load gcc" in script
    assert "symphony-dt[io]==" in container_recipe("docker", "9.9.9")
    for name in ("study", "dataset", "visit"):
        assert json_schema(name)["title"].startswith("symphony-dt")


def test_benchmark_runs_and_twin_is_competitive():
    from symphony_dt.bench import SCENARIOS, run_benchmark, scenario_cohort

    table = run_benchmark(["sparse"], n=60, n_test=20, n_particles=200, n_samples=200)
    cov = table[(table.method == "twin") & (table.metric == "c95") & (table.window == "all")]["estimate"]
    assert (cov >= 0.8).all()
    assert len(SCENARIOS) == 5
    for name in SCENARIOS:
        ds, theta = scenario_cohort(name, n=30, seed=1)
        assert len(ds.participants) == 30 and "mu_r" in theta
    with pytest.raises(ValueError, match="unknown scenario"):
        scenario_cohort("nope")
