"""Slow suite: compile pd_v2, fit a tiny synthetic cohort, resume, forecast both ways.

Run with `pytest --run-slow`. Needs CmdStan. The short chains do not converge, which is expected
and tested: the gates must say so.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from symphony_dt.cli import app
from symphony_dt.data import Standardisation, simulate_cohort
from symphony_dt.diagnostics import GateOverride, GateOverrideWarning
from symphony_dt.evaluate import ScoringSpec, freeze_spec, score_fold
from symphony_dt.fit import chain_is_complete, fit, load_fit
from symphony_dt.forecast import forecast_exact, forecast_fast, split_at_anchor

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    out = tmp_path_factory.mktemp("fit")
    cohort = simulate_cohort(40, seed=21)
    ids = cohort.dataset.participant_ids
    part = cohort.dataset.participants.set_index("participant_id").loc[ids]
    tests = list(range(0, 40, 4))
    split = split_at_anchor(cohort.stan_data, tests, cohort=part["cohort"].tolist(), sex=part["sex"].tolist())
    result = fit(split.fit, chains=4, iter_warmup=150, iter_sampling=100, seed=5, output_dir=out)
    return cohort, split, result, tests


def test_fit_writes_chains_and_manifest(fitted):
    _, split, result, _ = fitted
    assert len(result.csv_files) == 4
    assert all(chain_is_complete(f, 100) for f in result.csv_files)
    m = json.loads((result.output_dir / "fit_manifest.json").read_text())
    assert m["model_sha256"].startswith("061c1a77") and m["data_sha256"] == split.fit.sha256()
    assert result.draws("lp__").shape == (100, 4)
    assert "mu_r.1" in result.columns()


def test_gates_report_short_chains_honestly(fitted):
    report = fitted[2].gates()
    assert report.chain_exclusion["n_chains"] == 4
    assert report.converged == (report.meets_c2 and report.meets_c3 and report.meets_min_retained)
    with pytest.warns(GateOverrideWarning):
        loose = fitted[2].gates(
            override=GateOverride("smoke test", "CI", c2_max_rhat=10.0, c3_min_ess_bulk=1.0)
        )
    assert loose.override is not None


def test_resume_reuses_complete_chains(fitted):
    _, split, result, _ = fitted
    again = fit(
        split.fit,
        chains=4,
        iter_warmup=150,
        iter_sampling=100,
        seed=5,
        output_dir=result.output_dir,
        resume=True,
    )
    assert again.manifest["settings"]["reused_chains"] == [1, 2, 3, 4]
    assert again.csv_files == result.csv_files
    with pytest.raises(FileExistsError):
        fit(split.fit, chains=4, iter_warmup=10, iter_sampling=10, output_dir=result.output_dir)
    loaded = load_fit(result.output_dir)
    assert loaded.csv_files == result.csv_files


def test_inference_data(fitted):
    idata = fitted[2].to_inference_data()
    assert "mu_r" in idata.posterior and "eps" not in idata.posterior


def test_exact_and_fast_routes_agree_on_synthetic_data(fitted, tmp_path):
    cohort, split, result, tests = fitted
    exact = forecast_exact(result, split, total_keep=400, output_dir=tmp_path / "fc")
    assert exact.draws.shape == (400, len(split.meta))
    spec = ScoringSpec.default()
    scores = score_fold(
        split.fit.data,
        split.targets.data,
        split.meta,
        exact.draws,
        tests,
        spec=spec,
        frozen=freeze_spec(spec),
    )
    assert scores.scores["n_targets"] == len(split.meta)
    std = Standardisation.fit(cohort.dataset)
    i = int(split.meta["i"].iloc[0])
    rows = split.meta[(split.meta["i"] == i)]
    pid = cohort.dataset.participant_ids[i]
    hist = cohort.dataset.subset([pid])
    keep = hist.observations["time_days"] <= 2 * 365.25
    hist.observations = hist.observations[keep]
    hist.visits = hist.visits[hist.visits["time_days"] <= 2 * 365.25]
    first = rows.iloc[0]
    fast = forecast_fast(
        result, hist, [float(first["horizon"])], standardisation=std, n_particles=1500, n_draws=200, seed=2
    )
    k = int(first["k"])
    exact_draws = exact.draws[:, rows.index[0]]
    fast_draws = fast.draws[:, k - 1]
    width = np.subtract(*np.quantile(exact_draws, [0.95, 0.05]))
    assert abs(np.median(fast_draws) - np.median(exact_draws)) < max(width, 5.0)


def test_cli_fit_gates_and_ppc(tmp_path):
    runner = CliRunner()
    sim = tmp_path / "sim"
    assert runner.invoke(app, ["simulate", "--n", "20", "--out", str(sim)]).exit_code == 0
    res = runner.invoke(
        app,
        [
            "fit",
            str(sim / "stan_data.json"),
            "--out",
            str(tmp_path / "f"),
            "--warmup",
            "100",
            "--draws",
            "100",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "CONVERGED" in res.output
    assert runner.invoke(app, ["gates", str(tmp_path / "f"), "--json"]).exit_code in (0, 2)
    res = runner.invoke(app, ["ppc", str(tmp_path / "f"), "--out", str(tmp_path / "ppc"), "--keep", "400"])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "ppc" / "ppc_summary.json").exists()
    assert runner.invoke(app, ["check", "--compile"]).exit_code == 0
