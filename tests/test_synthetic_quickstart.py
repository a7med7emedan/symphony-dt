import json
import math

import pandas as pd
from typer.testing import CliRunner

from symphony_dt.cli import app
from symphony_dt.inference_backend.conditioning_numpyro import _json_safe
from symphony_dt.inference_backend.forecasting import _diag_value
from symphony_dt.io.ppmi import _REQUIRED_BASELINE_COLS, _REQUIRED_CLINIC_COLS
from symphony_dt.synthetic import make_synthetic_tables

runner = CliRunner()


def test_synthetic_tables_have_the_loader_columns_and_valid_ranges():
    baseline, clinic = make_synthetic_tables(n_patients=30, seed=1)
    assert set(_REQUIRED_BASELINE_COLS) <= set(baseline.columns)
    assert set(_REQUIRED_CLINIC_COLS) <= set(clinic.columns)
    assert baseline["patient_id"].is_unique
    assert set(baseline["sex"]) <= {"M", "F"}
    assert clinic["updrs3_total"].dropna().between(0, 132).all()
    assert clinic["moca_total"].dropna().between(0, 30).all()
    assert clinic["scopa_aut_total"].dropna().between(0, 69).all()
    assert (clinic.groupby("patient_id")["t_days_since_baseline"].diff().dropna() > 0).all()
    assert (clinic["updrs3_total"].isna() == clinic["miss_updrs3"]).all()


def test_synthetic_tables_are_deterministic_per_seed():
    a = make_synthetic_tables(n_patients=10, seed=7)[1]
    b = make_synthetic_tables(n_patients=10, seed=7)[1]
    c = make_synthetic_tables(n_patients=10, seed=8)[1]
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_quickstart_runs_the_pipeline_and_passes_parity(tmp_path):
    r = runner.invoke(app, ["quickstart", "--workdir", str(tmp_path / "qs"), "--n-patients", "40"])
    assert r.exit_code == 0, r.output
    results = next((tmp_path / "qs" / "outputs").glob("symphony-dt__*__demo")) / "results"
    assert (results / "patient_panel.csv").exists()
    assert (results / "forecast_risk_curves.csv").exists()


def test_aggregate_warns_when_shard_and_full_runs_are_mixed(tmp_path):
    qs = tmp_path / "qs"
    assert runner.invoke(app, ["quickstart", "--workdir", str(qs), "--n-patients", "40"]).exit_code == 0
    cfg = qs / "config.synthetic.yaml"
    for i in range(2):
        r = runner.invoke(app, ["run", "--config", str(cfg), "--mode", "shard", "--shard-id", str(i), "--num-shards", "2"])
        assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["aggregate-all", "--runs-root", str(qs / "outputs"), "--out-dir", str(tmp_path / "agg")])
    assert r.exit_code == 0, r.output
    assert "WARNING" in r.output and "__demo" in r.output


def test_json_safe_turns_non_finite_values_into_null():
    assert _json_safe(float("nan")) is None
    assert _json_safe(float("inf")) is None
    assert _json_safe(None) is None
    assert _json_safe(0.8) == 0.8
    assert json.dumps({"x": _json_safe(float("nan"))}) == '{"x": null}'


def test_missing_or_undefined_diagnostics_fail_the_gate():
    diag = {"accept_prob_mean": None, "rhat_r_m": float("nan"), "divergences": 0}
    assert _diag_value(diag, "accept_prob_mean", 0.0) == 0.0
    assert _diag_value(diag, "rhat_r_m", 99.0) == 99.0
    assert _diag_value(diag, "rhat_r_c", 99.0) == 99.0
    assert _diag_value(diag, "divergences", 1.0) == 0.0
    assert math.isclose(_diag_value({"rhat_r_n": 1.004}, "rhat_r_n", 99.0), 1.004)
