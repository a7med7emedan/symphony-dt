"""End-to-end NUTS path on synthetic data: fit, condition, forecast must not crash and must write valid JSON."""

import json

import pytest
from typer.testing import CliRunner

pytest.importorskip("numpyro")

from symphony_dt.cli import app  # noqa: E402

runner = CliRunner()


def test_nuts_fit_condition_forecast_on_synthetic(tmp_path):
    qs = tmp_path / "qs"
    assert runner.invoke(app, ["quickstart", "--workdir", str(qs), "--n-patients", "20"]).exit_code == 0
    cfg = qs / "config.synthetic.yaml"
    cfg.write_text(cfg.read_text().replace("n_patients: 25", "n_patients: 2"))
    scaffold, cond, post = tmp_path / "scaffold", tmp_path / "cond", tmp_path / "post"

    r = runner.invoke(app, ["nuts", "fit-scaffold", "--config", str(cfg), "--fold-id", "0", "--out-dir", str(scaffold),
                            "--seed", "0", "--num-warmup", "20", "--num-samples", "20", "--num-chains", "1"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["nuts", "condition", "--config", str(cfg), "--drawset-dir", str(scaffold), "--out-dir", str(cond),
                            "--mode", "demo", "--visit-index", "3", "--l-cond", "2", "--seed", "0",
                            "--num-warmup", "10", "--num-samples", "10", "--num-chains", "2"])
    assert r.exit_code == 0, r.output
    diags = list(cond.rglob("diagnostics.json"))
    assert diags, "conditioning wrote no diagnostics"
    for p in diags:
        text = p.read_text()
        assert "NaN" not in text and "Infinity" not in text
        d = json.loads(text)
        assert d["accept_prob_mean"] is not None  # recorded since accept_prob is requested from NumPyro
    r = runner.invoke(app, ["nuts", "forecast", "--config", str(cfg), "--drawset-dir", str(scaffold),
                            "--conditional-dir", str(cond), "--out-results-dir", str(post), "--mode", "demo",
                            "--visit-index", "3", "--l-cond", "2", "--n-draws-predictive", "10", "--seed", "0"])
    assert r.exit_code == 0, r.output
