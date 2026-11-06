"""`nuts select-tau-conf` on quickstart data with hand-made conditioning outputs (no NumPyro needed)."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from symphony_dt.cli import app
from symphony_dt.config import load_config
from symphony_dt.cv.splits import fold_train_test_ids, make_stratified_folds
from symphony_dt.inference.store import PosteriorDrawSet, save_drawset
from symphony_dt.inference_backend.conditioning_store import ConditionalRunArtifacts, save_conditional_run
from symphony_dt.io.ppmi import load_ppmi_inputs
from symphony_dt.trust.tau_conf import (
    TauConfSelectionConfig,
    load_tau_conf_json,
    select_tau_conf,
    selection_report,
)

runner = CliRunner()
L_COND = 10
VISIT = 1
_OK_DIAG = {"divergences": 0, "accept_prob_mean": 0.8, "rhat_r_m": 1.0, "rhat_r_c": 1.0, "rhat_r_n": 1.0}


def _drawset(path: Path) -> None:
    n = 200
    arrays = {
        "C_free": np.zeros((n, 6)),
        "sigma_xi": np.full((n, 3), 0.15),
        "mu_r": np.full((n, 3), -6.0),
        "a_u": np.zeros(n), "b_u": np.ones(n), "a_q": np.zeros(n), "b_q": np.ones(n),
        "a_s": np.zeros(n), "b_s": np.ones(n),
        "sigma_u": np.full(n, 5.0), "sigma_q": np.full(n, 1.5), "sigma_s": np.full(n, 3.0),
    }
    save_drawset(path, PosteriorDrawSet(arrays=arrays, metadata={}, diagnostics={}), overwrite=True)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("tau")
    qs = root / "qs"
    assert runner.invoke(app, ["quickstart", "--workdir", str(qs), "--n-patients", "20"]).exit_code == 0
    cfg_path = qs / "config.synthetic.yaml"
    cfg = load_config(cfg_path)
    ppmi = load_ppmi_inputs(cfg)
    ids = sorted(ppmi.clinic["patient_id"].unique().tolist())
    folds = make_stratified_folds(baseline=ppmi.baseline, patient_ids=ids, n_folds=cfg.cv.n_folds, seed=cfg.run.seed)
    train_ids, _ = fold_train_test_ids(folds, fold_id=0)

    draws, cond = root / "draws", root / "cond"
    _drawset(draws)
    # Latent spread grows across patients, so Delta spans the tau grid.
    for k, pid in enumerate(train_ids):
        half_width = 0.02 + 0.5 * k / max(1, len(train_ids) - 1)
        for ell in range(L_COND):
            x = np.random.default_rng(1000 * k + ell).uniform(0.3 - half_width / 2, 0.3 + half_width / 2, size=(40, 3))
            art = ConditionalRunArtifacts(samples_x_tn=np.clip(x, 1e-6, 1 - 1e-6), diagnostics=dict(_OK_DIAG), metadata={})
            save_conditional_run(cond / f"patient_{pid}" / f"visit_{VISIT}" / f"draw_{ell}", art, overwrite=True)
    return {"root": root, "cfg": cfg_path, "draws": draws, "cond": cond, "train_ids": train_ids}


def _select(ws, out_json: Path, visit_index: int = VISIT):
    return runner.invoke(app, [
        "nuts", "select-tau-conf", "--config", str(ws["cfg"]), "--drawset-dir", str(ws["draws"]),
        "--conditional-dir", str(ws["cond"]), "--out-json", str(out_json), "--fold-id", "0",
        "--visit-index", str(visit_index), "--l-cond", str(L_COND), "--n-draws-predictive", "60", "--seed", "0",
    ])


def test_select_tau_conf_writes_choice_and_grid(workspace) -> None:
    out_json = workspace["root"] / "tau_conf.json"
    r = _select(workspace, out_json)
    assert r.exit_code == 0, r.output
    text = out_json.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    rep = json.loads(text)

    grid_taus = [g["tau"] for g in rep["grid"]]
    assert grid_taus == list(TauConfSelectionConfig().tau_grid)
    assert rep["tau_conf"] in grid_taus
    assert rep["target_coverage"] == pytest.approx(0.90)
    assert "90%" in rep["interval"]
    assert rep["fold_id"] == 0 and rep["visit_index"] == VISIT and rep["l_cond"] == L_COND
    assert 0 < rep["n_anchors_eligible"] <= rep["n_anchors_considered"]
    assert rep["n_training_patients"] == len(workspace["train_ids"])
    # The fraction kept grows with tau, and the choice respects the minimum when feasible.
    fracs = [g["fraction_kept"] for g in rep["grid"]]
    assert fracs == sorted(fracs)
    if rep["feasible"]:
        assert rep["fraction_kept"] >= rep["min_fraction_kept"]
    chosen = next(g for g in rep["grid"] if g["tau"] == rep["tau_conf"])
    assert chosen["coverage"] == pytest.approx(rep["achieved_coverage"])


def test_forecast_reads_tau_conf_json_and_gates_on_delta(workspace) -> None:
    out_json = workspace["root"] / "tau_conf_fc.json"
    assert _select(workspace, out_json).exit_code == 0
    tau = load_tau_conf_json(out_json)["tau_conf"]
    post = workspace["root"] / "post"
    r = runner.invoke(app, [
        "nuts", "forecast", "--config", str(workspace["cfg"]), "--drawset-dir", str(workspace["draws"]),
        "--conditional-dir", str(workspace["cond"]), "--out-results-dir", str(post), "--mode", "demo",
        "--visit-index", str(VISIT), "--l-cond", str(L_COND), "--n-draws-predictive", "20",
        "--tau-conf-json", str(out_json),
    ])
    assert r.exit_code == 0, r.output
    pred = pd.read_csv(post / "forecast_one_step_predictions.csv")
    assert "delta_iqr" in pred.columns and len(pred) > 0
    assert (pred["delta_iqr"] <= tau + 1e-12).all()

    # A tighter tau_conf from file suppresses the anchors whose Delta exceeds it.
    tight = workspace["root"] / "tau_conf_tight.json"
    tight.write_text(json.dumps({"tau_conf": 0.1, "visit_index": VISIT, "l_cond": L_COND}), encoding="utf-8")
    post2 = workspace["root"] / "post_tight"
    r = runner.invoke(app, [
        "nuts", "forecast", "--config", str(workspace["cfg"]), "--drawset-dir", str(workspace["draws"]),
        "--conditional-dir", str(workspace["cond"]), "--out-results-dir", str(post2), "--mode", "demo",
        "--visit-index", str(VISIT), "--l-cond", str(L_COND), "--n-draws-predictive", "20",
        "--tau-conf-json", str(tight),
    ])
    assert r.exit_code == 0, r.output
    pred2 = pd.read_csv(post2 / "forecast_one_step_predictions.csv")
    assert 0 < len(pred2) < len(pred)
    assert (pred2["delta_iqr"] <= 0.1).all()


def test_forecast_rejects_both_tau_conf_sources(workspace) -> None:
    out_json = workspace["root"] / "tau_conf_both.json"
    out_json.write_text(json.dumps({"tau_conf": 0.2}), encoding="utf-8")
    r = runner.invoke(app, [
        "nuts", "forecast", "--config", str(workspace["cfg"]), "--drawset-dir", str(workspace["draws"]),
        "--conditional-dir", str(workspace["cond"]), "--out-results-dir", str(workspace["root"] / "x"),
        "--tau-conf", "0.3", "--tau-conf-json", str(out_json),
    ])
    assert r.exit_code != 0


def test_select_tau_conf_fails_cleanly_without_eligible_anchors(workspace) -> None:
    out_json = workspace["root"] / "none.json"
    r = _select(workspace, out_json, visit_index=VISIT + 1)  # nothing conditioned at this visit
    assert r.exit_code == 1
    assert not out_json.exists()


@pytest.mark.parametrize("payload", [{}, {"tau_conf": None}, {"tau_conf": 0.0}, {"tau_conf": 1.5}, {"tau_conf": True}])
def test_load_tau_conf_json_rejects_bad_values(tmp_path: Path, payload) -> None:
    p = tmp_path / "t.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_tau_conf_json(p)


def test_target_matches_the_90_percent_interval_and_missing_targets_are_skipped() -> None:
    assert TauConfSelectionConfig().target_coverage == pytest.approx(0.90)
    rng = np.random.default_rng(3)
    n = 120
    delta = rng.uniform(0.0, 0.6, size=n)
    y = rng.normal(size=(n, 3))
    y[::7, 1] = np.nan
    res = select_tau_conf(delta, y, y - 1.0, y + 1.0)
    assert res.feasible
    assert math.isfinite(res.achieved_coverage) and res.achieved_coverage == pytest.approx(1.0)
    rep = selection_report(res)
    assert rep["target_coverage"] == pytest.approx(0.90)
    assert len(rep["grid"]) == len(TauConfSelectionConfig().tau_grid)
