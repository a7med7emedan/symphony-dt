"""Package surface: labels, the frozen model, the registry, about(), the CLI, plots and the gates."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import symphony_dt
from symphony_dt.cli import app
from symphony_dt.errors import ModelIntegrityError

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "061c1a770c14c13ba9e2e0728938142279949bfd0091c061e0f5b16b75b5c42c"
PUBLIC = [
    "data",
    "io",
    "models",
    "fit",
    "diagnostics",
    "forecast",
    "evaluate",
    "calibration",
    "governance",
    "report",
    "plots",
    "_about",
    "config",
    "labels",
]


def test_import_prints_nothing_and_is_light():
    code = "import sys, symphony_dt; print(sorted(m for m in ('cmdstanpy', 'arviz', 'matplotlib') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]" and out.stderr == ""


def test_lazy_attributes():
    assert symphony_dt.data.TwinDataset is not None
    with pytest.raises(AttributeError):
        _ = symphony_dt.nothing_here


def test_every_public_callable_is_labelled_consistently():
    status = symphony_dt.validation_status()
    missing = []
    for mod_name in PUBLIC:
        mod = importlib.import_module(f"symphony_dt.{mod_name}")
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name)
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            label = getattr(obj, "__validation__", None)
            doc = inspect.getdoc(obj) or ""
            found = re.search(r"Validation: (VALIDATED|TESTED|EXPERIMENTAL)", doc)
            if label is None or found is None or found.group(1) != label.value:
                missing.append(f"{mod_name}.{name}")
    assert not missing, missing
    assert set(status.values()) <= {"VALIDATED", "TESTED", "EXPERIMENTAL"}


def test_model_of_record_is_byte_identical():
    from symphony_dt.models import get_model, list_models, model_card, pd_v2

    path = pd_v2.HERE / "model.stan"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA == pd_v2.MODEL_SHA256
    assert pd_v2.verify()["model.stan"] == MODEL_SHA
    spec = get_model()
    assert spec.validation == "VALIDATED" and "pd_v2" in list_models()
    assert "Known limits" in model_card()
    assert subprocess.run([sys.executable, str(ROOT / "tools" / "model_sha_gate.py")]).returncode == 0


def test_tampering_is_detected(tmp_path):
    from symphony_dt.models import ModelSpec, register_model

    bad = tmp_path / "m.stan"
    bad.write_text("model {}")
    spec = ModelSpec(name="tmp_model", programs={"model": bad}, digests={"model": "0" * 64})
    with pytest.raises(ModelIntegrityError):
        spec.verify()
    with pytest.raises(ModelIntegrityError):
        register_model(spec)
    good = ModelSpec(
        name="tmp_ok",
        programs={"model": bad},
        digests={"model": hashlib.sha256(bad.read_bytes()).hexdigest()},
    )
    register_model(good)
    with pytest.raises(ValueError, match="already"):
        register_model(good)
    with pytest.raises(ValueError, match="frozen"):
        register_model(ModelSpec(name="pd_v2", programs={"model": bad}, digests={}), replace=True)
    with pytest.raises(KeyError):
        good.program("gq")
    from symphony_dt.models import get_model, model_card

    with pytest.raises(KeyError):
        get_model("missing")
    assert "No card" in model_card("tmp_ok")


def test_cmdstan_helpers(monkeypatch, tmp_path):
    from symphony_dt import models

    monkeypatch.setenv("SYMPHONY_DT_CACHE", str(tmp_path / "cache"))
    assert models.cache_dir().exists()
    assert models.cmdstan_version() is None or re.match(r"\d+\.\d+", models.cmdstan_version() or "")


def test_about(capsys):
    text = symphony_dt.about()
    assert "Developed by Ahmed Hemedan" in text and symphony_dt.__version__ in text
    assert "a7med7emedan.github.io/symphony-dt" in capsys.readouterr().out


def test_cli_commands(tmp_path):
    runner = CliRunner()
    assert "Developed by Ahmed Hemedan" in runner.invoke(app, ["about"]).output
    sim = tmp_path / "sim"
    assert runner.invoke(app, ["simulate", "--n", "15", "--out", str(sim)]).exit_code == 0
    res = runner.invoke(app, ["validate-data", str(sim / "dataset")])
    assert res.exit_code == 0 and '"valid": true' in res.output
    assert '"model": "pd_v2"' in runner.invoke(app, ["envelope", str(sim / "dataset")]).output
    wide = tmp_path / "w.csv"
    wide.write_text("participant_id,cohort,age,sex,days,updrs3,moca,scopa_aut,ledd\np,PD,60,0,0,10,27,5,0\n")
    assert runner.invoke(app, ["validate-data", str(wide)]).exit_code == 0
    bad = tmp_path / "bad.csv"
    bad.write_text("participant_id,cohort,age,sex,days,updrs3,moca,scopa_aut,ledd\np,PD,60,0,0,10,99,5,0\n")
    assert runner.invoke(app, ["validate-data", str(bad)]).exit_code == 1
    split = tmp_path / "split"
    assert runner.invoke(app, ["split", str(sim / "stan_data.json"), "--out", str(split)]).exit_code == 0
    assert (split / "targets_meta.csv").exists()
    meta = json.loads((sim / "manifest.json").read_text())
    assert meta["outputs"]["sha256"]
    assert (
        runner.invoke(
            app, ["report", str(sim / "manifest.json"), "--out", str(tmp_path / "r.html")]
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["reproduce", str(sim / "manifest.json")]).exit_code == 0
    gq = tmp_path / "gq.csv"
    import pandas as pd

    n_tgt = len(pd.read_csv(split / "targets_meta.csv"))
    rng = np.random.default_rng(0)
    rows = ["lp__," + ",".join(f"y_rep_c.{q + 1}" for q in range(n_tgt))]
    rows += ["0," + ",".join(f"{v:.3f}" for v in rng.uniform(0, 20, n_tgt)) for _ in range(50)]
    gq.write_text("\n".join(rows) + "\n")
    res = runner.invoke(app, ["evaluate", str(split), str(gq), "--out", str(tmp_path / "scores")])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "scores" / "scores.json").exists() and (
        tmp_path / "scores" / "frozen_spec.json"
    ).exists()


def test_cli_check_runs():
    res = CliRunner().invoke(app, ["check"])
    assert "pd_v2 model sha256" in res.output


def test_plots(cohort):
    import matplotlib.pyplot as plt

    from symphony_dt import plots
    from symphony_dt.forecast import Forecast

    pid = cohort.dataset.participant_ids[0]
    import pandas as pd

    tg = pd.DataFrame(
        {"k": [1, 1], "horizon": [0.5, 1.0], "participant_id": [pid, pid], "y": [10.0, 12.0], "i": [0, 0]}
    )
    fc = Forecast(np.random.default_rng(0).normal(10, 2, size=(100, 2)), tg)
    assert plots.plot_trajectory(cohort.dataset, pid, forecast=fc) is not None
    assert plots.plot_forecast(fc, participant=0) is not None
    cells = {"A|all": {m: {"c95": [0.9, 0.8, 1.0], "c50": [0.5, 0.4, 0.6]} for m in ("twin", "locf", "lmm")}}
    cells["A|all"]["crps_diff_twin_minus_locf"] = [-1, -2, 0]
    cells["A|all"]["crps_diff_twin_minus_lmm"] = [0.5, -0.5, 1]
    assert plots.plot_coverage_by_window({"cells": cells}) is not None
    assert plots.plot_coverage_by_window({"cells": cells}, level="c50") is not None
    assert plots.plot_crps_differences({"cells": cells}) is not None
    assert plots.plot_rank_histogram(np.arange(200) % 200, 199, title="mu") is not None
    assert plots.plot_pit(np.linspace(0, 1, 50)) is not None
    assert plots.plot_effects({"a": np.arange(10.0), "b__": np.zeros(10)}) is not None
    plt.close("all")
