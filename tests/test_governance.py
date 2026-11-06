"""Manifests, fingerprints, provenance, cards, TRIPOD+AI, reproduce and reports."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from symphony_dt.cli import app
from symphony_dt.governance import (
    ProvenanceDocument,
    RunManifest,
    data_card_markdown,
    environment,
    fingerprint,
    model_card_markdown,
    reproduce,
    tripod_ai_items,
    tripod_ai_report,
)
from symphony_dt.report import html_report, write_html_report


def test_fingerprint_is_canonical(cohort):
    assert fingerprint({"b": 1, "a": np.array([1, 2])}) == fingerprint({"a": [1, 2], "b": 1})
    assert fingerprint(cohort.dataset) == cohort.dataset.fingerprint()
    assert fingerprint({"x": np.float64(1.5), "p": __import__("pathlib").Path("/a")})


def test_manifest_cycle(tmp_path):
    env = environment()
    assert env["python"] and "numpy" in env["packages"]
    m = RunManifest.create(
        "fit",
        model="pd_v2",
        seeds={"s": 1},
        gates={"converged": True, "diagnostics": {}},
        overrides=[{"reason": "r"}],
        specification={"frozen_sha256": "x"},
        started=0.0,
    )
    path = m.write(tmp_path / "m.json")
    back = RunManifest.read(path)
    assert back.kind == "fit" and back.seeds == {"s": 1}
    assert len(m.sha256()) == 64
    page = m.to_html()
    assert "<h2>Gates</h2>" in page and "Overrides" in page and "Specification" in page


def test_provenance(tmp_path):
    doc = ProvenanceDocument.for_run("fit", {"data": "a" * 64}, {"chains": "b" * 64})
    d = doc.to_dict()
    assert len(d["entity"]) == 2 and d["used"] and d["wasGeneratedBy"] and d["wasAssociatedWith"]
    assert json.loads(doc.write(tmp_path / "p.json").read_text())["prefix"]["prov"].startswith("http")


def test_cards(cohort):
    card = model_card_markdown()
    assert "061c1a770c14c13ba9e2e0728938142279949bfd0091c061e0f5b16b75b5c42c" in card
    assert "Known limits" in card
    dc = data_card_markdown(cohort.dataset, source="synthetic")
    assert "Fingerprint" in dc and "synthetic" in dc and "| PD |" in dc


def test_tripod():
    items = tripod_ai_items()
    assert len(items) >= 25 and items[0]["key"] == "title"
    text = tripod_ai_report({"title": "Title page"})
    assert "Title page" in text and "TO DO" in text and "Collins" in text
    with pytest.raises(KeyError):
        tripod_ai_report({"nope": "x"})


def test_reproduce_simulation(tmp_path):
    out = tmp_path / "sim"
    res = CliRunner().invoke(app, ["simulate", "--n", "12", "--seed", "4", "--out", str(out)])
    assert res.exit_code == 0, res.output
    rep = reproduce(out / "manifest.json", workdir=tmp_path / "again")
    assert rep.reproduced and set(rep.compared) == {"stan_data.json", "truth.json"}
    m = json.loads((out / "manifest.json").read_text())
    m["command"] = None
    (tmp_path / "nocmd.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="no command"):
        reproduce(tmp_path / "nocmd.json")
    m["command"] = ["simulate", "--out", "{out}"]
    m["outputs"] = {}
    (tmp_path / "noout.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="no output"):
        reproduce(tmp_path / "noout.json")


def test_html_report(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1])
    scores = {
        "cells": {
            "UPDRS-III|all": {
                "n_common": 5,
                "twin": {"c95": [0.9, 0.8, 1.0]},
                "crps_diff_twin_minus_locf": [-1, -2, 0],
            }
        }
    }
    page = html_report(
        [{"kind": "fit", "environment": {"python": "3"}}],
        gates={"converged": False},
        scores=scores,
        figures=[("a figure", fig)],
        notes=["note"],
    )
    assert "CONVERGED False" in page and "data:image/png;base64" in page and "Developed by Ahmed Hemedan" in page
    assert write_html_report(tmp_path / "r.html", [], title="t").exists()
