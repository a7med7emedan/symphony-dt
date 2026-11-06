import json
from pathlib import Path

from symphony_dt.inference_backend.backend import InferenceBackend, detect_backend_from_notebook


def test_detect_numpyro(tmp_path: Path) -> None:
    nb = {
        "cells": [
            {"cell_type": "code", "source": ["import numpyro\n", "from numpyro.infer import NUTS, MCMC\n", "m=MCMC(NUTS(model), 100, 100)\n"]},
        ]
    }
    p = tmp_path / "x.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    cfg = detect_backend_from_notebook(p)
    assert cfg.backend == InferenceBackend.NUMPYRO


def test_detect_unknown(tmp_path: Path) -> None:
    nb = {"cells": [{"cell_type": "code", "source": ["print('hi')\n"]}]}
    p = tmp_path / "y.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    cfg = detect_backend_from_notebook(p)
    assert cfg.backend == InferenceBackend.UNKNOWN
