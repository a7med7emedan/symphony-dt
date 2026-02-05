import json
from pathlib import Path

from symphony_dt.notebook.ingest import ingest_notebook


def test_ingest_notebook_minimal(tmp_path: Path) -> None:
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["hi"]},
            {"cell_type": "code", "source": ["import pandas as pd\n", "df=pd.read_csv('a.csv')\n", "df.to_csv('b.csv')\n"]},
        ]
    }
    p = tmp_path / "x.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")

    outp = tmp_path / "introspect.json"
    info = ingest_notebook(p, out_path=outp)
    assert info.n_cells == 2
    assert len(info.read_csv_calls) == 1
    assert len(info.to_csv_calls) == 1
    assert outp.exists()
