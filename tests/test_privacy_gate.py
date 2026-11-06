"""The privacy gate itself: each rule fires on a synthetic offender and the repository passes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_privacy_gate_blocks_and_passes(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import privacy_gate

    (tmp_path / "tests" / "data").mkdir(parents=True)
    (tmp_path / "ok.py").write_text("x = 1\n")
    (tmp_path / "leak.csv").write_text("a,b\n1,2\n")
    (tmp_path / "tests" / "data" / "fixture.csv").write_text("a,b\n1,2\n")
    (tmp_path / "ids.txt").write_text("PATNO,score\n")
    (tmp_path / "big.json").write_text("[" + "0," * 600_000 + "0]")
    (tmp_path / "paths.md").write_text("see /mnt/std-pool/x\n")
    (tmp_path / "nb.ipynb").write_text(json.dumps({"cells": [{"outputs": [{"text": "patno 1"}]}]}))
    found = privacy_gate.check(
        ["ok.py", "leak.csv", "tests/data/fixture.csv", "ids.txt", "big.json", "paths.md", "nb.ipynb"],
        tmp_path,
    )
    codes = sorted(f.split()[0] for f in found)
    assert codes == ["P1", "P2", "P3", "P3", "P4", "P5"]
    assert privacy_gate.main([]) == 0
