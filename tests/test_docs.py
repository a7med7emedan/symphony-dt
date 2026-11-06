"""Documentation checks: generated pages are current, prose rules hold, required statements exist."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generated_pages_are_current():
    sys.path.insert(0, str(ROOT / "tools"))
    import gen_docs_pages

    assert (ROOT / "docs" / "validation-status.md").read_text() == gen_docs_pages.validation_page()
    assert (ROOT / "docs" / "model-card.md").read_text() == gen_docs_pages.model_card_page()


def test_prose_rules_hold():
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "prose_check.py")], cwd=ROOT, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout[-2000:]


def test_required_statements():
    readme = (ROOT / "README.md").read_text()
    assert "Developed by Ahmed Hemedan" in readme
    assert "Validation manuscript in preparation" in readme
    assert "validation manuscript in preparation" in (ROOT / "docs" / "index.md").read_text().lower()
