#!/usr/bin/env python3
"""Generate docs pages from the package: the validation status table and the model card.

Run before building the docs. `tests/test_docs.py` fails when a generated page is out of date.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


def validation_page() -> str:
    from symphony_dt import validation_status

    status = validation_status()
    counts = Counter(status.values())
    lines = [
        "# Validation status",
        "",
        "Every public function and class carries one of three labels.",
        "",
        "- **VALIDATED**: the pd_v2 model and the evaluation code of record, as used in the SYMPHONY v2 study.",
        "- **TESTED**: correct on synthetic data and unit tests, not validated on a clinical cohort.",
        "- **EXPERIMENTAL**: works and is tested, but its design is not settled.",
        "",
        f"Counts: {counts['VALIDATED']} VALIDATED, {counts['TESTED']} TESTED, {counts['EXPERIMENTAL']} EXPERIMENTAL.",
        "",
        "| object | label |",
        "|---|---|",
    ]
    for name, label in status.items():
        short = name.removeprefix("symphony_dt.")
        lines.append(f'| `{short}` | <span class="validation-{label}">{label}</span> |')
    return "\n".join(lines) + "\n"


def model_card_page() -> str:
    from symphony_dt.governance import model_card_markdown

    return model_card_markdown("pd_v2")


def main() -> None:
    (DOCS / "validation-status.md").write_text(validation_page())
    (DOCS / "model-card.md").write_text(model_card_page())


if __name__ == "__main__":
    main()
