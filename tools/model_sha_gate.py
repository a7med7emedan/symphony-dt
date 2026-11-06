#!/usr/bin/env python3
"""Model-sha gate: the frozen pd_v2 model of record must be byte-identical to the study's file.

Reads the file from the source tree, not from an installed package, and compares its sha256 with the
value fixed in the release rules. Exit status 1 on a mismatch.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

MODEL_OF_RECORD = "061c1a770c14c13ba9e2e0728938142279949bfd0091c061e0f5b16b75b5c42c"
CONDITIONAL = "466fde1a32c81dce28cb1706825cd4c53f10562b8d0316fc7b49ccc6365d5d6f"


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "src" / "symphony_dt" / "models" / "pd_v2"
    ok = True
    for name, expected in (("model.stan", MODEL_OF_RECORD), ("conditional.stan", CONDITIONAL)):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        verdict = "MATCH" if digest == expected else "MISMATCH"
        ok &= digest == expected
        print(f"model-sha gate: pd_v2/{name} {digest} {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
