#!/usr/bin/env python3
"""Remove stderr streams and execution timing from executed tutorial notebooks.

Stderr carries warnings with local paths; timing metadata changes on every run. Neither belongs in
the published tutorials.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def clean(path: Path) -> None:
    nb = json.loads(path.read_text())
    for cell in nb.get("cells", []):
        if "outputs" in cell:
            cell["outputs"] = [o for o in cell["outputs"] if o.get("name") != "stderr"]
        cell.get("metadata", {}).pop("execution", None)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv] or sorted(Path("docs/tutorials").glob("*.ipynb"))
    for path in targets:
        clean(path)
        print(f"cleaned {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
