#!/usr/bin/env python3
"""Privacy gate: no participant data and no PPMI identifiers may enter this repository.

Rules, applied to every tracked file (or to the staged files with --staged):
  P1  data file types (csv, tsv, parquet, Excel, Arrow, R, SPSS, Stata, HDF5, netCDF, pickle, NumPy)
      are allowed only under tests/data/, the synthetic fixtures.
  P2  no JSON file over 1 MB anywhere.
  P3  no PPMI identifier pattern (PATNO and similar column names) in any file, except the files that
      document or implement the PPMI layout adapter and this gate, listed in ALLOW_IDENTIFIER.
  P4  no reference to the study's restricted data locations.
  P5  no notebook output that carries an identifier pattern.
Exit status 1 on any finding. Run before every commit and every push; CI runs it on every change.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DATA_SUFFIXES = {".csv", ".tsv", ".parquet", ".pq", ".xlsx", ".xls", ".feather", ".arrow", ".ipc",
                 ".rds", ".rdata", ".rda", ".sav", ".dta", ".sas7bdat", ".h5", ".hdf5", ".nc", ".pkl",
                 ".pickle", ".npy", ".npz", ".zip", ".gz", ".tar"}
DATA_ALLOWED_PREFIX = ("tests/data/",)
JSON_LIMIT = 1_000_000
IDENTIFIER = re.compile(r"\b(PATNO|patno|PAT_NO|ppmi_id|PPMI_ID|SUBJECT_ID_PPMI)\b")
RESTRICTED = re.compile(r"(ppmi_data_corpus|/mnt/std-pool|symphony_v2/results/|LONI\s*IDA\s*download)")
ALLOW_IDENTIFIER = {
    "tools/privacy_gate.py",
    "src/symphony_dt/io/mappings/ppmi.yaml",
    "src/symphony_dt/io/adapters.py",
    "docs/interoperability.md",
    "docs/governance.md",
    "tests/test_io.py",
    "tests/test_privacy_gate.py",
    "tests/golden/cases.py",
    "tests/golden/generate.py",
    "tests/golden/stubs/fold_guard.py",
    ".dev/INVENTORY.md",
    ".dev/PROGRESS.md",
}
ALLOW_RESTRICTED = {"tools/privacy_gate.py", "tests/test_privacy_gate.py", ".dev/INVENTORY.md",
                    ".dev/PROGRESS.md"}
STALE_PREFIX = "STALE_"


def tracked(staged: bool) -> list[str]:
    cmd = (["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged
           else ["git", "ls-files"])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def check(paths: list[str], root: Path) -> list[str]:
    findings: list[str] = []
    for rel in paths:
        path = root / rel
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in DATA_SUFFIXES and not rel.startswith(DATA_ALLOWED_PREFIX):
            findings.append(f"P1 data file outside tests/data/: {rel}")
        if suffix == ".json" and path.stat().st_size > JSON_LIMIT:
            findings.append(f"P2 JSON over 1 MB: {rel} ({path.stat().st_size} bytes)")
        if suffix in {".png", ".jpg", ".jpeg", ".ico", ".pdf", ".woff", ".woff2"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        in_stale = rel.startswith(STALE_PREFIX)
        if suffix == ".ipynb":
            try:
                nb = json.loads(text)
            except json.JSONDecodeError:
                findings.append(f"P5 notebook is not valid JSON: {rel}")
                continue
            outputs = json.dumps([c.get("outputs", []) for c in nb.get("cells", [])])
            if IDENTIFIER.search(outputs):
                findings.append(f"P5 notebook output carries an identifier pattern: {rel}")
        if IDENTIFIER.search(text) and rel not in ALLOW_IDENTIFIER and not in_stale:
            line = next(i for i, ln in enumerate(text.splitlines(), 1) if IDENTIFIER.search(ln))
            findings.append(f"P3 identifier pattern in {rel}:{line}")
        if IDENTIFIER.search(text) and in_stale and suffix in DATA_SUFFIXES | {".json", ".txt", ".log"}:
            findings.append(f"P3 identifier pattern in a data-like file {rel}")
        if RESTRICTED.search(text) and rel not in ALLOW_RESTRICTED:
            line = next(i for i, ln in enumerate(text.splitlines(), 1) if RESTRICTED.search(ln))
            findings.append(f"P4 reference to a restricted data location in {rel}:{line}")
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--staged", action="store_true", help="check staged files only")
    ap.add_argument("paths", nargs="*", help="explicit paths (default: all tracked files)")
    args = ap.parse_args(argv)
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                               check=True).stdout.strip())
    paths = args.paths or tracked(args.staged)
    findings = check(paths, root)
    for f in findings:
        print(f"PRIVACY GATE: {f}")
    print(f"privacy gate: {len(paths)} files checked, {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
