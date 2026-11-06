from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from symphony_dt.results import export_json

_TO_CSV_RE = re.compile(r"\.to_csv\(\s*([rRfF]?[\"'])(?P<path>.+?)\1")
_READ_CSV_RE = re.compile(r"pd\.read_csv\(\s*([rRfF]?[\"'])(?P<path>.+?)\1")
_ASSIGN_RE = re.compile(r"^\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<rhs>.+)$")
_COVERAGE_COL_RE = re.compile(r"\b(?P<col>cov(?:90|95)_(?:updrs3|moca|scopa|overall))\b")


@dataclass(frozen=True)
class NotebookIntrospection:
    notebook_path: str
    n_cells: int
    to_csv_calls: List[Dict[str, Any]]
    read_csv_calls: List[Dict[str, Any]]
    variable_assignments: List[Dict[str, Any]]
    # References to interval coverage columns. cov95_* is a deprecated alias of cov90_*.
    coverage_column_refs: List[Dict[str, Any]] = field(default_factory=list)


def ingest_notebook(notebook_path: Path, out_path: Optional[Path] = None) -> NotebookIntrospection:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    to_csv_calls: List[Dict[str, Any]] = []
    read_csv_calls: List[Dict[str, Any]] = []
    assignments: List[Dict[str, Any]] = []
    coverage_refs: List[Dict[str, Any]] = []

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, str):
            lines = src.splitlines()
        else:
            lines = []
            for x in src:
                lines.extend(str(x).splitlines())

        for ln, line in enumerate(lines, start=1):
            m1 = _TO_CSV_RE.search(line)
            if m1:
                to_csv_calls.append({"cell": idx, "line": ln, "path": m1.group("path"), "code": line.strip()})
            m2 = _READ_CSV_RE.search(line)
            if m2:
                read_csv_calls.append({"cell": idx, "line": ln, "path": m2.group("path"), "code": line.strip()})
            m3 = _ASSIGN_RE.match(line)
            if m3 and "==" not in line and "!=" not in line:
                assignments.append({"cell": idx, "line": ln, "var": m3.group("var"), "rhs": m3.group("rhs")[:200].strip()})
            for m4 in _COVERAGE_COL_RE.finditer(line):
                col = m4.group("col")
                coverage_refs.append({
                    "cell": idx,
                    "line": ln,
                    "column": col,
                    "canonical": col.replace("cov95_", "cov90_", 1),
                    "deprecated": col.startswith("cov95_"),
                })

    result = NotebookIntrospection(
        notebook_path=str(notebook_path),
        n_cells=len(cells),
        to_csv_calls=to_csv_calls,
        read_csv_calls=read_csv_calls,
        variable_assignments=assignments,
        coverage_column_refs=coverage_refs,
    )
    if out_path is not None:
        export_json(out_path, result.__dict__)
    return result
