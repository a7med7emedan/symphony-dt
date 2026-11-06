from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from symphony_dt.logging import get_logger
from symphony_dt.results import export_json, export_text


@dataclass(frozen=True)
class PreprocessingArtifacts:
    report: Dict[str, Any]
    validated_report: Dict[str, Any]
    data_dictionary_md: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_optional_text(path: Path) -> Optional[str]:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _read_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else None


def _infer_supports() -> Dict[str, Dict[str, float]]:
    return {
        "updrs3_total": {"L": 0.0, "U": 132.0},
        "moca_total": {"L": 0.0, "U": 30.0},
        "scopa_aut_total": {"L": 0.0, "U": 69.0},
    }


def _table_profile(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    return {
        "name": name,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_patients": int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0,
        "missing_by_column": missing,
        "dtypes": dtypes,
    }


def _validate_audit_like(report: Dict[str, Any]) -> Tuple[bool, str]:
    required_top = ["supports", "outputs"]
    for k in required_top:
        if k not in report:
            return False, f"missing top-level key: {k}"
    if not isinstance(report.get("supports"), dict):
        return False, "supports must be a dict"
    if not isinstance(report.get("outputs"), dict):
        return False, "outputs must be a dict"
    return True, "ok"


def export_preprocessing_artifacts(
    input_dir: Path,
    run_dir: Path,
    baseline: pd.DataFrame,
    clinic: pd.DataFrame,
    omics: pd.DataFrame,
) -> PreprocessingArtifacts:
    """
    Produces:
      - preprocessing_report.json
      - preprocessing_report.validated.json
      - data_dictionary.md (copied from input if present; else generated minimal dictionary)
    """
    log = get_logger(__name__)

    audit_in = input_dir / "preprocessing_audit.json"
    dd_in = input_dir / "data_dictionary.md"

    audit = _read_optional_json(audit_in)
    data_dictionary_md = _read_optional_text(dd_in)

    file_hashes: Dict[str, Any] = {}
    for p in input_dir.glob("*.csv"):
        file_hashes[p.name] = {"sha256": _sha256_file(p), "bytes": int(p.stat().st_size)}
    if audit_in.exists():
        file_hashes[audit_in.name] = {"sha256": _sha256_file(audit_in), "bytes": int(audit_in.stat().st_size)}
    if dd_in.exists():
        file_hashes[dd_in.name] = {"sha256": _sha256_file(dd_in), "bytes": int(dd_in.stat().st_size)}

    supports = (audit or {}).get("supports") if audit else None
    if not isinstance(supports, dict):
        supports = _infer_supports()

    report: Dict[str, Any] = {
        "supports": supports,
        "inputs": {
            "input_dir": str(input_dir),
            "file_hashes": file_hashes,
        },
        "tables_profile": {
            "baseline": _table_profile(baseline, "baseline"),
            "clinic_visits": _table_profile(clinic, "clinic_visits"),
            "omics_prebaseline": _table_profile(omics, "omics_prebaseline"),
        },
        "source_preprocessing_audit": audit if audit is not None else None,
        "source_data_dictionary_present": dd_in.exists(),
    }

    validated: Dict[str, Any] = dict(report)
    validated["validated"] = True

    if audit is not None:
        ok, msg = _validate_audit_like(audit)
        validated["audit_schema_valid"] = ok
        validated["audit_schema_message"] = msg
    else:
        validated["audit_schema_valid"] = False
        validated["audit_schema_message"] = "preprocessing_audit.json not found; using inferred supports"

    validated["summaries"] = {
        "n_patients": int(clinic["patient_id"].nunique()) if "patient_id" in clinic.columns else 0,
        "n_clinic_rows": int(len(clinic)),
        "n_baseline_rows": int(len(baseline)),
        "n_omics_rows": int(len(omics)),
        "omics_available_patients": int(omics["patient_id"].nunique()) if "patient_id" in omics.columns else 0,
    }

    if data_dictionary_md is None:
        cols = []
        for name, df in [("baseline", baseline), ("clinic_visits", clinic), ("omics_prebaseline", omics)]:
            cols.append(f"## {name}\n")
            cols.append("| column | dtype | missing | notes |\n|---|---:|---:|---|\n")
            for c in df.columns:
                cols.append(f"| {c} | {df[c].dtype} | {int(df[c].isna().sum())} | |\n")
            cols.append("\n")
        data_dictionary_md = "# SYMPHONY exportables — generated data dictionary\n\n" + "".join(cols)
        log.warning("Input data_dictionary.md not found; generated minimal dictionary.")

    export_json(run_dir / "preprocessing_report.json", report)
    export_json(run_dir / "preprocessing_report.validated.json", validated)
    export_text(run_dir / "data_dictionary.md", data_dictionary_md)

    log.info("Wrote preprocessing reports + data_dictionary.md into %s", run_dir)

    return PreprocessingArtifacts(report=report, validated_report=validated, data_dictionary_md=data_dictionary_md)
