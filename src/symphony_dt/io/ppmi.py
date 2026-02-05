from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd

from symphony_dt.config import AppConfig
from symphony_dt.logging import get_logger


@dataclass(frozen=True)
class PPMIData:
    baseline: pd.DataFrame
    clinic: pd.DataFrame
    omics: pd.DataFrame


_REQUIRED_BASELINE_COLS = ["patient_id", "sex", "age_baseline_years", "is_pd_like"]
_REQUIRED_CLINIC_COLS = [
    "patient_id",
    "visit_id",
    "t_days_since_baseline",
    "updrs3_total",
    "moca_total",
    "scopa_aut_total",
    "ledd_total",
    "miss_updrs3",
    "miss_moca",
    "miss_scopa",
    "sex",
    "age_baseline_years",
    "is_pd_like",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _validate_columns(df: pd.DataFrame, required: Sequence[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def _validate_sex_values(df: pd.DataFrame, col: str, table_name: str) -> None:
    vals = set(df[col].dropna().astype(str).str.upper().unique().tolist())
    allowed = {"M", "F"}
    bad = sorted([v for v in vals if v not in allowed])
    if bad:
        raise ValueError(f"{table_name}.{col} has unexpected values: {bad} (allowed: {sorted(allowed)})")


def _coerce_types(baseline: pd.DataFrame, clinic: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline = baseline.copy()
    clinic = clinic.copy()
    baseline["patient_id"] = baseline["patient_id"].astype(int)
    clinic["patient_id"] = clinic["patient_id"].astype(int)
    clinic["visit_id"] = clinic["visit_id"].astype(int)
    clinic["t_days_since_baseline"] = pd.to_numeric(clinic["t_days_since_baseline"], errors="coerce")
    for c in ["updrs3_total", "moca_total", "scopa_aut_total", "ledd_total"]:
        clinic[c] = pd.to_numeric(clinic[c], errors="coerce")
    for c in ["miss_updrs3", "miss_moca", "miss_scopa"]:
        clinic[c] = clinic[c].astype(bool)
    baseline["sex"] = baseline["sex"].astype(str).str.upper()
    clinic["sex"] = clinic["sex"].astype(str).str.upper()
    baseline["age_baseline_years"] = pd.to_numeric(baseline["age_baseline_years"], errors="coerce")
    clinic["age_baseline_years"] = pd.to_numeric(clinic["age_baseline_years"], errors="coerce")
    baseline["is_pd_like"] = baseline["is_pd_like"].astype(bool)
    clinic["is_pd_like"] = clinic["is_pd_like"].astype(bool)
    return baseline, clinic


def _select_paths(cfg: AppConfig) -> Tuple[Path, Path, Path]:
    input_dir = Path(cfg.io.input_dir)
    if cfg.io.cohort == "PDLIKE":
        baseline_path = input_dir / cfg.io.baseline_filename_pdlike
        clinic_path = input_dir / cfg.io.clinic_filename_pdlike
    else:
        baseline_path = input_dir / cfg.io.baseline_filename_all
        clinic_path = input_dir / cfg.io.clinic_filename_all
    omics_path = input_dir / cfg.io.omics_filename
    return baseline_path, clinic_path, omics_path


def load_ppmi_inputs(cfg: AppConfig) -> PPMIData:
    log = get_logger(__name__)
    baseline_path, clinic_path, omics_path = _select_paths(cfg)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
    if not clinic_path.exists():
        raise FileNotFoundError(f"Clinic file not found: {clinic_path}")

    baseline = _read_csv(baseline_path)
    clinic = _read_csv(clinic_path)

    _validate_columns(baseline, _REQUIRED_BASELINE_COLS, "baseline")
    _validate_columns(clinic, _REQUIRED_CLINIC_COLS, "clinic")

    baseline, clinic = _coerce_types(baseline, clinic)

    if cfg.cohort.require_sex:
        _validate_sex_values(baseline, "sex", "baseline")
        _validate_sex_values(clinic, "sex", "clinic")

    if cfg.cohort.require_age_baseline_years:
        if baseline["age_baseline_years"].isna().any():
            raise ValueError("baseline.age_baseline_years has missing values but require_age_baseline_years=true")
        if clinic["age_baseline_years"].isna().any():
            raise ValueError("clinic.age_baseline_years has missing values but require_age_baseline_years=true")

    if cfg.cohort.pd_like_only:
        baseline = baseline[baseline["is_pd_like"]].copy()
        clinic = clinic[clinic["is_pd_like"]].copy()

    bad = (
        clinic.sort_values(["patient_id", "t_days_since_baseline"])
        .groupby("patient_id")["t_days_since_baseline"]
        .apply(lambda s: (s.diff().dropna() <= 0).any())
    )
    bad_ids = bad[bad].index.tolist()
    if bad_ids:
        raise ValueError(f"Non-increasing t_days_since_baseline detected for patient_ids (first 10): {bad_ids[:10]}")

    if omics_path.exists():
        omics = _read_csv(omics_path)
        if "patient_id" not in omics.columns:
            raise ValueError("omics file missing 'patient_id' column")
        omics = omics.copy()
        omics["patient_id"] = pd.to_numeric(omics["patient_id"], errors="coerce").astype("Int64")
        omics = omics.dropna(subset=["patient_id"]).copy()
        omics["patient_id"] = omics["patient_id"].astype(int)
    else:
        log.warning("Omics file not found; proceeding with empty omics table: %s", omics_path)
        omics = pd.DataFrame({"patient_id": pd.Series(dtype="int64")})

    clinic_patients = set(clinic["patient_id"].unique().tolist())
    baseline = baseline[baseline["patient_id"].isin(clinic_patients)].copy()
    omics = omics[omics["patient_id"].isin(clinic_patients)].copy()

    log.info("Loaded baseline rows=%d, clinic rows=%d, omics rows=%d", len(baseline), len(clinic), len(omics))
    return PPMIData(baseline=baseline, clinic=clinic, omics=omics)
