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


def _encode_visit_id(series: pd.Series) -> pd.Series:
    """
    Deterministically encode visit_id to integer.

    Supports:
      - numeric visit_id already
      - string codes like SC, BL, V01, V02, ...

    Encoding rule:
      - SC -> -1
      - BL -> 0
      - V## -> ## (e.g., V01 -> 1)
      - otherwise: stable categorical codes after max known code
    """
    s = series.astype(str).str.strip()
    s_up = s.str.upper()

    # numeric fast-path
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        return num.astype(int)

    mapped = pd.Series([None] * len(s_up), index=s_up.index, dtype="float64")
    mapped[s_up == "SC"] = -1
    mapped[s_up == "BL"] = 0

    vmask = s_up.str.match(r"^V\d+$")
    mapped[vmask] = pd.to_numeric(s_up[vmask].str[1:], errors="coerce")

    unmapped = mapped.isna()
    if unmapped.any():
        remaining = s_up[unmapped].fillna("NA")
        cats = sorted(remaining.unique().tolist())
        base = int(pd.to_numeric(mapped, errors="coerce").dropna().max()) if mapped.notna().any() else 0
        code_map = {c: base + 1 + i for i, c in enumerate(cats)}
        mapped.loc[unmapped] = remaining.map(code_map).astype(float)

    return mapped.astype(int)


def _make_times_strictly_increasing(clinic: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """
    Ensure per-patient t_days_since_baseline is strictly increasing.

    Uses a vectorized strategy that preserves patient_id reliably across pandas versions:
      - sort by (patient_id, t_days_since_baseline, visit_id)
      - per patient: t := cummax(t) to remove decreases
      - add eps * cumcount to break ties => strictly increasing
    """
    df = clinic.copy()
    df["t_days_since_baseline"] = pd.to_numeric(df["t_days_since_baseline"], errors="coerce")
    df = df.sort_values(["patient_id", "t_days_since_baseline", "visit_id"]).copy()

    g = df.groupby("patient_id", sort=False)
    t_fixed = g["t_days_since_baseline"].cummax()
    bump = (g.cumcount().astype(float) * float(eps))

    df["t_days_since_baseline"] = t_fixed + bump
    return df.reset_index(drop=True)


def _coerce_types(baseline: pd.DataFrame, clinic: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline = baseline.copy()
    clinic = clinic.copy()

    baseline["patient_id"] = pd.to_numeric(baseline["patient_id"], errors="coerce").astype("Int64")
    clinic["patient_id"] = pd.to_numeric(clinic["patient_id"], errors="coerce").astype("Int64")
    baseline = baseline.dropna(subset=["patient_id"]).copy()
    clinic = clinic.dropna(subset=["patient_id"]).copy()
    baseline["patient_id"] = baseline["patient_id"].astype(int)
    clinic["patient_id"] = clinic["patient_id"].astype(int)

    clinic["visit_id"] = _encode_visit_id(clinic["visit_id"])

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

    clinic = _make_times_strictly_increasing(clinic)

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

    if omics_path.exists():
        omics = _read_csv(omics_path)
        if "patient_id" not in omics.columns:
            raise ValueError("omics file missing 'patient_id' column")
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
