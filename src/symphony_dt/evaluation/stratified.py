from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from symphony_dt.evaluation.metrics import ENDPOINTS, interval_coverage_by_patient, mae_by_patient


@dataclass(frozen=True)
class StratifyConfig:
    """
    Methods: stratify metrics by sex, baseline age quartile, and baseline LEDD quartile.
    LEDD is used only for stratification (never in the model).
    """
    age_bins: int = 4
    ledd_bins: int = 4
    # Follow-up count strata used for missingness sensitivity reporting (Methods robustness section)
    followup_bins: Tuple[int, ...] = (2, 3, 4, 5)  # interpreted as: 2, 3, 4, >=5


def _qcut_or_fallback(x: pd.Series, q: int, labels_prefix: str) -> pd.Series:
    """Robust quantile binning that tolerates ties/small samples."""
    s = pd.to_numeric(x, errors="coerce")

    def label_codes(codes: pd.Series) -> pd.Series:
        codes = codes.astype("Int64")
        out = codes.astype("string")
        m = codes.notna() & (codes >= 0)
        out.loc[m] = (labels_prefix + "Q" + (codes.loc[m].astype(int) + 1).astype(str)).astype("string")
        out.loc[~m] = "unknown"
        return out

    try:
        cat = pd.qcut(s, q=q, duplicates="drop")
        return label_codes(cat.cat.codes)
    except Exception:
        pass

    try:
        r = s.rank(method="average", na_option="keep")
        cat = pd.qcut(r, q=q, duplicates="drop")
        return label_codes(cat.cat.codes)
    except Exception:
        out = s.copy().astype("string")
        out[:] = "unknown"
        out.loc[s.notna()] = f"{labels_prefix}Q1"
        return out



def add_strata(
    pred: pd.DataFrame,
    baseline: pd.DataFrame,
    clinic: pd.DataFrame,
    cfg: StratifyConfig = StratifyConfig(),
) -> pd.DataFrame:
    """
    Adds stratification columns to a prediction table:
      - sex
      - age_quartile
      - ledd_quartile (based on anchor LEDD)
      - followup_count_stratum (based on # visits per patient in clinic table)

    The prediction table must contain:
      - patient_id
      - ledd_anchor (for LEDD quartile)
    """
    req = ["patient_id", "ledd_anchor"]
    missing = [c for c in req if c not in pred.columns]
    if missing:
        raise ValueError(f"pred missing required columns for stratification: {missing}")

    b_req = ["patient_id", "sex", "age_baseline_years"]
    b_missing = [c for c in b_req if c not in baseline.columns]
    if b_missing:
        raise ValueError(f"baseline missing required columns for stratification: {b_missing}")

    # Baseline join
    b = baseline[b_req].copy()
    b["patient_id"] = pd.to_numeric(b["patient_id"], errors="coerce").astype("Int64")
    b = b.dropna(subset=["patient_id"]).copy()
    b["patient_id"] = b["patient_id"].astype(int)
    b["sex"] = b["sex"].astype(str).str.upper()

    out = pred.copy()
    out["patient_id"] = pd.to_numeric(out["patient_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["patient_id"]).copy()
    out["patient_id"] = out["patient_id"].astype(int)

    out = out.merge(b, on="patient_id", how="left")

    # Age quartiles from baseline distribution (cohort-wide for current run/scope)
    out["age_quartile"] = _qcut_or_fallback(out["age_baseline_years"], q=cfg.age_bins, labels_prefix="AGE")

    # LEDD quartiles based on anchor LEDD distribution among available anchors
    out["ledd_quartile"] = _qcut_or_fallback(out["ledd_anchor"], q=cfg.ledd_bins, labels_prefix="LEDD")

    # Follow-up count stratum from clinic visits count per patient
    if "patient_id" not in clinic.columns:
        raise ValueError("clinic missing patient_id for follow-up strata")
    counts = clinic.groupby("patient_id").size().rename("n_visits").reset_index()
    counts["patient_id"] = counts["patient_id"].astype(int)
    out = out.merge(counts, on="patient_id", how="left")

    def followup_stratum(n: float) -> str:
        if not np.isfinite(n):
            return "unknown"
        n = int(n)
        if n <= cfg.followup_bins[0]:
            return f"n_visits={cfg.followup_bins[0]}"
        for v in cfg.followup_bins[1:-1]:
            if n == v:
                return f"n_visits={v}"
        return f"n_visits>={cfg.followup_bins[-1]}"

    out["followup_count_stratum"] = out["n_visits"].apply(followup_stratum)
    return out


def evaluation_stratified_mae_table(
    pred: pd.DataFrame,
    baseline: pd.DataFrame,
    clinic: pd.DataFrame,
    scope: str,
    cfg: StratifyConfig = StratifyConfig(),
) -> pd.DataFrame:
    """
    Produces required table: evaluation_stratified_mae.csv

    Rows are grouped by (scope, stratum_type, stratum_value).
    Metric aggregation follows Methods: per-patient average first, then cohort average.
    """
    df = add_strata(pred, baseline, clinic, cfg=cfg)

    # Compute per-patient MAE then attach strata via patient-level merge (stable)
    per_patient = mae_by_patient(df)

    # one patient can have multiple rows with different strata values (e.g., ledd_anchor differs by anchor)
    # For stratified reporting we attach strata by taking the most frequent stratum per patient.
    strata_cols = ["sex", "age_quartile", "followup_count_stratum"]
    # ledd_quartile is anchor-dependent; handle by per-row grouping later using original df.

    # patient-level "mode" helper
    def mode_series(s: pd.Series) -> str:
        s2 = s.dropna().astype(str)
        if s2.empty:
            return "unknown"
        return s2.value_counts().index[0]

    patient_strata = (
        df.groupby("patient_id", sort=True)[strata_cols]
        .agg({c: mode_series for c in strata_cols})
        .reset_index()
    )

    pp = per_patient.merge(patient_strata, on="patient_id", how="left")

    rows: List[Dict[str, object]] = []
    for stratum_type in ["sex", "age_quartile", "followup_count_stratum"]:
        for val, g in pp.groupby(stratum_type, dropna=False, sort=True):
            row: Dict[str, object] = {"scope": scope, "stratum_type": stratum_type, "stratum_value": str(val)}
            row["n_patients"] = int(g["patient_id"].nunique())
            for e in ENDPOINTS:
                row[f"mae_{e}"] = float(pd.to_numeric(g[f"mae_{e}"], errors="coerce").mean(skipna=True))
            row["mae_overall"] = float(pd.to_numeric(g["mae_overall"], errors="coerce").mean(skipna=True))
            rows.append(row)

    # LEDD quartile is anchor-dependent: stratify at the anchor-instance level but average by patient within stratum.
    # Approach:
    #   - compute per-patient-per-leddQ MAE by filtering instances by leddQ then per-patient averaging.
    for val, g in df.groupby("ledd_quartile", dropna=False, sort=True):
        pp2 = mae_by_patient(g)
        row = {"scope": scope, "stratum_type": "ledd_quartile", "stratum_value": str(val)}
        row["n_patients"] = int(pp2["patient_id"].nunique())
        for e in ENDPOINTS:
            row[f"mae_{e}"] = float(pd.to_numeric(pp2[f"mae_{e}"], errors="coerce").mean(skipna=True))
        row["mae_overall"] = float(pd.to_numeric(pp2["mae_overall"], errors="coerce").mean(skipna=True))
        rows.append(row)

    return pd.DataFrame(rows)


def evaluation_stratified_coverage_table(
    pred: pd.DataFrame,
    baseline: pd.DataFrame,
    clinic: pd.DataFrame,
    scope: str,
    cfg: StratifyConfig = StratifyConfig(),
) -> pd.DataFrame:
    """
    Adds stratified coverage rows. Not a separate required filename in your list;
    we will merge these rows into evaluation_coverage.csv in CLI.

    Same grouping logic as stratified MAE.
    """
    df = add_strata(pred, baseline, clinic, cfg=cfg)
    per_patient = interval_coverage_by_patient(df)

    strata_cols = ["sex", "age_quartile", "followup_count_stratum"]

    def mode_series(s: pd.Series) -> str:
        s2 = s.dropna().astype(str)
        if s2.empty:
            return "unknown"
        return s2.value_counts().index[0]

    patient_strata = (
        df.groupby("patient_id", sort=True)[strata_cols]
        .agg({c: mode_series for c in strata_cols})
        .reset_index()
    )

    pp = per_patient.merge(patient_strata, on="patient_id", how="left")

    rows: List[Dict[str, object]] = []
    for stratum_type in ["sex", "age_quartile", "followup_count_stratum"]:
        for val, g in pp.groupby(stratum_type, dropna=False, sort=True):
            row: Dict[str, object] = {"scope": scope, "stratum_type": stratum_type, "stratum_value": str(val)}
            row["n_patients"] = int(g["patient_id"].nunique())
            for e in ENDPOINTS:
                row[f"cov95_{e}"] = float(pd.to_numeric(g[f"cov95_{e}"], errors="coerce").mean(skipna=True))
            row["cov95_overall"] = float(pd.to_numeric(g["cov95_overall"], errors="coerce").mean(skipna=True))
            rows.append(row)

    for val, g in df.groupby("ledd_quartile", dropna=False, sort=True):
        pp2 = interval_coverage_by_patient(g)
        row = {"scope": scope, "stratum_type": "ledd_quartile", "stratum_value": str(val)}
        row["n_patients"] = int(pp2["patient_id"].nunique())
        for e in ENDPOINTS:
            row[f"cov95_{e}"] = float(pd.to_numeric(pp2[f"cov95_{e}"], errors="coerce").mean(skipna=True))
        row["cov95_overall"] = float(pd.to_numeric(pp2["cov95_overall"], errors="coerce").mean(skipna=True))
        rows.append(row)

    return pd.DataFrame(rows)
