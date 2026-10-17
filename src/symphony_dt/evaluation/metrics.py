from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


ENDPOINTS = ("updrs3", "moca", "scopa")


def _col(name: str) -> str:
    return name


def mae_by_patient(pred: pd.DataFrame) -> pd.DataFrame:
    """
    Methods: participant-level averaging to avoid domination by dense follow-up.

    Input: standardized forecast table (one-step or horizon_long) containing:
      - patient_id
      - y_target_* columns
      - pred_mean_* columns

    Output: per-patient MAE for each endpoint and overall mean (mean across endpoints).
    """
    required = ["patient_id"] + [f"y_target_{e}" for e in ENDPOINTS] + [f"pred_mean_{e}" for e in ENDPOINTS]
    missing = [c for c in required if c not in pred.columns]
    if missing:
        raise ValueError(f"missing required columns for MAE: {missing}")

    df = pred.copy()
    out_rows = []
    for pid, g in df.groupby("patient_id", sort=True):
        row = {"patient_id": int(pid)}
        maes = []
        for e in ENDPOINTS:
            y = pd.to_numeric(g[f"y_target_{e}"], errors="coerce").to_numpy(dtype=float)
            m = pd.to_numeric(g[f"pred_mean_{e}"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y) & np.isfinite(m)
            if mask.any():
                v = float(np.mean(np.abs(m[mask] - y[mask])))
            else:
                v = float("nan")
            row[f"mae_{e}"] = v
            if np.isfinite(v):
                maes.append(v)
        row["mae_overall"] = float(np.mean(maes)) if maes else float("nan")
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def interval_coverage_by_patient(pred: pd.DataFrame) -> pd.DataFrame:
    """
    Coverage of the [pred_p05, pred_p95] interval per patient (participant-level average).

    Every forecast engine writes the 5th and 95th percentiles, so the nominal level is 90%.
    The output columns keep the historical cov95_* names for schema compatibility.

    Requires:
      y_target_* and pred_p05_*, pred_p95_* columns.
    """
    required = ["patient_id"]
    for e in ENDPOINTS:
        required.extend([f"y_target_{e}", f"pred_p05_{e}", f"pred_p95_{e}"])
    missing = [c for c in required if c not in pred.columns]
    if missing:
        raise ValueError(f"missing required columns for coverage: {missing}")

    df = pred.copy()
    out_rows = []
    for pid, g in df.groupby("patient_id", sort=True):
        row = {"patient_id": int(pid)}
        covs = []
        for e in ENDPOINTS:
            y = pd.to_numeric(g[f"y_target_{e}"], errors="coerce").to_numpy(dtype=float)
            lo = pd.to_numeric(g[f"pred_p05_{e}"], errors="coerce").to_numpy(dtype=float)
            hi = pd.to_numeric(g[f"pred_p95_{e}"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
            if mask.any():
                v = float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])))
            else:
                v = float("nan")
            row[f"cov95_{e}"] = v
            if np.isfinite(v):
                covs.append(v)
        row["cov95_overall"] = float(np.mean(covs)) if covs else float("nan")
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def summarize_mae(per_patient: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in ["mae_updrs3", "mae_moca", "mae_scopa", "mae_overall"]:
        if col in per_patient.columns:
            v = pd.to_numeric(per_patient[col], errors="coerce")
            out[col] = float(v.mean(skipna=True))
    return out


def summarize_coverage(per_patient: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in ["cov95_updrs3", "cov95_moca", "cov95_scopa", "cov95_overall"]:
        if col in per_patient.columns:
            v = pd.to_numeric(per_patient[col], errors="coerce")
            out[col] = float(v.mean(skipna=True))
    return out
