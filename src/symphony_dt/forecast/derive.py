from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from symphony_dt.model.observation import SUPPORTS


def horizon_long_to_wide(horizon_long: pd.DataFrame) -> pd.DataFrame:
    if horizon_long.empty:
        return horizon_long.copy()
    key = ["patient_id","anchor_visit_id","target_visit_id","horizon_center_days","t_anchor_days","t_target_days","dt_days"]
    keep = key + [
        "y_anchor_updrs3","y_anchor_moca","y_anchor_scopa",
        "y_target_updrs3","y_target_moca","y_target_scopa",
        "ledd_anchor","ledd_target",
        "pred_mean_updrs3","pred_p05_updrs3","pred_p95_updrs3",
        "pred_mean_moca","pred_p05_moca","pred_p95_moca",
        "pred_mean_scopa","pred_p05_scopa","pred_p95_scopa",
        "engine",
    ]
    cols = [c for c in keep if c in horizon_long.columns] + [c for c in horizon_long.columns if c not in keep]
    return horizon_long[cols].copy()


def horizon_long_to_uq(horizon_long: pd.DataFrame) -> pd.DataFrame:
    df = horizon_long.copy()
    if df.empty:
        return df
    df["pred_width_90_updrs3"] = pd.to_numeric(df["pred_p95_updrs3"], errors="coerce") - pd.to_numeric(df["pred_p05_updrs3"], errors="coerce")
    df["pred_width_90_moca"] = pd.to_numeric(df["pred_p95_moca"], errors="coerce") - pd.to_numeric(df["pred_p05_moca"], errors="coerce")
    df["pred_width_90_scopa"] = pd.to_numeric(df["pred_p95_scopa"], errors="coerce") - pd.to_numeric(df["pred_p05_scopa"], errors="coerce")
    return df


def horizon_long_to_uq_refined(horizon_long_uq: pd.DataFrame) -> pd.DataFrame:
    df = horizon_long_uq.copy()
    if df.empty:
        return df
    rng_u = SUPPORTS["U"].U - SUPPORTS["U"].L
    rng_q = SUPPORTS["Q"].U - SUPPORTS["Q"].L
    rng_s = SUPPORTS["S"].U - SUPPORTS["S"].L
    df["pred_rel_width_90_updrs3"] = pd.to_numeric(df["pred_width_90_updrs3"], errors="coerce") / float(rng_u)
    df["pred_rel_width_90_moca"] = pd.to_numeric(df["pred_width_90_moca"], errors="coerce") / float(rng_q)
    df["pred_rel_width_90_scopa"] = pd.to_numeric(df["pred_width_90_scopa"], errors="coerce") / float(rng_s)
    return df


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _event_prob_from_mean_interval(mean: np.ndarray, p05: np.ndarray, p95: np.ndarray, threshold: np.ndarray, direction: str) -> np.ndarray:
    z = 1.6448536269514722
    sigma = (p95 - p05) / (2.0 * z)
    sigma = np.maximum(sigma, 1e-6)
    u = (threshold - mean) / sigma
    if direction == ">=":
        return 1.0 - _normal_cdf(u)
    if direction == "<=":
        return _normal_cdf(u)
    raise ValueError("direction must be one of: >=, <=")


def build_risk_curves(horizon_long: pd.DataFrame) -> pd.DataFrame:
    if horizon_long.empty:
        return pd.DataFrame()
    df = horizon_long.copy()
    rows: List[Dict[str, object]] = []

    def add_rows(endpoint: str, y_anchor_col: str, mean_col: str, p05_col: str, p95_col: str, thresholds: Sequence[Tuple[str, float]]):
        y0 = pd.to_numeric(df[y_anchor_col], errors="coerce").to_numpy(dtype=float)
        mean = pd.to_numeric(df[mean_col], errors="coerce").to_numpy(dtype=float)
        p05 = pd.to_numeric(df[p05_col], errors="coerce").to_numpy(dtype=float)
        p95 = pd.to_numeric(df[p95_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y0) & np.isfinite(mean) & np.isfinite(p05) & np.isfinite(p95)
        if not ok.any():
            return
        for direction, delta_thr in thresholds:
            thr = y0 + float(delta_thr)
            prob = _event_prob_from_mean_interval(mean, p05, p95, thr, direction=direction)
            for idx in np.where(ok)[0]:
                rows.append(
                    {
                        "patient_id": int(df.iloc[idx]["patient_id"]),
                        "anchor_visit_id": int(df.iloc[idx]["anchor_visit_id"]),
                        "target_visit_id": int(df.iloc[idx]["target_visit_id"]),
                        "horizon_center_days": float(df.iloc[idx]["horizon_center_days"]),
                        "t_anchor_days": float(df.iloc[idx]["t_anchor_days"]),
                        "t_target_days": float(df.iloc[idx]["t_target_days"]),
                        "dt_days": float(df.iloc[idx]["dt_days"]),
                        "endpoint": endpoint,
                        "direction": direction,
                        "delta_threshold": float(delta_thr),
                        "prob_event": float(np.clip(prob[idx], 0.0, 1.0)),
                        "engine": str(df.iloc[idx].get("engine", "")),
                    }
                )

    add_rows("updrs3","y_anchor_updrs3","pred_mean_updrs3","pred_p05_updrs3","pred_p95_updrs3",[(">=",5.0),(">=",10.0)])
    add_rows("moca","y_anchor_moca","pred_mean_moca","pred_p05_moca","pred_p95_moca",[("<=",-2.0),("<=",-3.0)])
    return pd.DataFrame(rows)


def build_horizon_states(horizon_long: pd.DataFrame) -> pd.DataFrame:
    if horizon_long.empty:
        return pd.DataFrame()
    df = horizon_long.copy()
    u = pd.to_numeric(df["y_anchor_updrs3"], errors="coerce") / 132.0
    q = 1.0 - (pd.to_numeric(df["y_anchor_moca"], errors="coerce") / 30.0)
    s = pd.to_numeric(df["y_anchor_scopa"], errors="coerce") / 69.0
    u = u.clip(0.0, 1.0)
    q = q.clip(0.0, 1.0)
    s = s.clip(0.0, 1.0)

    def bin_tertile(x: pd.Series) -> pd.Series:
        try:
            return pd.qcut(x, q=3, labels=[0,1,2])
        except Exception:  # noqa: BLE001
            r = x.rank(method="average", na_option="keep")
            return pd.qcut(r, q=3, labels=[0,1,2])

    bM = bin_tertile(u).astype("Int64")
    bC = bin_tertile(q).astype("Int64")
    bN = bin_tertile(s).astype("Int64")
    state_id = "M"+bM.astype(str)+"_C"+bC.astype(str)+"_N"+bN.astype(str)

    out = df[["patient_id","anchor_visit_id","target_visit_id","horizon_center_days","t_anchor_days","t_target_days","dt_days"]].copy()
    out["xM_proxy"] = u.astype(float)
    out["xC_proxy"] = q.astype(float)
    out["xN_proxy"] = s.astype(float)
    out["state_id"] = state_id.astype(str)
    out["engine"] = df.get("engine", "locf")
    out["state_engine"] = "proxy_bins_v1"
    return out
