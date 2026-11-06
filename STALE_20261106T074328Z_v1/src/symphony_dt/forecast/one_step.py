from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from symphony_dt.model.observation import SUPPORTS


def build_one_step_dataset(clinic: pd.DataFrame) -> pd.DataFrame:
    required = [
        "patient_id", "visit_id", "t_days_since_baseline",
        "updrs3_total", "moca_total", "scopa_aut_total", "ledd_total",
        "miss_updrs3", "miss_moca", "miss_scopa",
    ]
    missing = [c for c in required if c not in clinic.columns]
    if missing:
        raise ValueError(f"clinic missing required columns for one-step dataset: {missing}")

    df = clinic.sort_values(["patient_id", "t_days_since_baseline", "visit_id"]).copy()

    rows: List[Dict[str, object]] = []
    for pid, g in df.groupby("patient_id", sort=True):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        for n in range(len(g) - 1):
            a = g.loc[n]
            t = g.loc[n + 1]
            rows.append(
                {
                    "patient_id": int(pid),
                    "anchor_visit_id": int(a["visit_id"]),
                    "target_visit_id": int(t["visit_id"]),
                    "t_anchor_days": float(a["t_days_since_baseline"]),
                    "t_target_days": float(t["t_days_since_baseline"]),
                    "dt_days": float(t["t_days_since_baseline"] - a["t_days_since_baseline"]),
                    "y_anchor_updrs3": None if pd.isna(a["updrs3_total"]) else float(a["updrs3_total"]),
                    "y_anchor_moca": None if pd.isna(a["moca_total"]) else float(a["moca_total"]),
                    "y_anchor_scopa": None if pd.isna(a["scopa_aut_total"]) else float(a["scopa_aut_total"]),
                    "y_target_updrs3": None if pd.isna(t["updrs3_total"]) else float(t["updrs3_total"]),
                    "y_target_moca": None if pd.isna(t["moca_total"]) else float(t["moca_total"]),
                    "y_target_scopa": None if pd.isna(t["scopa_aut_total"]) else float(t["scopa_aut_total"]),
                    "ledd_anchor": None if pd.isna(a["ledd_total"]) else float(a["ledd_total"]),
                    "ledd_target": None if pd.isna(t["ledd_total"]) else float(t["ledd_total"]),
                }
            )
    return pd.DataFrame(rows)


def _clip(v: np.ndarray, L: float, U: float) -> np.ndarray:
    return np.clip(v, L, U)


def predict_one_step_locf(instances: pd.DataFrame) -> pd.DataFrame:
    if instances.empty:
        return instances.copy()

    df = instances.copy()

    ranges = {
        "updrs3": SUPPORTS["U"].U - SUPPORTS["U"].L,
        "moca": SUPPORTS["Q"].U - SUPPORTS["Q"].L,
        "scopa": SUPPORTS["S"].U - SUPPORTS["S"].L,
    }
    half_width = {k: 0.10 * v for k, v in ranges.items()}
    mid = {
        "updrs3": (SUPPORTS["U"].L + SUPPORTS["U"].U) / 2.0,
        "moca": (SUPPORTS["Q"].L + SUPPORTS["Q"].U) / 2.0,
        "scopa": (SUPPORTS["S"].L + SUPPORTS["S"].U) / 2.0,
    }

    def mean_from_anchor(col: str, midv: float) -> np.ndarray:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(x), x, midv)

    mu_u = mean_from_anchor("y_anchor_updrs3", mid["updrs3"])
    mu_q = mean_from_anchor("y_anchor_moca", mid["moca"])
    mu_s = mean_from_anchor("y_anchor_scopa", mid["scopa"])

    df["pred_mean_updrs3"] = _clip(mu_u, SUPPORTS["U"].L, SUPPORTS["U"].U)
    df["pred_p05_updrs3"] = _clip(mu_u - half_width["updrs3"], SUPPORTS["U"].L, SUPPORTS["U"].U)
    df["pred_p95_updrs3"] = _clip(mu_u + half_width["updrs3"], SUPPORTS["U"].L, SUPPORTS["U"].U)

    df["pred_mean_moca"] = _clip(mu_q, SUPPORTS["Q"].L, SUPPORTS["Q"].U)
    df["pred_p05_moca"] = _clip(mu_q - half_width["moca"], SUPPORTS["Q"].L, SUPPORTS["Q"].U)
    df["pred_p95_moca"] = _clip(mu_q + half_width["moca"], SUPPORTS["Q"].L, SUPPORTS["Q"].U)

    df["pred_mean_scopa"] = _clip(mu_s, SUPPORTS["S"].L, SUPPORTS["S"].U)
    df["pred_p05_scopa"] = _clip(mu_s - half_width["scopa"], SUPPORTS["S"].L, SUPPORTS["S"].U)
    df["pred_p95_scopa"] = _clip(mu_s + half_width["scopa"], SUPPORTS["S"].L, SUPPORTS["S"].U)

    df["engine"] = "locf"
    return df
