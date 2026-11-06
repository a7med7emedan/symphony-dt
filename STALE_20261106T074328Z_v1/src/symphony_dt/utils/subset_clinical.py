from __future__ import annotations

from typing import List

import pandas as pd


def select_demo_patient_ids_clinical_best(clinic: pd.DataFrame, n_patients: int) -> List[int]:
    required_cols = {
        "patient_id",
        "t_days_since_baseline",
        "updrs3_total",
        "moca_total",
        "scopa_aut_total",
        "miss_updrs3",
        "miss_moca",
        "miss_scopa",
    }
    missing = sorted([c for c in required_cols if c not in clinic.columns])
    if missing:
        raise ValueError(f"clinic missing columns needed for clinical_best selection: {missing}")

    if clinic.empty:
        return []

    def patient_score(df: pd.DataFrame) -> float:
        df = df.sort_values("t_days_since_baseline")
        n_visits = int(len(df))
        tmax = float(df["t_days_since_baseline"].max())
        anchor = df.iloc[0]
        miss_anchor = bool(anchor["miss_updrs3"]) or bool(anchor["miss_moca"]) or bool(anchor["miss_scopa"])
        if miss_anchor:
            return -1e12
        miss_rate = float(
            (df["miss_updrs3"].astype(int) + df["miss_moca"].astype(int) + df["miss_scopa"].astype(int)).mean() / 3.0
        )
        times = df["t_days_since_baseline"].to_numpy(dtype=float)
        horizons = [180.0, 365.0, 730.0]
        has_any_h = sum(int((abs(times - h) <= 30.0).any()) for h in horizons)
        return (10.0 * n_visits) + (0.01 * tmax) + (5.0 * has_any_h) - (50.0 * miss_rate)

    rows = []
    for pid, df in clinic.groupby("patient_id", sort=True):
        if len(df) < 2:
            continue
        s = patient_score(df)
        if s <= -1e11:
            continue
        rows.append((int(pid), float(s)))

    rows.sort(key=lambda x: (-x[1], x[0]))
    return [pid for pid, _ in rows[:n_patients]]
