from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def build_states_long(horizon_states: pd.DataFrame) -> pd.DataFrame:
    """
    Produce states_long.csv from forecast_horizon_states.csv.

    Each row corresponds to a (patient_id, anchor_visit_id) state assignment with proxy x values.

    Assumptions (explicit):
      - Using proxy state assignments from Module 20 (state_engine column).
      - Later modules will replace this with notebook-defined latent-state summaries.

    Output columns (stable):
      patient_id, visit_id, t_days_since_baseline, state_id, state_engine,
      xM_proxy, xC_proxy, xN_proxy
    """
    required = [
        "patient_id",
        "anchor_visit_id",
        "t_anchor_days",
        "state_id",
        "state_engine",
        "xM_proxy",
        "xC_proxy",
        "xN_proxy",
    ]
    missing = [c for c in required if c not in horizon_states.columns]
    if missing:
        raise ValueError(f"horizon_states missing required columns: {missing}")

    df = horizon_states.copy()

    # Reduce to unique (patient, anchor_visit) using first occurrence.
    df = df.sort_values(["patient_id", "anchor_visit_id"]).drop_duplicates(
        subset=["patient_id", "anchor_visit_id"], keep="first"
    )

    out = pd.DataFrame(
        {
            "patient_id": df["patient_id"].astype(int),
            "visit_id": df["anchor_visit_id"].astype(int),
            "t_days_since_baseline": pd.to_numeric(df["t_anchor_days"], errors="coerce").astype(float),
            "state_id": df["state_id"].astype(str),
            "state_engine": df["state_engine"].astype(str),
            "xM_proxy": pd.to_numeric(df["xM_proxy"], errors="coerce").astype(float),
            "xC_proxy": pd.to_numeric(df["xC_proxy"], errors="coerce").astype(float),
            "xN_proxy": pd.to_numeric(df["xN_proxy"], errors="coerce").astype(float),
        }
    )
    return out


def build_state_occupancy(states_long: pd.DataFrame) -> pd.DataFrame:
    """
    Produce state_occupancy.csv.

    Defines occupancy as fraction of patients ever observed in each state across anchor visits.

    Output:
      state_id, state_engine, n_patients, n_total_patients, frac_patients
    """
    required = ["patient_id", "state_id", "state_engine"]
    missing = [c for c in required if c not in states_long.columns]
    if missing:
        raise ValueError(f"states_long missing required columns: {missing}")

    df = states_long.copy()
    n_total = int(df["patient_id"].nunique())
    g = df.groupby(["state_id", "state_engine"], dropna=False)["patient_id"].nunique().reset_index()
    g = g.rename(columns={"patient_id": "n_patients"})
    g["n_total_patients"] = n_total
    g["frac_patients"] = g["n_patients"] / float(n_total) if n_total > 0 else 0.0
    return g.sort_values(["state_engine", "state_id"]).reset_index(drop=True)
