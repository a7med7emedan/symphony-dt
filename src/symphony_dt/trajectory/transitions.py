from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def build_transitions_long(states_long: pd.DataFrame) -> pd.DataFrame:
    """
    Produce transitions_long.csv.

    Each row corresponds to a transition between consecutive anchor visits for a patient:
      (state_t -> state_t1) with dt_days.

    Uses states_long ordered by t_days_since_baseline.

    Output columns:
      patient_id, from_visit_id, to_visit_id, t_from, t_to, dt_days,
      from_state_id, to_state_id, state_engine
    """
    required = ["patient_id", "visit_id", "t_days_since_baseline", "state_id", "state_engine"]
    missing = [c for c in required if c not in states_long.columns]
    if missing:
        raise ValueError(f"states_long missing required columns: {missing}")

    df = states_long.sort_values(["patient_id", "t_days_since_baseline", "visit_id"]).copy()

    rows: List[Dict[str, object]] = []
    for pid, g in df.groupby("patient_id", sort=True):
        g = g.sort_values(["t_days_since_baseline", "visit_id"]).reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(len(g) - 1):
            a = g.loc[i]
            b = g.loc[i + 1]
            rows.append(
                {
                    "patient_id": int(pid),
                    "from_visit_id": int(a["visit_id"]),
                    "to_visit_id": int(b["visit_id"]),
                    "t_from": float(a["t_days_since_baseline"]),
                    "t_to": float(b["t_days_since_baseline"]),
                    "dt_days": float(b["t_days_since_baseline"] - a["t_days_since_baseline"]),
                    "from_state_id": str(a["state_id"]),
                    "to_state_id": str(b["state_id"]),
                    "state_engine": str(a["state_engine"]),
                }
            )
    return pd.DataFrame(rows)


def build_transitions_edge_list(transitions_long: pd.DataFrame) -> pd.DataFrame:
    """
    Produce transitions_edge_list.csv as aggregated counts and rates by edge.

    Output:
      state_engine, from_state_id, to_state_id, n_transitions, n_patients, frac_transitions
    """
    required = ["patient_id", "from_state_id", "to_state_id", "state_engine"]
    missing = [c for c in required if c not in transitions_long.columns]
    if missing:
        raise ValueError(f"transitions_long missing required columns: {missing}")

    df = transitions_long.copy()
    n_total = int(len(df))
    g = (
        df.groupby(["state_engine", "from_state_id", "to_state_id"], dropna=False)
        .agg(n_transitions=("patient_id", "size"), n_patients=("patient_id", "nunique"))
        .reset_index()
    )
    g["frac_transitions"] = g["n_transitions"] / float(n_total) if n_total > 0 else 0.0
    return g.sort_values(["state_engine", "from_state_id", "to_state_id"]).reset_index(drop=True)
