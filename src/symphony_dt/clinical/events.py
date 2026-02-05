from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_EVENT_DEFS: List[Tuple[str, str, float]] = [
    ("updrs3", ">=", 5.0),
    ("updrs3", ">=", 10.0),
    ("moca", "<=", -2.0),
    ("moca", "<=", -3.0),
]


def _event_id(endpoint: str, direction: str, thr: float) -> str:
    thr_s = str(int(thr)) if float(thr).is_integer() else str(thr)
    return f"{endpoint}{direction}{thr_s}"


def build_clinical_event_rates_by_state(risk: pd.DataFrame, horizon_states: pd.DataFrame) -> pd.DataFrame:
    key = ["patient_id", "anchor_visit_id", "target_visit_id", "horizon_center_days"]
    df = risk.merge(horizon_states, on=key, how="left")
    df["prob_event"] = pd.to_numeric(df["prob_event"], errors="coerce")
    gcols = ["state_engine", "state_id", "endpoint", "direction", "delta_threshold", "horizon_center_days"]
    out = (
        df.groupby(gcols, dropna=False)
        .agg(n_instances=("prob_event", "size"), n_patients=("patient_id", "nunique"), prob_event_mean=("prob_event", "mean"))
        .reset_index()
    )
    out["engine"] = df["engine"].iloc[0] if "engine" in df.columns and len(df) else ""
    return out.sort_values(gcols).reset_index(drop=True)


def build_clinical_event_rates_by_trajectory(risk: pd.DataFrame, trajectory_clusters: pd.DataFrame) -> pd.DataFrame:
    df = risk.merge(trajectory_clusters[["patient_id", "trajectory_cluster", "cluster_engine"]], on="patient_id", how="left")
    df["prob_event"] = pd.to_numeric(df["prob_event"], errors="coerce")
    gcols = ["cluster_engine", "trajectory_cluster", "endpoint", "direction", "delta_threshold", "horizon_center_days"]
    out = (
        df.groupby(gcols, dropna=False)
        .agg(n_instances=("prob_event", "size"), n_patients=("patient_id", "nunique"), prob_event_mean=("prob_event", "mean"))
        .reset_index()
    )
    out["engine"] = df["engine"].iloc[0] if "engine" in df.columns and len(df) else ""
    return out.sort_values(gcols).reset_index(drop=True)


def build_clinical_event_rates_by_trust(risk: pd.DataFrame) -> pd.DataFrame:
    df = risk.copy()
    df["trust_label"] = "allow_all_baseline"
    df["prob_event"] = pd.to_numeric(df["prob_event"], errors="coerce")
    gcols = ["trust_label", "endpoint", "direction", "delta_threshold", "horizon_center_days"]
    out = (
        df.groupby(gcols, dropna=False)
        .agg(n_instances=("prob_event", "size"), n_patients=("patient_id", "nunique"), prob_event_mean=("prob_event", "mean"))
        .reset_index()
    )
    out["engine"] = df["engine"].iloc[0] if "engine" in df.columns and len(df) else ""
    out["trust_engine"] = "baseline_no_gating"
    return out.sort_values(gcols).reset_index(drop=True)


def build_clinical_event_pairs(risk: pd.DataFrame) -> pd.DataFrame:
    df = risk.copy()
    df["event_id"] = [
        _event_id(e, d, float(t)) for e, d, t in zip(
            df["endpoint"].astype(str),
            df["direction"].astype(str),
            pd.to_numeric(df["delta_threshold"], errors="coerce"),
        )
    ]
    key = ["patient_id", "anchor_visit_id", "target_visit_id", "horizon_center_days"]
    wide = df.pivot_table(index=key, columns="event_id", values="prob_event", aggfunc="mean").reset_index()
    events = sorted([_event_id(*x) for x in _EVENT_DEFS])
    present = [e for e in events if e in wide.columns]
    rows: List[Dict[str, object]] = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            pa = pd.to_numeric(wide[a], errors="coerce").to_numpy(dtype=float)
            pb = pd.to_numeric(wide[b], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(pa) & np.isfinite(pb)
            if not m.any():
                continue
            rows.append(
                {
                    "event_a": a,
                    "event_b": b,
                    "n_instances": int(m.sum()),
                    "prob_a_mean": float(np.mean(pa[m])),
                    "prob_b_mean": float(np.mean(pb[m])),
                    "prob_joint_mean_est": float(np.mean(pa[m] * pb[m])),
                }
            )
    return pd.DataFrame(rows).sort_values(["event_a", "event_b"]).reset_index(drop=True)


def build_personalization_diagnostics(
    *,
    baseline: pd.DataFrame,
    clinic: pd.DataFrame,
    pred_one_step: pd.DataFrame,
    pred_horizon_long: pd.DataFrame,
    risk: pd.DataFrame,
    states_long: pd.DataFrame,
    transitions_long: pd.DataFrame,
    trajectory_clusters: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"metric": "n_patients_baseline", "value": int(baseline["patient_id"].nunique()) if "patient_id" in baseline.columns else 0},
        {"metric": "n_patients_clinic", "value": int(clinic["patient_id"].nunique()) if "patient_id" in clinic.columns else 0},
        {"metric": "n_rows_forecast_one_step", "value": int(len(pred_one_step))},
        {"metric": "n_rows_forecast_horizon_long", "value": int(len(pred_horizon_long))},
        {"metric": "n_rows_forecast_risk_curves", "value": int(len(risk))},
        {"metric": "n_rows_states_long", "value": int(len(states_long))},
        {"metric": "n_rows_transitions_long", "value": int(len(transitions_long))},
        {"metric": "n_rows_trajectory_clusters", "value": int(len(trajectory_clusters))},
    ]
    return pd.DataFrame(rows)


def build_clinician_summary_risk_table(risk: pd.DataFrame, horizon_states: pd.DataFrame, trajectory_clusters: pd.DataFrame) -> pd.DataFrame:
    key = ["patient_id", "anchor_visit_id", "target_visit_id", "horizon_center_days"]
    df = risk.merge(horizon_states, on=key, how="left").merge(trajectory_clusters, on="patient_id", how="left")
    df["prob_event"] = pd.to_numeric(df["prob_event"], errors="coerce")
    gcols = ["state_engine","state_id","cluster_engine","trajectory_cluster","horizon_center_days","endpoint","direction","delta_threshold"]
    out = df.groupby(gcols, dropna=False).agg(
        n_instances=("prob_event","size"),
        n_patients=("patient_id","nunique"),
        prob_event_mean=("prob_event","mean"),
    ).reset_index()
    out["engine"] = df["engine"].iloc[0] if "engine" in df.columns and len(df) else ""
    return out.sort_values(gcols).reset_index(drop=True)


def build_patient_panel(baseline: pd.DataFrame, clinic: pd.DataFrame, trajectory_clusters: pd.DataFrame, states_long: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    b = baseline[["patient_id","sex","age_baseline_years"]].copy()
    b["patient_id"] = b["patient_id"].astype(int)
    n_visits = clinic.groupby("patient_id").size().rename("n_visits").reset_index()
    latest = states_long.sort_values(["patient_id","t_days_since_baseline"]).groupby("patient_id").tail(1)[["patient_id","state_id","state_engine"]]
    latest = latest.rename(columns={"state_id":"latest_state_id"})
    tc = trajectory_clusters[["patient_id","trajectory_cluster","cluster_engine"]].copy()
    r = risk.copy()
    r["event_id"] = [
        _event_id(e, d, float(t)) for e, d, t in zip(
            r["endpoint"].astype(str),
            r["direction"].astype(str),
            pd.to_numeric(r["delta_threshold"], errors="coerce"),
        )
    ]
    piv = r.groupby(["patient_id","horizon_center_days","event_id"], dropna=False)["prob_event"].mean().reset_index()
    piv["col"] = piv["event_id"] + "_h" + piv["horizon_center_days"].astype(int).astype(str)
    wide = piv.pivot_table(index="patient_id", columns="col", values="prob_event", aggfunc="mean").reset_index()
    out = b.merge(n_visits, on="patient_id", how="left").merge(latest, on="patient_id", how="left").merge(tc, on="patient_id", how="left").merge(wide, on="patient_id", how="left")
    return out.sort_values("patient_id").reset_index(drop=True)
