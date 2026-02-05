from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def build_trajectory_clusters(states_long: pd.DataFrame) -> pd.DataFrame:
    """
    Produce trajectory_clusters.csv.

    Proxy clustering (explicit, will be replaced by notebook-parity definition):
      - represent each patient by mean of proxy latent x values across their visits
      - assign cluster by tertiles of xM_proxy + xC_proxy + xN_proxy (severity score)

    Output:
      patient_id, trajectory_cluster, severity_score_mean, n_visits, state_engine, cluster_engine
    """
    required = ["patient_id", "xM_proxy", "xC_proxy", "xN_proxy", "state_engine"]
    missing = [c for c in required if c not in states_long.columns]
    if missing:
        raise ValueError(f"states_long missing required columns for clustering: {missing}")

    df = states_long.copy()
    agg = (
        df.groupby(["patient_id", "state_engine"], dropna=False)
        .agg(
            xM_mean=("xM_proxy", "mean"),
            xC_mean=("xC_proxy", "mean"),
            xN_mean=("xN_proxy", "mean"),
            n_visits=("visit_id", "size"),
        )
        .reset_index()
    )
    score = agg["xM_mean"] + agg["xC_mean"] + agg["xN_mean"]
    agg["severity_score_mean"] = score

    # tertile clustering
    try:
        agg["trajectory_cluster"] = pd.qcut(score, q=3, labels=[0, 1, 2]).astype("Int64")
    except Exception:  # noqa: BLE001
        r = score.rank(method="average", na_option="keep")
        agg["trajectory_cluster"] = pd.qcut(r, q=3, labels=[0, 1, 2]).astype("Int64")

    agg["cluster_engine"] = "proxy_tertiles_v1"
    out = agg[
        ["patient_id", "trajectory_cluster", "severity_score_mean", "n_visits", "state_engine", "cluster_engine"]
    ].copy()
    out["patient_id"] = out["patient_id"].astype(int)
    return out.sort_values(["state_engine", "trajectory_cluster", "patient_id"]).reset_index(drop=True)
