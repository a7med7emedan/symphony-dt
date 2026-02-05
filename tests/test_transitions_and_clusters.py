import pandas as pd

from symphony_dt.trajectory.clusters import build_trajectory_clusters
from symphony_dt.trajectory.transitions import build_transitions_edge_list, build_transitions_long


def test_transitions_edge_and_clusters() -> None:
    states_long = pd.DataFrame(
        {
            "patient_id": [1, 1, 1, 2, 2],
            "visit_id": [10, 11, 12, 20, 21],
            "t_days_since_baseline": [0.0, 100.0, 200.0, 0.0, 120.0],
            "state_id": ["A", "B", "B", "A", "A"],
            "state_engine": ["proxy_bins_v1"] * 5,
            "xM_proxy": [0.1, 0.2, 0.2, 0.1, 0.1],
            "xC_proxy": [0.2, 0.3, 0.3, 0.2, 0.2],
            "xN_proxy": [0.3, 0.4, 0.4, 0.3, 0.3],
        }
    )
    tl = build_transitions_long(states_long)
    assert len(tl) == 3
    el = build_transitions_edge_list(tl)
    assert "n_transitions" in el.columns

    tc = build_trajectory_clusters(states_long)
    assert "trajectory_cluster" in tc.columns
