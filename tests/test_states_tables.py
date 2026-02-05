import pandas as pd

from symphony_dt.state.states import build_state_occupancy, build_states_long


def test_states_long_and_occupancy() -> None:
    hs = pd.DataFrame(
        {
            "patient_id": [1, 1, 2],
            "anchor_visit_id": [10, 10, 20],
            "target_visit_id": [11, 12, 21],
            "horizon_center_days": [180.0, 365.0, 180.0],
            "t_anchor_days": [0.0, 0.0, 0.0],
            "t_target_days": [190.0, 370.0, 190.0],
            "dt_days": [190.0, 370.0, 190.0],
            "xM_proxy": [0.1, 0.1, 0.2],
            "xC_proxy": [0.2, 0.2, 0.3],
            "xN_proxy": [0.3, 0.3, 0.4],
            "state_id": ["M0_C0_N0", "M0_C0_N0", "M0_C0_N1"],
            "engine": ["locf", "locf", "locf"],
            "state_engine": ["proxy_bins_v1", "proxy_bins_v1", "proxy_bins_v1"],
        }
    )
    sl = build_states_long(hs)
    assert len(sl) == 2
    occ = build_state_occupancy(sl)
    assert {"state_id", "n_patients", "frac_patients"}.issubset(occ.columns)
