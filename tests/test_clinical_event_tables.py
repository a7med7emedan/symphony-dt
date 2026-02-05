import pandas as pd

from symphony_dt.clinical.events import (
    build_clinical_event_pairs,
    build_clinical_event_rates_by_state,
    build_clinical_event_rates_by_trajectory,
    build_clinical_event_rates_by_trust,
    build_clinician_summary_risk_table,
    build_patient_panel,
    build_personalization_diagnostics,
)


def test_clinical_tables_smoke() -> None:
    risk = pd.DataFrame(
        {
            "patient_id": [1, 1],
            "anchor_visit_id": [10, 10],
            "target_visit_id": [11, 11],
            "horizon_center_days": [180.0, 180.0],
            "t_anchor_days": [0.0, 0.0],
            "t_target_days": [190.0, 190.0],
            "dt_days": [190.0, 190.0],
            "endpoint": ["updrs3", "moca"],
            "direction": [">=", "<="],
            "delta_threshold": [5.0, -2.0],
            "prob_event": [0.2, 0.3],
            "engine": ["locf", "locf"],
        }
    )
    hs = pd.DataFrame(
        {
            "patient_id": [1],
            "anchor_visit_id": [10],
            "target_visit_id": [11],
            "horizon_center_days": [180.0],
            "state_id": ["M0_C0_N0"],
            "state_engine": ["proxy_bins_v1"],
        }
    )
    traj = pd.DataFrame({"patient_id": [1], "trajectory_cluster": [0], "cluster_engine": ["proxy_tertiles_v1"]})
    baseline = pd.DataFrame({"patient_id": [1], "sex": ["M"], "age_baseline_years": [60.0]})
    clinic = pd.DataFrame({"patient_id": [1, 1], "visit_id": [10, 11], "t_days_since_baseline": [0.0, 190.0]})
    states_long = pd.DataFrame(
        {
            "patient_id": [1, 1],
            "visit_id": [10, 11],
            "t_days_since_baseline": [0.0, 190.0],
            "state_id": ["A", "B"],
            "state_engine": ["proxy_bins_v1", "proxy_bins_v1"],
            "xM_proxy": [0.1, 0.2],
            "xC_proxy": [0.2, 0.3],
            "xN_proxy": [0.3, 0.4],
        }
    )
    transitions_long = pd.DataFrame({"patient_id": [1], "from_state_id": ["A"], "to_state_id": ["B"], "state_engine": ["proxy_bins_v1"]})

    rs = build_clinical_event_rates_by_state(risk, hs)
    assert len(rs) >= 1
    rt = build_clinical_event_rates_by_trajectory(risk, traj)
    assert len(rt) >= 1
    rtr = build_clinical_event_rates_by_trust(risk)
    assert len(rtr) >= 1

    pairs = build_clinical_event_pairs(risk)
    assert isinstance(pairs, pd.DataFrame)

    clinician = build_clinician_summary_risk_table(risk, hs, traj)
    assert len(clinician) >= 1

    panel = build_patient_panel(baseline, clinic, traj, states_long, risk)
    assert len(panel) == 1

    diag = build_personalization_diagnostics(
        baseline=baseline,
        clinic=clinic,
        pred_one_step=pd.DataFrame(),
        pred_horizon_long=pd.DataFrame(),
        risk=risk,
        states_long=states_long,
        transitions_long=transitions_long,
        trajectory_clusters=traj,
    )
    assert len(diag) >= 1
