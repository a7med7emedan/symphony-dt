import numpy as np

from symphony_dt.personalization.conditioning import ConditionalRun, ConditioningMixture, ConditioningRunDiagnostics, filter_successful_runs
from symphony_dt.personalization.gating import TrustConfig, decide_trust_and_suppression


def _run(ok: bool, pid=1, n=0, ell=0):
    d = ConditioningRunDiagnostics(
        divergences=0 if ok else 1,
        accept_prob_mean=0.8,
        rhat_r_m=1.0,
        rhat_r_c=1.0,
        rhat_r_n=1.0,
    )
    xs = np.random.default_rng(0).uniform(0.2, 0.3, size=(50, 3))
    return ConditionalRun(patient_id=pid, visit_index=n, global_draw_index=ell, diagnostics=d, samples_x_tn=xs)


def test_rule_1_fewer_than_two_visits() -> None:
    runs = [_run(True)]
    ok, min_pass = filter_successful_runs(runs, l_cond=10)
    mix = ConditioningMixture(patient_id=1, visit_index=0, l_cond=10, successful_runs=ok)
    dec = decide_trust_and_suppression(
        n_visits_total=1,
        missing_at_anchor=(False, False, False),
        anchor_values=(10.0, 28.0, 5.0),
        mixture=mix,
        cfg=TrustConfig(tau_conf=0.5),
        min_required_success=min_pass,
    )
    assert not dec.allow_forecasts


def test_rule_4_insufficient_runs() -> None:
    runs = [_run(True, ell=0), _run(False, ell=1)]
    ok, min_pass = filter_successful_runs(runs, l_cond=50)  # min_pass = max(10,ceil(20))=20
    mix = ConditioningMixture(patient_id=1, visit_index=0, l_cond=50, successful_runs=ok)
    dec = decide_trust_and_suppression(
        n_visits_total=3,
        missing_at_anchor=(False, False, False),
        anchor_values=(10.0, 28.0, 5.0),
        mixture=mix,
        cfg=TrustConfig(tau_conf=0.5),
        min_required_success=min_pass,
    )
    assert not dec.allow_forecasts
    assert "rule_4" in dec.reasons


def test_rule_5_boundary_suppresses_events_not_forecasts() -> None:
    runs = [_run(True, ell=i) for i in range(30)]
    ok, min_pass = filter_successful_runs(runs, l_cond=50)
    mix = ConditioningMixture(patient_id=1, visit_index=0, l_cond=50, successful_runs=ok)
    dec = decide_trust_and_suppression(
        n_visits_total=3,
        missing_at_anchor=(False, False, False),
        anchor_values=(0.0, 28.0, 5.0),  # UPDRS at boundary L=0
        mixture=mix,
        cfg=TrustConfig(tau_conf=0.5),
        min_required_success=min_pass,
    )
    assert dec.allow_forecasts
    assert not dec.allow_threshold_events
    assert "rule_5_U" in dec.reasons
