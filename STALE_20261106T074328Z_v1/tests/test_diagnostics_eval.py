from symphony_dt.inference.diagnostics import ConvergenceCriteria, evaluate_convergence


def test_convergence_passes() -> None:
    diag = {
        "divergences": 0,
        "rhat": {"a": 1.001, "b": 1.01},
        "ess_bulk": {"a": 500, "b": 400},
    }
    rep = evaluate_convergence(diag, criteria=ConvergenceCriteria())
    assert rep.passed


def test_convergence_fails_rhat_ess_div() -> None:
    diag = {
        "divergences": 2,
        "rhat": {"a": 1.02},
        "ess_bulk": {"a": 100},
    }
    rep = evaluate_convergence(diag, criteria=ConvergenceCriteria())
    assert not rep.passed
    assert any("divergences" in f for f in rep.failures)
    assert any("rhat" in f for f in rep.failures)
    assert any("ess_bulk" in f for f in rep.failures)
