import numpy as np
from symphony_dt.explain.stability import sign_probability, should_report_explanation


def test_sign_probability_and_gate() -> None:
    s = np.ones(100)
    assert sign_probability(s) == 1.0
    assert should_report_explanation(s)
