"""Decision curve analysis: net benefit by hand, reference strategies, known answers and outputs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from symphony_dt.evaluate import decision_curve, net_benefit, score_fold
from tests.golden import cases


@pytest.fixture(scope="module")
def fold(frozen_default):
    spec, frozen = frozen_default
    sp, tests, age, draws = cases.fold_case(1)
    fs = score_fold(sp.fit.data, sp.targets.data, sp.meta, draws, tests, spec=spec, frozen=frozen, age=age)
    return fs.target_scores


def _table(p_tw, ev, *, k=1, p_lo=None, p_lm=None, per_participant=1):
    n = len(ev)
    return pd.DataFrame(
        {
            "i": np.arange(n) // per_participant,
            "k": k,
            "ev": np.asarray(ev, float),
            "p_tw": np.asarray(p_tw, float),
            "p_lo": np.asarray(p_tw if p_lo is None else p_lo, float),
            "p_lm": np.asarray(p_tw if p_lm is None else p_lm, float),
        }
    )


def test_net_benefit_by_hand():
    p, e = [0.9, 0.2, 0.6, 0.1], [1, 0, 0, 1]
    # pt 0.5: flags rows 0 and 2, one true and one false flag, weight 1.
    assert net_benefit(p, e, 0.5) == pytest.approx(0.25 - 0.25 * 1.0)
    # pt 0.25: the same flags, weight 1/3.
    assert net_benefit(p, e, 0.25) == pytest.approx(0.25 - 0.25 / 3)
    # pt 0.6: a probability equal to the threshold is flagged.
    assert net_benefit(p, e, 0.6) == pytest.approx(0.25 - 0.25 * 1.5)
    with pytest.raises(ValueError, match="threshold"):
        net_benefit(p, e, 1.0)
    with pytest.raises(ValueError, match="equal length"):
        net_benefit([0.1], [1, 0], 0.2)


def test_reference_strategies_and_a_perfect_forecast():
    rng = np.random.default_rng(3)
    ev = rng.binomial(1, 0.3, 400)
    res = decision_curve(_table(ev, ev), thresholds=(0.1, 0.2, 0.4), min_cell=1, bootstrap_replicates=100)
    t = res.table.set_index(["strategy", "threshold"])
    rate = ev.mean()
    for pt in (0.1, 0.2, 0.4):
        assert t.loc[("all", pt), "net_benefit"] == pytest.approx(rate - (1 - rate) * pt / (1 - pt), abs=1e-4)
        assert t.loc[("none", pt), "net_benefit"] == 0.0
        # A forecast that equals the outcome gains the full event rate at every threshold.
        assert t.loc[("twin", pt), "net_benefit"] == pytest.approx(rate, abs=1e-4)
        assert t.loc[("twin", pt), "net_benefit_lo"] <= rate <= t.loc[("twin", pt), "net_benefit_hi"]


def test_calibrated_forecast_is_not_beaten_by_the_references():
    # A calibrated forecast has expected net benefit at least that of flagging all or none.
    rng = np.random.default_rng(11)
    p = rng.beta(2, 5, 20000)
    ev = rng.binomial(1, p)
    inverted = 1 - p
    res = decision_curve(
        _table(p, ev, p_lo=inverted), thresholds=np.arange(0.05, 0.6, 0.05), min_cell=1, bootstrap_replicates=50
    )
    t = res.table.pivot(index="threshold", columns="strategy", values="net_benefit")
    best_reference = np.maximum(t["all"], t["none"])
    assert (t["twin"] >= best_reference - 0.005).all()
    assert (t["locf"] < t["twin"]).all()
    assert (t["locf"] < best_reference).any()


def test_matches_net_benefit_on_the_scored_fold(fold):
    res = decision_curve(fold, thresholds=(0.1, 0.3), min_cell=1, bootstrap_replicates=200)
    assert set(res.table["strategy"]) == {"twin", "locf", "lmm", "all", "none"}
    for _, row in res.table.iterrows():
        sub = fold[fold["k"] == row["k"]].dropna(subset=["ev", "p_tw", "p_lo", "p_lm"])
        assert row["n_targets"] == len(sub)
        assert row["n_participants"] == sub["i"].nunique()
        col = {"twin": "p_tw", "locf": "p_lo", "lmm": "p_lm"}.get(row["strategy"])
        if col is not None:
            expected = net_benefit(sub[col], sub["ev"], row["threshold"])
            assert row["net_benefit"] == pytest.approx(expected, abs=1e-4)
            assert row["net_benefit_lo"] <= row["net_benefit"] + 1e-4
            assert row["net_benefit"] <= row["net_benefit_hi"] + 1e-4


def test_small_endpoints_are_suppressed():
    ev = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    res = decision_curve(_table(np.full(12, 0.3), ev), thresholds=(0.2,), min_cell=10, bootstrap_replicates=50)
    assert res.table["suppressed"].all()
    assert res.table[["net_benefit", "n_targets", "n_participants", "event_rate"]].isna().all().all()
    assert "suppressed" in res.to_markdown(thresholds=(0.2,))


def test_bad_input():
    t = _table([0.2, 0.4], [0, 1])
    with pytest.raises(ValueError, match="unknown methods"):
        decision_curve(t, methods=("oracle",))
    with pytest.raises(ValueError, match="thresholds"):
        decision_curve(t, thresholds=(0.0, 0.5))
    with pytest.raises(ValueError, match="needs columns"):
        decision_curve(t.drop(columns="ev"))
    with pytest.raises(ValueError, match="min_cell"):
        decision_curve(t, min_cell=0)


def test_markdown_and_plot(fold):
    res = decision_curve(fold, min_cell=1, bootstrap_replicates=100)
    md = res.to_markdown()
    assert md.splitlines()[0] == "| endpoint | threshold | twin | locf | lmm | all | none |"
    assert "| updrs3 | 0.10 |" in md
    ax = res.plot(endpoint="moca")
    assert ax.get_xlabel() == "Threshold probability"
    assert {line.get_label() for line in ax.get_lines()} >= {"twin", "all", "none"}
