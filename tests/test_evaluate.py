"""Scoring metrics, the LMM, events, bootstrap and the pre-registration freeze."""

from __future__ import annotations

import numpy as np
import pandas as pd
import properscoring as ps
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from symphony_dt.errors import UnfrozenSpecificationError
from symphony_dt.evaluate import (
    DeviationRecord,
    ScoringSpec,
    auc_with_ties,
    bootstrap_mean,
    brier,
    brier_skill,
    central_interval,
    cluster_bootstrap_weights,
    crps_ensemble,
    event_indicator,
    event_probability,
    fit_lmm_em,
    freeze_spec,
    interval_coverage,
    lmm_condition,
    load_frozen_spec,
    mae_median,
    pool_folds,
    require_frozen,
    score_fold,
)
from tests.golden import cases

finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


@settings(max_examples=150, deadline=None)
@given(st.lists(finite, min_size=1, max_size=60), finite)
def test_crps_equals_properscoring(samples, y):
    assert crps_ensemble(samples, y) == pytest.approx(
        float(ps.crps_ensemble(y, np.array(samples))), rel=1e-9, abs=1e-9
    )


@settings(max_examples=80, deadline=None)
@given(st.lists(finite, min_size=2, max_size=60), finite)
def test_crps_properties(samples, y):
    c = crps_ensemble(samples, y)
    assert c >= -1e-9
    assert crps_ensemble([v + 5 for v in samples], y + 5) == pytest.approx(c, abs=1e-6)


def test_intervals_and_mae():
    x = np.arange(101.0)
    assert central_interval(x, 0.5) == (25.0, 75.0)
    assert interval_coverage(x, 25.0, 0.5) == 1.0
    assert interval_coverage(x, 24.9, 0.5) == 0.0
    assert mae_median(x, 40) == 10.0
    with pytest.raises(ValueError):
        central_interval(x, 1.0)


def test_events_and_brier():
    assert event_indicator(15, 10, 1) == 1.0
    assert event_indicator(14.9, 10, 1) == 0.0
    assert event_indicator(26, 28, 2) == 1.0
    assert event_probability([15.4, 14.4, 16, 9], 10, 1) == 0.5
    assert event_probability([26, 27, 25.6, 30], 28, 2) == 0.5
    assert brier([1, 0], [1, 1]) == 0.5
    assert brier_skill(0.1, 0.2) == pytest.approx(0.5)


def test_auc_with_ties():
    assert auc_with_ties([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert auc_with_ties([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert np.isnan(auc_with_ties([0.1, 0.2], [1, 1]))
    assert auc_with_ties([0.1, 0.9], [0, 1], w=[2.0, 1.0]) == 1.0


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.floats(0, 1), st.integers(0, 1)), min_size=2, max_size=40))
def test_auc_matches_mann_whitney(pairs):
    p = np.array([a for a, _ in pairs])
    e = np.array([b for _, b in pairs], float)
    if e.min() == e.max():
        assert np.isnan(auc_with_ties(p, e))
        return
    pos, neg = p[e == 1], p[e == 0]
    ref = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert auc_with_ties(p, e) == pytest.approx(ref)


def test_lmm_recovers_parameters():
    rng = np.random.default_rng(0)
    groups = []
    for _ in range(300):
        t = np.sort(rng.uniform(0, 5, size=5))
        u = rng.normal(0, [2.0, 0.5])
        groups.append((t, 10 + u[0] + (1.5 + u[1]) * t + rng.normal(0, 1.0, size=5)))
    fit = fit_lmm_em(groups)
    assert fit.beta == pytest.approx([10, 1.5], abs=0.4)
    assert np.sqrt(fit.s2) == pytest.approx(1.0, abs=0.15)
    mean, cov = lmm_condition(fit, np.array([]), np.array([]))
    assert mean.tolist() == [0, 0] and np.allclose(cov, fit.G)
    mean, _ = lmm_condition(fit, groups[0][0], groups[0][1])
    assert mean.shape == (2,)
    assert set(fit.as_dict()) == {"beta", "G", "s2", "iters"}


def test_bootstrap():
    W = cluster_bootstrap_weights(5, B=200, seed=1)
    assert W.shape == (200, 5) and np.all(W.sum(axis=1) == 5)
    est = bootstrap_mean(np.ones(10), np.repeat(np.arange(5), 2), W)
    assert est == [1.0, 1.0, 1.0]


def test_spec_freeze_cycle(tmp_path):
    spec = ScoringSpec.default()
    frozen = freeze_spec(spec, tmp_path / "spec.json")
    loaded, frozen2 = load_frozen_spec(tmp_path / "spec.json")
    assert loaded == spec and frozen2.sha256 == frozen.sha256
    assert require_frozen(spec, frozen)["deviation"] is None
    with pytest.raises(UnfrozenSpecificationError, match="frozen specification"):
        require_frozen(spec, None)
    changed = spec.model_copy(update={"min_bin": 20})
    with pytest.raises(UnfrozenSpecificationError, match="DeviationRecord"):
        require_frozen(changed, frozen)
    rec = require_frozen(changed, frozen, DeviationRecord("smaller bins", "A. Author", {"min_bin": 20}))
    assert rec["deviation"]["reason"] == "smaller bins"
    with pytest.raises(ValueError):
        DeviationRecord("", "x")
    raw = (tmp_path / "spec.json").read_text().replace('"min_bin": 30', '"min_bin": 31')
    (tmp_path / "bad.json").write_text(raw)
    with pytest.raises(UnfrozenSpecificationError, match="does not match"):
        load_frozen_spec(tmp_path / "bad.json")


def test_score_fold_refuses_unfrozen_and_bad_input(frozen_default):
    spec, frozen = frozen_default
    sp, tests, age, draws = cases.fold_case(1)
    with pytest.raises(UnfrozenSpecificationError):
        score_fold(sp.fit.data, sp.targets.data, sp.meta, draws, tests, spec=spec, frozen=None)
    with pytest.raises(ValueError, match="twin_draws"):
        score_fold(sp.fit.data, sp.targets.data, sp.meta, draws[:, :-1], tests, spec=spec, frozen=frozen)
    with pytest.raises(ValueError, match="non-test"):
        score_fold(sp.fit.data, sp.targets.data, sp.meta, draws, tests[1:], spec=spec, frozen=frozen)
    bad = draws.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        score_fold(sp.fit.data, sp.targets.data, sp.meta, bad, tests, spec=spec, frozen=frozen)
    with pytest.raises(ValueError, match="does not match"):
        score_fold(sp.fit.data, sp.targets.data, sp.meta.iloc[1:], draws, tests, spec=spec, frozen=frozen)
    with pytest.raises(ValueError, match="masking"):
        score_fold(_unmasked(sp), sp.targets.data, sp.meta, draws, tests, spec=spec, frozen=frozen)


def _unmasked(sp):
    """Put the targets back into the fit data, which the scorer must refuse as leakage."""
    d = dict(sp.fit.data)
    tgt = sp.targets.data
    merged = {**d, "n_obs": d["n_obs"] + tgt["n_obs"]}
    obs = {k: [] for k in ("obs_node", "obs_k", "y", "obs_ledd")}
    starts, counts = [], []
    for i in range(d["N"]):
        starts.append(len(obs["y"]))
        for src in (d, tgt):
            a, c = src["obs_start"][i], src["obs_count"][i]
            for k in obs:
                obs[k].extend(src[k][a : a + c])
        counts.append(len(obs["y"]) - starts[-1])
    return {**merged, **obs, "obs_start": starts, "obs_count": counts}


def test_pool_folds_rules(frozen_default):
    spec, frozen = frozen_default
    sp, tests, age, draws = cases.fold_case(1)
    fs = score_fold(sp.fit.data, sp.targets.data, sp.meta, draws, tests, spec=spec, frozen=frozen, age=age)
    rec = {"converged": True, "max_rhat": 1.0, "min_ess_bulk": 500.0}
    pooled = pool_folds(
        [(fs.target_scores, rec), (None, None), (fs.target_scores, {**rec, "converged": False})],
        spec=spec,
        frozen=frozen,
    )
    assert pooled.scores["folds"][2] == "missing"
    assert pooled.scores["folds"][3].startswith("not converged")
    assert len(pooled.sha256()) == 64 and pooled.lines
    dup = fs.target_scores.assign(fold=2)
    with pytest.raises(ValueError, match="more than one fold"):
        pool_folds([(fs.target_scores, rec), (dup, rec)], spec=spec, frozen=frozen)
    with pytest.raises(ValueError, match="no scored fold"):
        pool_folds([(None, None)], spec=spec, frozen=frozen)


def test_fold_scores_write(tmp_path, frozen_default):
    spec, frozen = frozen_default
    sp, tests, age, draws = cases.fold_case(2)
    fs = score_fold(sp.fit.data, sp.targets.data, sp.meta, draws, tests, spec=spec, frozen=frozen)
    out = fs.write(tmp_path / "s")
    assert (out / "scores.json").exists() and (out / "target_scores.csv").exists()
    assert pd.read_csv(out / "forecast_quantiles.csv").shape[0] == draws.shape[1]
    assert fs.specification["frozen_sha256"] == frozen.sha256
