"""Calibration by subgroup: counts, estimates, suppression, PIT tests and multiplicity."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from symphony_dt.calibration.ppc import pit_values
from symphony_dt.evaluate import age_band, crps_ensemble, score_fold, subgroup_calibration
from symphony_dt.evaluate.subgroups import _adjust
from tests.golden import cases


@pytest.fixture(scope="module")
def fold(frozen_default):
    spec, frozen = frozen_default
    sp, tests, age, draws = cases.fold_case(1)
    fs = score_fold(sp.fit.data, sp.targets.data, sp.meta, draws, tests, spec=spec, frozen=frozen, age=age)
    return fs.target_scores, draws


def test_age_band_labels():
    bands = age_band([40, 55, 64.9, 65, 80, np.nan])
    assert bands.tolist() == ["under 55", "55 to 64", "55 to 64", "65 to 74", "75 and over", "unknown"]
    with pytest.raises(ValueError, match="increase"):
        age_band([50], edges=(60, 60))


def test_overall_row_matches_the_target_table(fold):
    ts, draws = fold
    res = subgroup_calibration(ts, {}, twin_draws=draws, min_cell=1, bootstrap_replicates=200)
    overall = res.table[(res.table["grouping"] == "all") & (res.table["method"] == "twin")]
    for _, row in overall.iterrows():
        sub = ts[ts["k"] == row["k"]]
        assert row["n_targets"] == len(sub)
        assert row["n_participants"] == sub["i"].nunique()
        assert row["c95"] == pytest.approx(sub["tw_c95"].mean(), abs=1e-4)
        assert row["crps"] == pytest.approx(sub["tw_crps"].mean(), abs=1e-4)
        assert row["c95_lo"] <= row["c95"] <= row["c95_hi"]
    expected = pit_values(draws, ts["y"].to_numpy(float), seed=0)
    k1 = (ts["k"] == overall.iloc[0]["k"]).to_numpy()
    assert overall.iloc[0]["pit_mean"] == pytest.approx(expected[k1].mean(), abs=1e-4)


def test_levels_partition_the_cohort(fold):
    ts, draws = fold
    site = {i: f"site {i % 3}" for i in ts["i"].unique()}
    groups = {"sex": "sex", "cohort": "cohort", "site": site, "age band": age_band(ts["age"])}
    res = subgroup_calibration(ts, groups, twin_draws=draws, min_cell=1, bootstrap_replicates=100)
    tw = res.table[res.table["method"] == "twin"]
    total = tw[tw["grouping"] == "all"].set_index("k")["n_targets"]
    for name in groups:
        per = tw[tw["grouping"] == name].groupby("k")["n_targets"].sum()
        assert per.to_dict() == total.to_dict()
    assert set(tw.loc[tw["grouping"] == "site", "level"]) == {"site 0", "site 1", "site 2"}
    assert set(res.table["method"]) == {"twin", "locf", "lmm"}
    assert res.table.loc[res.table["method"] == "locf", "pit_ks_p"].isna().all()


def test_small_cells_are_suppressed(fold):
    ts, draws = fold
    res = subgroup_calibration(ts, {"sex": "sex"}, twin_draws=draws, min_cell=10**6, bootstrap_replicates=50)
    assert res.table["suppressed"].all()
    assert res.table[["n_participants", "n_targets", "c95", "crps", "pit_mean"]].isna().all().all()
    md = res.to_markdown()
    assert "suppressed" in md and f"<{10**6}" in md


def test_detects_a_miscalibrated_subgroup():
    rng = np.random.default_rng(3)
    n, per = 300, 3
    i = np.repeat(np.arange(n), per)
    group = np.where(i % 2 == 0, "calibrated", "too narrow")
    y = rng.normal(0, np.where(group == "calibrated", 1.0, 3.0))
    draws = rng.normal(0, 1, size=(400, y.size))
    lo95, hi95 = np.quantile(draws, [0.025, 0.975], axis=0)
    lo50, hi50 = np.quantile(draws, [0.25, 0.75], axis=0)
    ts = pd.DataFrame(
        {
            "i": i,
            "k": 1,
            "y": y,
            "tw_c95": ((lo95 <= y) & (y <= hi95)).astype(float),
            "tw_c50": ((lo50 <= y) & (y <= hi50)).astype(float),
            "tw_crps": [crps_ensemble(draws[:, j], y[j]) for j in range(y.size)],
        }
    )
    labels = {int(p): ("calibrated" if p % 2 == 0 else "too narrow") for p in range(n)}
    res = subgroup_calibration(ts, {"spread": labels}, twin_draws=draws, methods=("twin",), bootstrap_replicates=300)
    t = res.table[res.table["grouping"] == "spread"].set_index("level")
    good, bad = t.loc["calibrated"], t.loc["too narrow"]
    assert good["c95_lo"] <= 0.95 <= good["c95_hi"]
    assert bad["c95_hi"] < 0.8
    assert bad["pit_ks_p_adjusted"] < 1e-6
    assert good["pit_ks_p_adjusted"] > 0.01
    assert bad["crps"] > good["crps"]


def test_multiplicity_adjustments():
    p = np.array([0.01, 0.04, np.nan, 0.03, 0.5])
    finite = p[np.isfinite(p)]
    bh = _adjust(p, "bh")
    assert np.allclose(bh[np.isfinite(p)], stats.false_discovery_control(finite, method="bh"))
    assert np.isnan(bh[2])
    assert np.allclose(_adjust(p, "bonferroni")[np.isfinite(p)], np.minimum(finite * 4, 1))
    assert np.array_equal(_adjust(p, "none"), p, equal_nan=True)


def test_rejects_bad_input(fold):
    ts, draws = fold
    with pytest.raises(ValueError, match="no column"):
        subgroup_calibration(ts, {"x": "nope"})
    with pytest.raises(ValueError, match="no label"):
        subgroup_calibration(ts, {"site": {0: "a"}})
    with pytest.raises(ValueError, match="one label per target row"):
        subgroup_calibration(ts, {"x": ["a", "b"]})
    with pytest.raises(ValueError, match="reserved"):
        subgroup_calibration(ts, {"all": "sex"})
    with pytest.raises(ValueError, match="not both"):
        subgroup_calibration(ts, {}, twin_draws=draws, pit=np.zeros(len(ts)))
    with pytest.raises(ValueError, match="twin_draws"):
        subgroup_calibration(ts, {}, twin_draws=draws[:, :-1])
    with pytest.raises(ValueError, match="unknown methods"):
        subgroup_calibration(ts, {}, methods=("oracle",))
    with pytest.raises(ValueError, match="multiplicity"):
        subgroup_calibration(ts, {}, multiplicity="holm")


def test_markdown_and_plot(fold):
    ts, draws = fold
    res = subgroup_calibration(ts, {"sex": "sex"}, twin_draws=draws, min_cell=1, bootstrap_replicates=50)
    md = res.to_markdown(metric="crps")
    assert md.startswith("| grouping | level | endpoint | participants | twin | locf | lmm | PIT KS p |")
    for metric in ("c95", "c50", "crps"):
        assert res.plot(metric=metric) is not None
    assert res.plot(endpoint="moca") is not None
    with pytest.raises(ValueError):
        res.to_markdown(metric="auc")
