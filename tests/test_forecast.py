"""Anchor split, thinning, GQ CSV reading, the Forecast object and the fast route."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from symphony_dt.data import Standardisation, simulate_cohort
from symphony_dt.forecast import (
    Forecast,
    forecast_fast,
    global_draws_from_csv,
    read_gq_csv,
    split_at_anchor,
    thin_draws,
)


def test_split_masks_only_test_tails(cohort):
    d = cohort.stan_data.data
    tests = [0, 3, 7]
    sp = split_at_anchor(cohort.stan_data, tests, anchor_years=2.0, zero_dose_std=-9.0)
    assert sp.fit.data["n_obs"] + sp.targets.data["n_obs"] == d["n_obs"]
    assert set(sp.meta["i"]) <= set(tests)
    assert (sp.meta["t"] > 2.0).all()
    assert (sp.meta["horizon"] > 0).all()
    assert sp.fit.data["h"] == d["h"] and sp.fit.data["K"] == d["K"]
    for i in range(d["N"]):
        if i not in tests:
            assert sp.fit.data["obs_count"][i] == d["obs_count"][i]
    up = sp.meta["k"].to_numpy() == 1
    ledd = np.asarray(sp.targets.data["obs_ledd"])
    assert np.all(ledd[~up] == 0.0)
    assert sp.meta["sex"].eq(-1).all() and sp.test_index == tests


def test_split_errors(cohort):
    with pytest.raises(ValueError, match="outside"):
        split_at_anchor(cohort.stan_data, [999])
    with pytest.raises(ValueError, match="no observation"):
        split_at_anchor(cohort.stan_data, [0], anchor_years=1000.0)


def test_split_write(cohort, tmp_path):
    sp = split_at_anchor(cohort.stan_data.data, [1, 2], cohort=["PD"] * 40, sex=[0] * 40)
    paths = sp.write(tmp_path)
    assert all(p.exists() for p in paths.values())
    assert set(pd.read_csv(paths["meta"])["cohort"]) == {"PD"}


def _chain_csv(path, chain, n=100, extra=()):
    cols = ["lp__", "accept_stat__", "mu_r.1", *extra]
    rng = np.random.default_rng(chain)
    lines = ["# model = test", "# Adaptation terminated", ",".join(cols)]
    lines += [",".join(f"{v:.6f}" for v in rng.normal(size=len(cols))) for _ in range(n)]
    lines += ["#", "#  Elapsed Time: 1 seconds (Warm-up)"]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_thin_draws(tmp_path):
    files = [_chain_csv(tmp_path / f"m_{c}.csv", c) for c in (1, 2, 3, 4)]
    out = thin_draws(files, tmp_path / "thin.csv", total_keep=40)
    rows = [ln for ln in out.read_text().splitlines() if ln and not ln.startswith("#")]
    assert len(rows) == 41 and rows[0].startswith("lp__")
    with pytest.raises(ValueError, match="kept"):
        thin_draws(files, tmp_path / "t2.csv", total_keep=300)
    with pytest.raises(ValueError, match="cannot keep"):
        thin_draws(files, tmp_path / "t3.csv", total_keep=1000)
    odd = _chain_csv(tmp_path / "m_5.csv", 5, extra=("x",))
    with pytest.raises(ValueError, match="header differs"):
        thin_draws([*files[:3], odd], tmp_path / "t4.csv", total_keep=40)
    short = _chain_csv(tmp_path / "m_6.csv", 6, n=50)
    with pytest.raises(ValueError, match="unequal"):
        thin_draws([*files[:3], short], tmp_path / "t5.csv", total_keep=40)


def test_read_gq_csv(tmp_path):
    p = tmp_path / "gq.csv"
    p.write_text("# c\nlp__,y_rep_c.1,y_rep_c.2,mu_c.1\n0,1,2,3\n0,4,5,6\n")
    assert read_gq_csv(p, 2).tolist() == [[1, 2], [4, 5]]
    with pytest.raises(ValueError, match="count or order"):
        read_gq_csv(p, 3)
    q = tmp_path / "gq2.csv"
    q.write_text("y_rep_c.1,x,y_rep_c.2\n1,2,3\n")
    with pytest.raises(ValueError, match="contiguous"):
        read_gq_csv(q, 2)


def test_forecast_object(tmp_path):
    rng = np.random.default_rng(0)
    targets = pd.DataFrame(
        {
            "i": [0, 0, 1],
            "k": [1, 2, 3],
            "horizon": [0.5, 1.0, 2.0],
            "y": [10.0, 28, 5],
            "cohort": ["PD", "PD", "Prodromal"],
        }
    )
    fc = Forecast(rng.normal([10, 27, 5], 2, size=(200, 3)), targets)
    assert fc.n_draws == 200 and fc.mean().shape == (3,)
    q = fc.quantiles()
    assert {"q025", "q5", "q975"} <= set(q.columns)
    iv = fc.interval(0.9)
    assert (iv["lower"] < iv["upper"]).all()
    p = fc.event_probability([5.0, 30.0, 1.0])
    assert 0 <= p[0] <= 1 and 0 <= p[1] <= 1 and np.isnan(p[2])
    assert len(fc.to_frame()) == 600
    assert fc.to_csv(tmp_path / "q.csv").exists()
    assert fc.to_netcdf(tmp_path / "f.nc").exists()
    ax = fc.plot()
    assert ax is not None
    with pytest.raises(ValueError, match="do not match"):
        Forecast(np.zeros((5, 2)), targets)


def test_fast_route_with_known_parameters():
    cohort = simulate_cohort(30, seed=5)
    pid = cohort.dataset.participant_ids[0]
    hist = cohort.dataset.subset([pid])
    std = Standardisation.fit(cohort.dataset)
    fc = forecast_fast([cohort.theta], hist, [0.5, 1.0], standardisation=std, n_particles=300, n_draws=60, seed=3)
    assert fc.draws.shape == (60, 6) and fc.method == "fast"
    assert fc.metadata["min_ess"] >= 1.0
    lo = np.array([0, 0, 0] * 2)
    hi = np.array([132, 30, 69] * 2)
    assert np.all(fc.draws >= lo) and np.all(fc.draws <= hi)
    with pytest.raises(ValueError, match="exactly one"):
        forecast_fast([cohort.theta], cohort.dataset, [1.0])
    sens = forecast_fast([cohort.theta], hist, [1.0], standardisation=std, n_particles=50, n_draws=5, ledd_std=0.5)
    assert sens.draws.shape == (5, 3)


def test_global_draws_from_csv(tmp_path):
    names = ["lp__", "lambda_ledd", "c_MM", "c_CM", "c_CC", "c_CN", "c_NM", "c_NN"]
    for base in ("mu_r", "sigma_r", "sigma_xi", "a", "b", "sigma_obs", "kappa"):
        names += [f"{base}.{j}" for j in (1, 2, 3)]
    names += [f"Gamma.{r}.{c}" for r in (1, 2, 3) for c in (1, 2, 3)]
    p = tmp_path / "m_1.csv"
    rows = [",".join(["0.5"] * len(names)) for _ in range(10)]
    p.write_text("\n".join([",".join(names), *rows]) + "\n")
    draws = global_draws_from_csv([p], max_draws=4)
    assert len(draws) == 4 and draws[0]["Gamma"].shape == (3, 3) and draws[0]["c_MM"] == 0.5
