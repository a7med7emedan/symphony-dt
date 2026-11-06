"""Gate constants, overrides, chain exclusion and the gate report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from symphony_dt import diagnostics as D
from symphony_dt.errors import GateError


def summary(rhat=1.005, ess=500.0):
    idx = ["lp__", "mu_r[1]", "b[1]", "s0[1,1]", "eps[2,3]", "bmat[1,1]", "b_raw[1,1]"]
    return pd.DataFrame(
        {
            "R_hat": [1.2, rhat, 1.001, 2.0, 2.0, 2.0, 2.0],
            "ESS_bulk": [10, ess, 900, 1, 1, 1, 1],
            "ESS_tail": [10, 450, 800, 1, 1, 1, 1],
        },
        index=idx,
    )


def lp(n_chains=4, trapped=()):
    rng = np.random.default_rng(1)
    x = rng.normal(-500, 10, size=(200, n_chains))
    for c in trapped:
        x[:, c] -= 500
    return x


def test_constants_are_the_study_gates():
    assert (D.C2_MAX_RHAT, D.C3_MIN_ESS_BULK, D.MIN_RETAINED_CHAINS, D.TRAP_MULTIPLE) == (1.01, 400.0, 4, 5.0)


def test_is_global():
    assert D.is_global("mu_r[1]") and D.is_global("Gamma.1.2")
    for name in ("lp__", "s0[1,2]", "eps.1.3", "bmat", "b_raw[2,1]"):
        assert not D.is_global(name)


def test_retained_chains_rule():
    keep = D.retained_chains(lp(5, trapped=(3,)))
    assert keep["excluded_chains"] == [4] and keep["n_retained"] == 4 and keep["meets_min_retained"]
    assert keep["pre_declared"] is True
    with pytest.raises(GateError):
        D.retained_chains(np.zeros(5))


def test_converged_derives_from_three_verdicts_only():
    ok = D.assess_tables(lp(), summary())
    assert ok.converged and ok.meets_c2 and ok.meets_c3 and ok.meets_min_retained
    assert ok.max_rhat == 1.005 and set(ok.diagnostics) == {"mu_r[1]", "b[1]"}
    assert not D.assess_tables(lp(), summary(rhat=1.02)).converged
    assert not D.assess_tables(lp(), summary(ess=399.0)).converged
    trapped = D.assess_tables(lp(4, trapped=(0,)), summary())
    assert not trapped.converged and trapped.chain_exclusion["n_retained"] == 3
    assert ok.lowest_ess(1) == [("mu_r[1]", 500.0)]
    assert "CONVERGED True" in ok.summary_lines()[-1]
    assert ok.to_dict()["converged"] is True
    with pytest.raises(GateError, match="no global"):
        D.assess_tables(lp(), summary().loc[["lp__", "s0[1,1]"]])


def test_sampler_columns_feed_the_report():
    sampler = {
        "divergent__": np.zeros((200, 4)),
        "treedepth__": np.full((200, 4), 10.0),
        "energy__": np.random.default_rng(2).normal(size=(200, 4)).cumsum(axis=0),
    }
    sampler["divergent__"][0, 0] = 1
    rep = D.assess_tables(lp(), summary(), sampler=sampler)
    assert rep.divergences == 1 and rep.max_treedepth_hits == 800 and len(rep.e_bfmi) == 4


def test_override_is_explicit_and_logged():
    with pytest.raises(GateError, match="reason"):
        D.GateOverride(reason="", author="x", c2_max_rhat=1.05)
    with pytest.raises(GateError, match="at least one"):
        D.GateOverride(reason="r", author="a")
    ov = D.GateOverride(reason="pilot run", author="A. Author", c2_max_rhat=1.05)
    with pytest.warns(D.GateOverrideWarning):
        rep = D.assess_tables(lp(), summary(rhat=1.02), override=ov)
    assert rep.converged and rep.override["reason"] == "pilot run"
    assert rep.thresholds["c2_max_rhat"] == 1.05
    with pytest.warns(D.GateOverrideWarning):
        assert D.retained_chains(lp(), override=D.GateOverride("r", "a", trap_multiple=2.0))["pre_declared"] is False


def test_csv_helpers(tmp_path):
    files = []
    for c in (10, 2, 1):
        p = tmp_path / f"m-1_{c}.csv"
        frame = pd.DataFrame({"lp__": np.arange(5.0), "divergent__": 0, "treedepth__": 3, "energy__": 1.0 + c})
        p.write_text("# comment\n" + frame.to_csv(index=False))
        files.append(p)
    ordered = D.chain_csvs_in_order(files)
    assert [o.rsplit("_", 1)[1] for o in ordered] == ["1.csv", "2.csv", "10.csv"]
    cols = D.read_sampler_columns(ordered)
    assert cols["lp__"].shape == (5, 3)
    short = tmp_path / "m-1_3.csv"
    short.write_text("lp__\n1\n")
    with pytest.raises(GateError, match="unequal"):
        D.read_sampler_columns([*ordered, short])
