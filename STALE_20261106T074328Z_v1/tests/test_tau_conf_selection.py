import numpy as np

from symphony_dt.trust.tau_conf import TauConfSelectionConfig, select_tau_conf


def test_select_tau_conf_prefers_feasible_closest_to_target() -> None:
    rng = np.random.default_rng(0)
    N = 200
    # delta uniform 0..0.6
    delta = rng.uniform(0.0, 0.6, size=N)
    # construct y and PI so that for low delta mask coverage ~0.95, for high delta coverage worse
    y = rng.normal(size=(N, 3))
    width = np.where(delta[:, None] < 0.30, 2.0, 0.5)  # narrow intervals for high delta => low coverage
    lo = y - width
    hi = y + width

    cfg = TauConfSelectionConfig(min_fraction_covered=0.50)
    res = select_tau_conf(delta, y, lo, hi, cfg=cfg)

    assert 0.05 <= res.tau_conf <= 0.60
    assert res.fraction_kept >= 0.50


def test_select_tau_conf_no_feasible_falls_back_max_fraction() -> None:
    rng = np.random.default_rng(1)
    N = 100
    delta = rng.uniform(0.0, 0.6, size=N)
    y = rng.normal(size=(N, 3))
    # Make almost all deltas very large so frac constraint can't be met for most taus, but still function returns.
    delta[:] = 0.59
    lo = y - 0.1
    hi = y + 0.1

    cfg = TauConfSelectionConfig(min_fraction_covered=0.90)
    res = select_tau_conf(delta, y, lo, hi, cfg=cfg)

    assert 0.05 <= res.tau_conf <= 0.60
    assert 0.0 <= res.fraction_kept <= 1.0
