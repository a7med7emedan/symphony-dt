import numpy as np

from symphony_dt.global_uq.subsample import GlobalSubsampleConfig, choose_l_cond


def test_choose_l_cond_selects_small_when_stable() -> None:
    # Fake predict function: means and widths do not change with L, so smallest should be chosen.
    def predict_fn(pids, draw_idx):
        n = len(pids)
        means = np.zeros((n, 3), dtype=float)
        widths = np.ones((n, 3), dtype=float) * 10.0
        return means, widths

    cfg = GlobalSubsampleConfig(n_probe_patients=50)
    res = choose_l_cond(
        fold_id=0,
        L_total=200,
        train_patient_ids=list(range(1, 201)),
        seed=123,
        predict_fn=predict_fn,
        cfg=cfg,
    )
    assert res.l_cond == 10
    assert len(res.probe_patient_ids) == 50
    assert len(res.draw_indices) == 10


def test_choose_l_cond_falls_back_to_max_when_unstable() -> None:
    # Fake predict function: means change a lot for small L, stabilize only at max.
    def predict_fn(pids, draw_idx):
        n = len(pids)
        L = len(draw_idx)
        means = np.ones((n, 3), dtype=float) * (100.0 / L)  # big at small L
        widths = np.ones((n, 3), dtype=float) * (10.0 + 50.0 / L)
        return means, widths

    cfg = GlobalSubsampleConfig(n_probe_patients=50)
    res = choose_l_cond(
        fold_id=1,
        L_total=200,
        train_patient_ids=list(range(1, 301)),
        seed=123,
        predict_fn=predict_fn,
        cfg=cfg,
    )
    assert res.l_cond in cfg.candidates_l
    assert res.l_cond == max([c for c in cfg.candidates_l if c <= 200])
