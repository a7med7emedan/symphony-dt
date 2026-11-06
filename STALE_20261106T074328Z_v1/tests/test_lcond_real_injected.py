import numpy as np
from pathlib import Path

from symphony_dt.global_uq.subsample import GlobalSubsampleConfig
from symphony_dt.inference.store import PosteriorDrawSet, save_drawset
from symphony_dt.inference_backend.lcond_real import select_l_cond_from_drawset


def test_select_l_cond_from_drawset_injected(tmp_path: Path) -> None:
    # Save dummy drawset with 200 draws
    ds = PosteriorDrawSet(arrays={"mu_r": np.zeros((200, 3))}, metadata={}, diagnostics={})
    draw_dir = tmp_path / "draws"
    save_drawset(draw_dir, ds, overwrite=True)

    def predictor(probe_ids, draw_idx, drawset):
        n = len(probe_ids)
        means = np.zeros((n, 3))
        widths = np.ones((n, 3)) * 10.0
        return means, widths

    res = select_l_cond_from_drawset(
        fold_id=0,
        drawset_dir=draw_dir,
        train_patient_ids=list(range(1, 500)),
        seed=123,
        predictor=predictor,
        cfg=GlobalSubsampleConfig(n_probe_patients=50),
    )
    assert res.l_cond == 10
    assert len(res.draw_indices) == 10
