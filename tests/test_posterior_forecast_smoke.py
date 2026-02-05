import numpy as np
import pandas as pd
from pathlib import Path

from symphony_dt.inference.store import PosteriorDrawSet, save_drawset
from symphony_dt.inference_backend.conditioning_store import ConditionalRunArtifacts, save_conditional_run
from symphony_dt.inference_backend.forecasting import posterior_predict_instance, PosteriorForecastConfig


def test_posterior_predict_instance_smoke(tmp_path: Path) -> None:
    # Create minimal drawset with needed arrays
    ds = PosteriorDrawSet(
        arrays={
            "C_free": np.zeros((200, 6), dtype=float),
            "sigma_xi": np.ones((200, 3), dtype=float) * 0.15,
            "mu_r": np.ones((200, 3), dtype=float) * 0.1,
            "a_u": np.zeros((200,)),
            "b_u": np.ones((200,)),
            "a_q": np.zeros((200,)),
            "b_q": np.ones((200,)),
            "a_s": np.zeros((200,)),
            "b_s": np.ones((200,)),
            "sigma_u": np.ones((200,)) * 5.0,
            "sigma_q": np.ones((200,)) * 1.5,
            "sigma_s": np.ones((200,)) * 3.0,
        },
        metadata={},
        diagnostics={},
    )
    draw_dir = tmp_path / "draws"
    save_drawset(draw_dir, ds, overwrite=True)

    # Create one conditional run artifact for patient 1 visit 0 draw 0
    cond_root = tmp_path / "cond"
    run_dir = cond_root / "patient_1" / "visit_0" / "draw_0"
    art = ConditionalRunArtifacts(
        samples_x_tn=np.clip(np.random.default_rng(0).uniform(0.2, 0.3, size=(50, 3)), 1e-6, 1-1e-6),
        diagnostics={"divergences": 0, "accept_prob_mean": 0.8, "rhat_r_m": 1.0, "rhat_r_c": 1.0, "rhat_r_n": 1.0},
        metadata={},
    )
    save_conditional_run(run_dir, art, overwrite=True)

    pred, reasons = posterior_predict_instance(
        patient_id=1,
        anchor_visit_index=0,
        t_anchor=0.0,
        t_target=30.0,
        y_anchor=np.array([10.0, 28.0, 5.0]),
        missing_anchor=(False, False, False),
        n_visits_total=2,
        drawset_dir=draw_dir,
        conditional_root=cond_root,
        selected_draw_indices=[0],
        l_cond=10,
        tau_conf=1.0,
        cfg=PosteriorForecastConfig(n_draws_predictive=200, seed=0),
    )
    assert pred is not None
    assert "pred_mean_updrs3" in pred
