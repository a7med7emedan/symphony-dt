import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from symphony_dt.inference_backend.conditioning_numpyro import NumPyroConditioningConfig, run_conditioning_numpyro


@pytest.mark.skipif(pytest.importorskip("numpyro", reason="numpyro not installed") is None, reason="numpyro not installed")
def test_conditioning_runs_if_numpyro_available(tmp_path: Path) -> None:
    clinic = pd.DataFrame(
        {
            "patient_id": [1, 1],
            "visit_id": [10, 11],
            "t_days_since_baseline": [0.0, 30.0],
            "updrs3_total": [10.0, 11.0],
            "moca_total": [28.0, 28.0],
            "scopa_aut_total": [5.0, 5.0],
        }
    )
    global_params = {
        "sigma_xi": [0.15, 0.15, 0.15],
        "C_free": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "a_u": 0.0, "b_u": 1.0,
        "a_q": 0.0, "b_q": 1.0,
        "a_s": 0.0, "b_s": 1.0,
        "sigma_u": 5.0, "sigma_q": 1.5, "sigma_s": 3.0,
    }
    cfg = NumPyroConditioningConfig(num_warmup=10, num_samples=10, num_chains=2, seed=0)
    art = run_conditioning_numpyro(
        patient_id=1,
        visit_index=1,
        global_draw_index=0,
        patient_clinic=clinic,
        global_params=global_params,
        out_dir=tmp_path,
        cfg=cfg,
    )
    assert art.samples_x_tn.shape[1] == 3
