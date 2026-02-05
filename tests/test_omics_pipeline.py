from pathlib import Path

import pandas as pd

from symphony_dt.omics.pipeline import OmicsConfig, build_omics_covariates


def test_omics_z_mode_standardizes_by_train_sd(tmp_path: Path) -> None:
    omics = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "z_p1": [1.0, 2.0, 3.0],
            "z_p2": [10.0, 10.0, 10.0],
        }
    )
    cfg = OmicsConfig(mode="z", z_prefix="z_")
    art = build_omics_covariates(
        omics=omics,
        patient_ids_all=[1, 2, 3],
        train_patient_ids=[1, 2],
        cfg=cfg,
        output_dir=tmp_path,
    )
    assert "z_p1" in art.z.columns
    assert "z_p2" in art.z.columns
    assert (tmp_path / "omics_training_stats.json").exists()


def test_omics_raw_mode_selects_latest_prebaseline_and_scores(tmp_path: Path) -> None:
    mem = pd.DataFrame(
        {
            "pathway": ["A"] * 5 + ["B"] * 5,
            "feature": [f"f{i}" for i in range(1, 6)] * 2,
            "is_member": [1] * 10,
        }
    )
    mem_path = tmp_path / "mem.csv"
    mem.to_csv(mem_path, index=False)

    omics = pd.DataFrame(
        {
            "patient_id": [1, 1, 2],
            "t_draw_days": [-10, -1, 5],
            "f1": [0.0, 1.0, 10.0],
            "f2": [0.0, 1.0, 10.0],
            "f3": [0.0, 1.0, 10.0],
            "f4": [0.0, 1.0, 10.0],
            "f5": [0.0, 1.0, 10.0],
        }
    )

    cfg = OmicsConfig(mode="raw", pathway_membership_path=str(mem_path), draw_time_col="t_draw_days")
    art = build_omics_covariates(
        omics=omics,
        patient_ids_all=[1, 2],
        train_patient_ids=[1],
        cfg=cfg,
        output_dir=tmp_path,
    )
    assert any(c.startswith("z_") for c in art.z.columns if c != "patient_id")
    row2 = art.z[art.z["patient_id"] == 2].iloc[0]
    z_cols = [c for c in art.z.columns if c != "patient_id"]
    assert float(row2[z_cols].sum()) == 0.0
    assert (tmp_path / "omics_training_stats.json").exists()
    assert (tmp_path / "omics_pathways.json").exists()
