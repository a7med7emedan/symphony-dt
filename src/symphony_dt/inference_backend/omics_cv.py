from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from symphony_dt.omics.pipeline import OmicsArtifacts, OmicsConfig, build_omics_covariates


@dataclass(frozen=True)
class OmicsCVConfig:
    """
    Per-fold omics pipeline configuration.

    This wraps Module 8 logic and standardizes fold artifact outputs:
      - fold_dir/omics_z.csv
      - fold_dir/omics_missing.csv
      - fold_dir/omics_training_stats.json
      - fold_dir/omics_pathways.json (raw mode)
    """
    omics: OmicsConfig


def build_fold_omics(
    *,
    omics_df: pd.DataFrame,
    patient_ids_all: Sequence[int],
    train_patient_ids: Sequence[int],
    fold_dir: Path,
    cfg: OmicsCVConfig,
) -> OmicsArtifacts:
    fold_dir.mkdir(parents=True, exist_ok=True)
    return build_omics_covariates(
        omics=omics_df,
        patient_ids_all=patient_ids_all,
        train_patient_ids=train_patient_ids,
        cfg=cfg.omics,
        output_dir=fold_dir,
    )
