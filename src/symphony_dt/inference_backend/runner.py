from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from symphony_dt.inference_backend.backend import InferenceBackend
from symphony_dt.inference_backend.numpyro_impl import NumPyroRunConfig, run_scaffold_numpyro
from symphony_dt.inference.store import PosteriorDrawSet


@dataclass(frozen=True)
class ScaffoldRunConfig:
    backend: InferenceBackend
    seed: int = 0
    P: int = 0
    numpyro: NumPyroRunConfig = NumPyroRunConfig()


def run_scaffold(
    *,
    clinic: pd.DataFrame,
    baseline: pd.DataFrame,
    patient_ids_train: Sequence[int],
    output_dir: Path,
    cfg: ScaffoldRunConfig,
) -> PosteriorDrawSet:
    if cfg.backend == InferenceBackend.NUMPYRO:
        return run_scaffold_numpyro(
            clinic=clinic,
            baseline=baseline,
            patient_ids_train=patient_ids_train,
            P=cfg.P,
            output_dir=output_dir,
            cfg=NumPyroRunConfig(
                num_warmup=cfg.numpyro.num_warmup,
                num_samples=cfg.numpyro.num_samples,
                num_chains=cfg.numpyro.num_chains,
                target_accept_prob=cfg.numpyro.target_accept_prob,
                max_tree_depth=cfg.numpyro.max_tree_depth,
                seed=cfg.seed,
            ),
        )
    raise NotImplementedError(f"Backend not implemented in runner: {cfg.backend}")
