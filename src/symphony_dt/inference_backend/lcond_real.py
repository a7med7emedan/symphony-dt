from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from symphony_dt.global_uq.subsample import GlobalSubsampleConfig, GlobalSubsampleResult, choose_l_cond
from symphony_dt.inference.store import PosteriorDrawSet, load_drawset
from symphony_dt.logging import get_logger


@dataclass(frozen=True)
class OneStepPredictorConfig:
    """
    Adapter config for L_cond selection using *real* one-step predictions.

    In Module 29, the predictor remains backend-injected because the full conditioning engine
    is implemented in Module 30. This module wires the global subsample selection to
    a drawset and a predictor interface.
    """
    L_total: int = 200
    # thresholds are in GlobalSubsampleConfig


def select_l_cond_from_drawset(
    *,
    fold_id: int,
    drawset_dir: Path,
    train_patient_ids: Sequence[int],
    seed: int,
    predictor: Callable[[List[int], List[int], PosteriorDrawSet], Tuple[np.ndarray, np.ndarray]],
    cfg: GlobalSubsampleConfig = GlobalSubsampleConfig(),
    pred_cfg: OneStepPredictorConfig = OneStepPredictorConfig(),
) -> GlobalSubsampleResult:
    """
    Select L_cond per Methods using a stored PosteriorDrawSet and an injected predictor.

    predictor signature:
      predictor(probe_patient_ids, draw_indices, drawset) -> (means, widths)
        means: (N,3) predictive means for U,Q,S
        widths: (N,3) 95% interval widths for U,Q,S
    """
    log = get_logger(__name__)
    drawset = load_drawset(drawset_dir)

    # ensure we have at least L_total draws
    n = drawset.n_draws()
    if n < pred_cfg.L_total:
        raise ValueError(f"drawset has {n} draws but L_total={pred_cfg.L_total} required by Methods")

    def predict_fn(probe_ids: List[int], draw_idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        return predictor(probe_ids, draw_idx, drawset)

    res = choose_l_cond(
        fold_id=fold_id,
        L_total=pred_cfg.L_total,
        train_patient_ids=train_patient_ids,
        seed=seed,
        predict_fn=predict_fn,
        cfg=cfg,
    )
    log.info("Selected L_cond=%d for fold=%d", res.l_cond, fold_id)
    return res
