from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from symphony_dt.inference.diagnostics import ConvergenceCriteria, DiagnosticsReport, evaluate_convergence
from symphony_dt.inference.store import PosteriorDrawSet


@dataclass(frozen=True)
class DrawExtractionConfig:
    """
    Methods:
      - Use L=200 approximately independent posterior draws per fold.

    This module implements:
      - deterministic extraction/downsampling to L draws
      - optional convergence validation (if rhat/ess present)
    """
    L_target: int = 200
    seed: int = 0
    allow_random_subsample: bool = True


def _flatten_chain_draws(a: np.ndarray) -> np.ndarray:
    # If (chains, draws, ...) flatten to (chains*draws, ...)
    if a.ndim >= 2 and a.shape[0] < 32 and a.shape[1] >= 50:
        return a.reshape((-1,) + a.shape[2:])
    return a


def extract_draws(drawset: PosteriorDrawSet, cfg: DrawExtractionConfig) -> PosteriorDrawSet:
    arrays = {k: _flatten_chain_draws(v) for k, v in drawset.arrays.items()}

    n = None
    for a in arrays.values():
        if n is None:
            n = int(a.shape[0])
        elif int(a.shape[0]) != n:
            raise ValueError("Inconsistent draw axis across arrays after flattening")
    n = int(n or 0)
    if n == 0:
        return PosteriorDrawSet(arrays=arrays, metadata=dict(drawset.metadata), diagnostics=dict(drawset.diagnostics))

    if n <= cfg.L_target:
        meta = dict(drawset.metadata)
        meta["L_extracted"] = n
        meta["L_target"] = cfg.L_target
        return PosteriorDrawSet(arrays=arrays, metadata=meta, diagnostics=dict(drawset.diagnostics))

    rng = np.random.default_rng(int(cfg.seed))
    if cfg.allow_random_subsample:
        idx = rng.choice(n, size=cfg.L_target, replace=False)
        idx = np.sort(idx).astype(int)
    else:
        step = max(1, n // cfg.L_target)
        idx = np.arange(0, n, step, dtype=int)[: cfg.L_target]

    arrays2 = {k: v[idx] for k, v in arrays.items()}
    meta = dict(drawset.metadata)
    meta["L_extracted"] = int(cfg.L_target)
    meta["L_target"] = cfg.L_target
    meta["extract_indices"] = idx.tolist()
    return PosteriorDrawSet(arrays=arrays2, metadata=meta, diagnostics=dict(drawset.diagnostics))


def validate_convergence_if_present(
    drawset: PosteriorDrawSet,
    criteria: ConvergenceCriteria = ConvergenceCriteria(),
    required_params: Optional[List[str]] = None,
) -> DiagnosticsReport:
    return evaluate_convergence(drawset.diagnostics, criteria=criteria, required_params=required_params)
