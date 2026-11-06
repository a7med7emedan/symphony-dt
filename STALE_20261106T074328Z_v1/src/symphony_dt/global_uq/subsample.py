from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


CANDIDATES_L = (10, 15, 20, 25, 30, 40, 50)


@dataclass(frozen=True)
class GlobalSubsampleConfig:
    """
    Methods: Global-uncertainty propagation with adaptive subsampling.

    - Start with L=200 posterior draws.
    - Choose fold-specific L_cond from candidates.
    - Use a fixed set of 50 training participants selected once per fold by fixed seed.
    - Compare each candidate to next larger candidate for stability of:
        * predictive mean change thresholds:
            UPDRS <= 0.5
            MoCA  <= 0.2
            SCOPA <= 1.0
        * relative 95% interval width change <= 10% for all endpoints

    This module implements selection logic and deterministic index sampling.
    The actual predictive function is injected (so it can be backed by notebook parity
    or true model predictions in later modules).
    """
    n_probe_patients: int = 50
    candidates_l: Tuple[int, ...] = CANDIDATES_L
    mean_change_thresholds: Tuple[float, float, float] = (0.5, 0.2, 1.0)  # (U,Q,S)
    rel_width_threshold: float = 0.10
    # Seed salt to separate from other RNG uses.
    seed_salt: int = 91011


@dataclass(frozen=True)
class GlobalSubsampleResult:
    fold_id: int
    l_cond: int
    draw_indices: List[int]  # indices into [0, L-1]
    probe_patient_ids: List[int]
    diagnostics: Dict[str, float]  # summary of stability deltas


def select_training_probe_patients(
    train_patient_ids: Sequence[int],
    n_probe: int,
    seed: int,
    fold_id: int,
    seed_salt: int = 91011,
) -> List[int]:
    """
    Deterministic selection of probe patients from training IDs.

    Uses a seeded shuffle (stable across runs given same inputs) and returns first n_probe.
    """
    ids = sorted(set(int(x) for x in train_patient_ids))
    if n_probe <= 0:
        raise ValueError("n_probe must be > 0")
    if len(ids) <= n_probe:
        return ids

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(seed_salt), int(fold_id)]))
    perm = np.array(ids, dtype=np.int64)
    rng.shuffle(perm)
    return [int(x) for x in perm[:n_probe].tolist()]


def sample_draw_indices(
    L_total: int,
    L_sub: int,
    seed: int,
    fold_id: int,
    seed_salt: int = 91011,
) -> List[int]:
    """
    Deterministically sample L_sub indices without replacement from range(L_total).
    """
    if not (0 < L_sub <= L_total):
        raise ValueError("Require 0 < L_sub <= L_total")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(seed_salt), int(fold_id), int(L_sub)]))
    idx = rng.choice(L_total, size=L_sub, replace=False)
    idx = np.sort(idx).astype(int)
    return idx.tolist()


def _stability_metrics(
    pred_small: Tuple[np.ndarray, np.ndarray],
    pred_big: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute stability metrics between two predictive outputs over probe patients.

    Inputs:
      pred_* = (means, widths), each shape (N,3) for (U,Q,S).
    Returns:
      mean_abs_change: shape (3,)
      mean_rel_width_change: shape (3,)
    """
    mean_s, width_s = pred_small
    mean_b, width_b = pred_big
    if mean_s.shape != mean_b.shape or width_s.shape != width_b.shape:
        raise ValueError("pred_small and pred_big shapes must match")
    if mean_s.shape[-1] != 3:
        raise ValueError("last dim must be 3 (U,Q,S)")

    mean_abs_change = np.mean(np.abs(mean_s - mean_b), axis=0)

    denom = np.maximum(width_b, 1e-12)
    rel_width_change = np.mean(np.abs(width_s - width_b) / denom, axis=0)
    return mean_abs_change, rel_width_change


def choose_l_cond(
    fold_id: int,
    L_total: int,
    train_patient_ids: Sequence[int],
    seed: int,
    predict_fn: Callable[[List[int], List[int]], Tuple[np.ndarray, np.ndarray]],
    cfg: GlobalSubsampleConfig = GlobalSubsampleConfig(),
) -> GlobalSubsampleResult:
    """
    Choose smallest L_cond satisfying stability criteria relative to the next larger candidate.

    predict_fn signature:
      predict_fn(probe_patient_ids, draw_indices) -> (means, widths)
        means: ndarray (N,3) predictive means at one-step ahead for (U,Q,S)
        widths: ndarray (N,3) predictive 95% interval widths for (U,Q,S)

    Note: This function does NOT assume any particular inference backend.
    """
    if L_total <= 0:
        raise ValueError("L_total must be > 0")

    probe_ids = select_training_probe_patients(
        train_patient_ids=train_patient_ids,
        n_probe=cfg.n_probe_patients,
        seed=seed,
        fold_id=fold_id,
        seed_salt=cfg.seed_salt,
    )

    candidates = [c for c in cfg.candidates_l if c <= L_total]
    if len(candidates) < 1:
        raise ValueError("No candidates_l <= L_total")

    # Precompute indices and predictions for each candidate
    preds: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    idxs: Dict[int, List[int]] = {}
    for c in candidates:
        di = sample_draw_indices(L_total=L_total, L_sub=c, seed=seed, fold_id=fold_id, seed_salt=cfg.seed_salt)
        idxs[c] = di
        preds[c] = predict_fn(probe_ids, di)

    diagnostics: Dict[str, float] = {}
    chosen = candidates[-1]  # fallback to max
    chosen_idx = idxs[chosen]

    for i, c in enumerate(candidates[:-1]):
        c_next = candidates[i + 1]
        mac, rwd = _stability_metrics(preds[c], preds[c_next])

        diagnostics[f"mean_abs_change_U@{c}_vs_{c_next}"] = float(mac[0])
        diagnostics[f"mean_abs_change_Q@{c}_vs_{c_next}"] = float(mac[1])
        diagnostics[f"mean_abs_change_S@{c}_vs_{c_next}"] = float(mac[2])

        diagnostics[f"rel_width_change_U@{c}_vs_{c_next}"] = float(rwd[0])
        diagnostics[f"rel_width_change_Q@{c}_vs_{c_next}"] = float(rwd[1])
        diagnostics[f"rel_width_change_S@{c}_vs_{c_next}"] = float(rwd[2])

        thr = np.array(cfg.mean_change_thresholds, dtype=float)
        if (mac <= thr).all() and (rwd <= cfg.rel_width_threshold).all():
            chosen = c
            chosen_idx = idxs[c]
            break

    return GlobalSubsampleResult(
        fold_id=int(fold_id),
        l_cond=int(chosen),
        draw_indices=[int(x) for x in chosen_idx],
        probe_patient_ids=[int(x) for x in probe_ids],
        diagnostics=diagnostics,
    )
