from __future__ import annotations

import math

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from symphony_dt.inference.store import load_drawset
from symphony_dt.inference_backend.conditioning_store import load_conditional_run
from symphony_dt.logging import get_logger
from symphony_dt.math.core import logit, sigmoid, softplus
from symphony_dt.math.truncnorm import truncnorm_rvs
from symphony_dt.model.params import CouplingMatrix
from symphony_dt.model.observation import ObservationParams, SUPPORTS, mu_from_x
from symphony_dt.personalization.conditioning import ConditioningMixture, ConditionalRun, ConditioningRunDiagnostics, filter_successful_runs
from symphony_dt.personalization.gating import TrustConfig, decide_trust_and_suppression


@dataclass(frozen=True)
class PosteriorForecastConfig:
    """
    Posterior predictive forecasting per Methods.

    This implementation consumes:
      - global posterior draws (L=200) stored in DrawStore (Module 9)
      - per-patient conditional run artifacts (Module 30) for selected L_cond indices

    It produces predictive summaries (mean, p05, p95) for U/Q/S.
    """
    n_draws_predictive: int = 2000
    seed: int = 0


def _unpack_C(c_free: np.ndarray) -> np.ndarray:
    c = np.asarray(c_free, dtype=np.float64).reshape(6)
    C = np.zeros((3, 3), dtype=np.float64)
    C[0, 0] = c[0]
    C[1, 0] = c[1]
    C[1, 1] = c[2]
    C[1, 2] = c[3]
    C[2, 0] = c[4]
    C[2, 2] = c[5]
    return C


def _xi(dt: float, sigma_xi: np.ndarray, eps: np.ndarray) -> np.ndarray:
    drift = -0.5 * dt * (sigma_xi ** 2)
    diff = np.sqrt(dt) * (sigma_xi * eps)
    return np.exp(drift + diff)


def _propagate_s(
    s_t: np.ndarray,
    r: np.ndarray,
    C: np.ndarray,
    sigma_xi: np.ndarray,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = sigmoid(s_t)
    Cx = C @ x
    drive = softplus(r + Cx)
    eps = rng.normal(size=3)
    xi = _xi(dt, sigma_xi, eps)
    return s_t + dt * (drive * xi)


def _sample_obs_from_mu(mu: np.ndarray, obs: ObservationParams, rng: np.random.Generator) -> np.ndarray:
    sig = np.array([obs.sigma_u, obs.sigma_q, obs.sigma_s], dtype=np.float64)
    L = np.array([SUPPORTS["U"].L, SUPPORTS["Q"].L, SUPPORTS["S"].L], dtype=np.float64)
    U = np.array([SUPPORTS["U"].U, SUPPORTS["Q"].U, SUPPORTS["S"].U], dtype=np.float64)
    y = np.zeros(3, dtype=np.float64)
    for k in range(3):
        y[k] = truncnorm_rvs(mu[k], sig[k], L[k], U[k], size=1, rng=rng)[0]
    return y


def _diag_value(diag: Dict[str, Any], key: str, fail_value: float) -> float:
    """Read a diagnostic as float; absent, None or non-finite values return the failing value."""
    v = diag.get(key)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return fail_value
    return f if math.isfinite(f) else fail_value


def load_conditioning_mixture(
    *,
    conditional_root: Path,
    patient_id: int,
    visit_index: int,
    selected_draw_indices: Sequence[int],
    l_cond: int,
) -> Tuple[ConditioningMixture, int]:
    """
    Load conditional runs for (patient, visit_index) for each global draw index in selected_draw_indices.

    Expects Module 30 directory layout:
      conditional_root/patient_<id>/visit_<n>/draw_<ell>/
    """
    runs: List[ConditionalRun] = []
    for ell in selected_draw_indices:
        d = conditional_root / f"patient_{patient_id}" / f"visit_{visit_index}" / f"draw_{int(ell)}"
        try:
            art = load_conditional_run(d)
        except FileNotFoundError:
            continue
        diag = art.diagnostics
        # A missing or undefined diagnostic (None, NaN) must fail the gate, never crash it.
        crd = ConditioningRunDiagnostics(
            divergences=int(_diag_value(diag, "divergences", 1.0)),
            accept_prob_mean=_diag_value(diag, "accept_prob_mean", 0.0),
            rhat_r_m=_diag_value(diag, "rhat_r_m", 99.0),
            rhat_r_c=_diag_value(diag, "rhat_r_c", 99.0),
            rhat_r_n=_diag_value(diag, "rhat_r_n", 99.0),
        )
        runs.append(
            ConditionalRun(
                patient_id=int(patient_id),
                visit_index=int(visit_index),
                global_draw_index=int(ell),
                diagnostics=crd,
                samples_x_tn=np.asarray(art.samples_x_tn, dtype=np.float64),
            )
        )

    ok, min_pass = filter_successful_runs(runs, l_cond=l_cond)
    mix = ConditioningMixture(patient_id=int(patient_id), visit_index=int(visit_index), l_cond=int(l_cond), successful_runs=ok)
    return mix, min_pass


def posterior_predict_instance(
    *,
    patient_id: int,
    anchor_visit_index: int,
    t_anchor: float,
    t_target: float,
    y_anchor: np.ndarray,  # (3,) with NaNs for missing
    missing_anchor: Tuple[bool, bool, bool],
    n_visits_total: int,
    drawset_dir: Path,
    conditional_root: Path,
    selected_draw_indices: Sequence[int],
    l_cond: int,
    tau_conf: float,
    cfg: PosteriorForecastConfig,
) -> Tuple[Optional[Dict[str, float]], Dict[str, str]]:
    """
    Produce predictive summaries for one target time.

    Returns:
      (pred_dict or None if suppressed, reasons)
    pred_dict has keys: pred_mean_*, pred_p05_*, pred_p95_* for endpoints.
    """
    drawset = load_drawset(drawset_dir)

    mix, min_pass = load_conditioning_mixture(
        conditional_root=conditional_root,
        patient_id=patient_id,
        visit_index=anchor_visit_index,
        selected_draw_indices=selected_draw_indices,
        l_cond=l_cond,
    )

    # gating decision
    # boundary detection uses exact values; if NaN, treat as None
    u0 = None if not np.isfinite(y_anchor[0]) else float(y_anchor[0])
    q0 = None if not np.isfinite(y_anchor[1]) else float(y_anchor[1])
    s0 = None if not np.isfinite(y_anchor[2]) else float(y_anchor[2])

    dec = decide_trust_and_suppression(
        n_visits_total=n_visits_total,
        missing_at_anchor=missing_anchor,
        anchor_values=(u0, q0, s0),
        mixture=mix,
        cfg=TrustConfig(tau_conf=float(tau_conf)),
        min_required_success=min_pass,
    )
    if not dec.allow_forecasts:
        return None, dec.reasons

    # Extract global param draws arrays (shape (L_total, ...))
    # Here we assume drawset already extracted to L=200 by Module 27.
    # If larger, we index into it.
    C_free_all = drawset.arrays.get("C_free", None)
    sigma_xi_all = drawset.arrays.get("sigma_xi", None)
    mu_r_all = drawset.arrays.get("mu_r", None)

    # Observation params arrays
    a_u_all = drawset.arrays.get("a_u", None)
    b_u_all = drawset.arrays.get("b_u", None)
    a_q_all = drawset.arrays.get("a_q", None)
    b_q_all = drawset.arrays.get("b_q", None)
    a_s_all = drawset.arrays.get("a_s", None)
    b_s_all = drawset.arrays.get("b_s", None)
    sigma_u_all = drawset.arrays.get("sigma_u", None)
    sigma_q_all = drawset.arrays.get("sigma_q", None)
    sigma_s_all = drawset.arrays.get("sigma_s", None)

    if C_free_all is None or sigma_xi_all is None:
        raise ValueError("Drawset missing required arrays: C_free and sigma_xi")

    dt = float(t_target - t_anchor)
    if dt <= 0:
        raise ValueError("t_target must be > t_anchor")

    rng = np.random.default_rng(int(cfg.seed) + int(patient_id) * 1000 + int(anchor_visit_index) * 10 + int(dt))

    # Build predictive draws
    S = int(cfg.n_draws_predictive)
    y_draws = np.zeros((S, 3), dtype=np.float64)

    # Sample pool of x(tn) from successful runs (mixture)
    x_pool = mix.sample_x_pool()
    if x_pool.shape[0] == 0:
        return None, {"rule_4": "insufficient_successful_conditioning_runs"}

    # Determine which global draws are represented (successful)
    successful_ells = [r.global_draw_index for r in mix.successful_runs]
    if not successful_ells:
        return None, {"rule_4": "insufficient_successful_conditioning_runs"}

    for s_idx in range(S):
        ell = int(rng.choice(successful_ells))
        # pick one x sample from that run
        run = mix.successful_runs[int(rng.integers(0, len(mix.successful_runs)))]
        x_tn = run.samples_x_tn[int(rng.integers(0, run.samples_x_tn.shape[0]))]
        s_tn = logit(x_tn)

        C = _unpack_C(C_free_all[ell])
        sigma_xi = np.asarray(sigma_xi_all[ell], dtype=np.float64).reshape(3)

        # r: if per-patient r samples exist in drawset (not yet), fallback to mu_r
        if mu_r_all is not None:
            r = np.asarray(mu_r_all[ell], dtype=np.float64).reshape(3)
        else:
            r = np.zeros(3, dtype=np.float64)

        s_target = _propagate_s(s_t=s_tn, r=r, C=C, sigma_xi=sigma_xi, dt=dt, rng=rng)
        x_target = sigmoid(s_target)

        # observation params for ell (fallbacks if arrays absent)
        obs = ObservationParams(
            a_u=float(a_u_all[ell]) if a_u_all is not None else 0.0,
            b_u=float(b_u_all[ell]) if b_u_all is not None else 1.0,
            a_q=float(a_q_all[ell]) if a_q_all is not None else 0.0,
            b_q=float(b_q_all[ell]) if b_q_all is not None else 1.0,
            a_s=float(a_s_all[ell]) if a_s_all is not None else 0.0,
            b_s=float(b_s_all[ell]) if b_s_all is not None else 1.0,
            sigma_u=float(sigma_u_all[ell]) if sigma_u_all is not None else 5.0,
            sigma_q=float(sigma_q_all[ell]) if sigma_q_all is not None else 1.5,
            sigma_s=float(sigma_s_all[ell]) if sigma_s_all is not None else 3.0,
        )
        obs.validate()

        mu = mu_from_x(x_target, obs)
        y_draws[s_idx] = _sample_obs_from_mu(mu, obs, rng)

    # summarize
    mean = np.mean(y_draws, axis=0)
    p05 = np.quantile(y_draws, 0.05, axis=0)
    p95 = np.quantile(y_draws, 0.95, axis=0)

    pred = {
        "pred_mean_updrs3": float(mean[0]),
        "pred_p05_updrs3": float(p05[0]),
        "pred_p95_updrs3": float(p95[0]),
        "pred_mean_moca": float(mean[1]),
        "pred_p05_moca": float(p05[1]),
        "pred_p95_moca": float(p95[1]),
        "pred_mean_scopa": float(mean[2]),
        "pred_p05_scopa": float(p05[2]),
        "pred_p95_scopa": float(p95[2]),
        "engine": "posterior_sim",
    }
    # if boundary rule suppressed events, that affects event-prob reporting later; forecasts still ok.
    return pred, dec.reasons
