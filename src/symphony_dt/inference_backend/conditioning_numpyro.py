from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _json_safe(v):
    """Convert NaN/inf to None for JSON serialization."""
    if v is None:
        return None
    try:
        fv = float(v)
        if not math.isfinite(fv):
            return None
        return fv
    except Exception:
        return v
import pandas as pd

from symphony_dt.inference_backend.conditioning_store import ConditionalRunArtifacts, save_conditional_run
from symphony_dt.logging import get_logger
from symphony_dt.math.core import sigmoid, softplus
from symphony_dt.model.params import CouplingMatrix
from symphony_dt.model.observation import ObservationParams, SUPPORTS, mu_from_x
from symphony_dt.personalization.conditioning import ConditioningRunDiagnostics


@dataclass(frozen=True)
class NumPyroConditioningConfig:
    """
    Methods: Sequential conditioning uses NUTS with warm-starting.

    Operational budget (Methods):
      - warmup 250, sample 250, 4 chains
    Defaults here are smaller for practicality; set to Methods values in production config.
    """
    num_warmup: int = 250
    num_samples: int = 250
    num_chains: int = 4
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    seed: int = 0


def _require_numpyro() -> None:
    try:
        import jax  # noqa: F401
        import numpyro  # noqa: F401
        from numpyro.infer import MCMC, NUTS  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "Conditional NumPyro backend requires optional dependencies: jax, numpyro. "
            "Install with e.g. `pip install numpyro jax jaxlib` for your platform."
        ) from e


def _pack_obs_arrays(patient_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return times (T,), y (T,3) with NaNs, mask (T,3) bool."""
    g = patient_df.sort_values(["t_days_since_baseline", "visit_id"]).reset_index(drop=True)
    t = g["t_days_since_baseline"].to_numpy(dtype=np.float64)
    y = np.stack(
        [
            g["updrs3_total"].to_numpy(dtype=np.float64),
            g["moca_total"].to_numpy(dtype=np.float64),
            g["scopa_aut_total"].to_numpy(dtype=np.float64),
        ],
        axis=-1,
    )
    m = np.isfinite(y)
    return t, y, m


def _c_free_from_C(C: CouplingMatrix) -> np.ndarray:
    A = C.as_array()
    return np.array([A[0, 0], A[1, 0], A[1, 1], A[1, 2], A[2, 0], A[2, 2]], dtype=np.float64)


def run_conditioning_numpyro(
    *,
    patient_id: int,
    visit_index: int,
    global_draw_index: int,
    patient_clinic: pd.DataFrame,
    global_params: Dict[str, Any],
    out_dir: Path,
    cfg: NumPyroConditioningConfig,
    warm_start_samples: Optional[np.ndarray] = None,
) -> ConditionalRunArtifacts:
    """
    Runs conditional posterior for a single patient at conditioning visit_index.

    This conditions on visits up to visit_index (inclusive). It samples:
      - s0 (3,)
      - r (3,)
      - eps for each observed interval up to n-1

    Global parameters Phi are treated as fixed (taken from global_params).

    Returns and saves:
      - samples_x_tn: x(t_n) samples (S,3)
      - diagnostics fields needed by Methods gating rules (divergences, accept_prob_mean, rhat_r_*)
    """
    _require_numpyro()
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from numpyro.diagnostics import gelman_rubin

    log = get_logger(__name__)

    df = patient_clinic.sort_values(["t_days_since_baseline", "visit_id"]).reset_index(drop=True)
    if visit_index < 0 or visit_index >= len(df):
        raise ValueError("visit_index out of range for patient_clinic")

    df_cond = df.iloc[: visit_index + 1].copy()
    t_np, y_np, mask_np = _pack_obs_arrays(df_cond)
    T = len(t_np)
    if T < 1:
        raise ValueError("No conditioning visits")

    # Fixed global params
    # Expect: mu_r (3,), Gamma (3,2) not used here; A not used here; C_free or C components; sigma_xi (3,)
    # For conditioning, we require C and sigma_xi and observation params.
    sigma_xi = np.asarray(global_params.get("sigma_xi", [0.15, 0.15, 0.15]), dtype=np.float64).reshape(3)
    c_free = np.asarray(global_params.get("C_free", _c_free_from_C(CouplingMatrix(0.0,0.0,0.0,0.0,0.0,0.0))), dtype=np.float64).reshape(6)

    obs = ObservationParams(
        a_u=float(global_params.get("a_u", 0.0)),
        b_u=float(global_params.get("b_u", 1.0)),
        a_q=float(global_params.get("a_q", 0.0)),
        b_q=float(global_params.get("b_q", 1.0)),
        a_s=float(global_params.get("a_s", 0.0)),
        b_s=float(global_params.get("b_s", 1.0)),
        sigma_u=float(global_params.get("sigma_u", 5.0)),
        sigma_q=float(global_params.get("sigma_q", 1.5)),
        sigma_s=float(global_params.get("sigma_s", 3.0)),
    )
    obs.validate()

    y = jnp.array(y_np)
    m = jnp.array(mask_np)
    t = jnp.array(t_np)

    def unpack_C(c_free_v):
        C = jnp.zeros((3, 3))
        C = C.at[0, 0].set(c_free_v[0])
        C = C.at[1, 0].set(c_free_v[1])
        C = C.at[1, 1].set(c_free_v[2])
        C = C.at[1, 2].set(c_free_v[3])
        C = C.at[2, 0].set(c_free_v[4])
        C = C.at[2, 2].set(c_free_v[5])
        return C


    def mu_from_x_jax(x, obs):
        """JAX-safe observation mean mapping (no numpy conversion)."""
        eps = 1e-6
        x = jnp.clip(x, eps, 1.0 - eps)

        def sigmoid(z):
            return 1.0 / (1.0 + jnp.exp(-z))

        def logit(p):
            return jnp.log(p) - jnp.log1p(-p)

        # Supports
        L_U, U_U = 0.0, 132.0
        L_Q, U_Q = 0.0, 30.0
        L_S, U_S = 0.0, 69.0

        # h(p; a, b) = sigmoid(a + b*logit(p))
        h_u = sigmoid(obs.a_u + obs.b_u * logit(x[0]))
        h_q = sigmoid(obs.a_q + obs.b_q * logit(x[1]))
        h_s = sigmoid(obs.a_s + obs.b_s * logit(x[2]))

        mu_u = L_U + (U_U - L_U) * h_u
        mu_q = L_Q + (U_Q - L_Q) * (1.0 - h_q)
        mu_s = L_S + (U_S - L_S) * h_s
        return jnp.stack([mu_u, mu_q, mu_s], axis=0)

    def model():
        # Prior for s0 and r only (conditional given fixed Phi); shrinkage handled at scaffold level.
        s0 = numpyro.sample("s0", dist.Normal(0.0, 1.0).expand((3,)))
        r = numpyro.sample("r", dist.Normal(0.0, 1.0).expand((3,)))

        C = unpack_C(jnp.array(c_free))
        sx = jnp.array(sigma_xi)

        # eps per interval
        if T > 1:
            eps = numpyro.sample("epsilon", dist.Normal(0.0, 1.0).expand((T - 1, 3)))
        else:
            eps = jnp.zeros((0, 3))

        # recurse states
        s = s0
        xs = []
        for k in range(T):
            xk = 1.0 / (1.0 + jnp.exp(-s))
            xs.append(xk)
            if k < T - 1:
                dt = jnp.maximum(t[k + 1] - t[k], 0.0)
                Cx = C @ xk
                drive = jnp.log1p(jnp.exp(r + Cx))  # softplus
                drift = -0.5 * dt * (sx ** 2)
                diff = jnp.sqrt(dt) * (sx * eps[k])
                xi = jnp.exp(drift + diff)
                s = s + dt * (drive * xi)

        x_path = jnp.stack(xs, axis=0)  # (T,3)

        # likelihood on observed y with masking, approximating truncation by clipping (inputs already bounded)
        mu = []
        for k in range(T):
            mu_k = mu_from_x_jax(x_path[k], obs)  # uses numpy under hood; convert
            mu.append(mu_k)
        mu = jnp.stack(mu, axis=0)

        sig = jnp.array([obs.sigma_u, obs.sigma_q, obs.sigma_s])
        L = jnp.array([SUPPORTS["U"].L, SUPPORTS["Q"].L, SUPPORTS["S"].L])
        U = jnp.array([SUPPORTS["U"].U, SUPPORTS["Q"].U, SUPPORTS["S"].U])
        y_clip = jnp.clip(y, L, U)

        with numpyro.handlers.mask(mask=m):
            numpyro.sample("y", dist.Normal(mu, sig), obs=y_clip)

        numpyro.deterministic("x_tn", x_path[-1])

    kernel = NUTS(model, target_accept_prob=cfg.target_accept_prob, max_tree_depth=cfg.max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=cfg.num_warmup, num_samples=cfg.num_samples, num_chains=cfg.num_chains, progress_bar=False)

    rng_key = jax.random.PRNGKey(int(cfg.seed) + int(patient_id) * 1000 + int(visit_index) * 10 + int(global_draw_index))
    init_params = None
    if warm_start_samples is not None and warm_start_samples.size > 0:
        # Use median of previous samples as initial point for s0,r,epsilon
        med = np.median(warm_start_samples, axis=0)
        init_params = {"s0": jnp.array(med[:3])}  # minimal warm start

    mcmc.run(rng_key, init_params=init_params)
    samples = mcmc.get_samples(group_by_chain=True)  # (chains, draws, ...)
    extra = mcmc.get_extra_fields(group_by_chain=True)

    x_tn = np.asarray(samples["x_tn"]).reshape(cfg.num_chains * cfg.num_samples, 3).astype(np.float64)

    divergences = int(np.asarray(extra.get("diverging", np.zeros((cfg.num_chains, cfg.num_samples), dtype=bool))).sum())
    accept_prob = extra.get("accept_prob", None)
    if accept_prob is None:
        # compatibility across numpyro versions
        accept_prob = extra.get("mean_accept_prob", None)
    if accept_prob is None:
        accept_prob = extra.get("acceptance_rate", None)
    if accept_prob is not None:
        accept_prob_mean = float(np.asarray(accept_prob).mean())
    else:
        accept_prob_mean = None
# Rhat for r components (chains, draws, 3)
    r_chain = np.asarray(samples["r"])
    # gelman_rubin expects (chains, draws, ...)
    # R-hat is undefined for <2 chains or <2 draws; do not fail the run.
    if r_chain.ndim < 3 or r_chain.shape[0] < 2 or r_chain.shape[1] < 2:
        rhat = np.array([np.nan, np.nan, np.nan], dtype=float)
    else:
        rhat = np.asarray(gelman_rubin(r_chain)).reshape(3)
    diag = ConditioningRunDiagnostics(
        divergences=divergences,
        accept_prob_mean=accept_prob_mean,
        rhat_r_m=float(rhat[0]),
        rhat_r_c=float(rhat[1]),
        rhat_r_n=float(rhat[2]),
    )

    artifacts = ConditionalRunArtifacts(
        samples_x_tn=x_tn,
        diagnostics={
            "divergences": diag.divergences,
            "accept_prob_mean": _json_safe(diag.accept_prob_mean),
            "rhat_r_m": _json_safe(diag.rhat_r_m),
            "rhat_r_c": _json_safe(diag.rhat_r_c),
            "rhat_r_n": _json_safe(diag.rhat_r_n),
        },
        metadata={
            "backend": "numpyro",
            "patient_id": int(patient_id),
            "visit_index": int(visit_index),
            "global_draw_index": int(global_draw_index),
            "num_warmup": int(cfg.num_warmup),
            "num_samples": int(cfg.num_samples),
            "num_chains": int(cfg.num_chains),
            "seed": int(cfg.seed),
        },
    )

    run_dir = out_dir / f"patient_{patient_id}" / f"visit_{visit_index}" / f"draw_{global_draw_index}"
    save_conditional_run(run_dir, artifacts, overwrite=True)
    log.info("Saved conditional run to %s", run_dir)
    return artifacts
