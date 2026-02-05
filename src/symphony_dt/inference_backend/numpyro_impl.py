from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from symphony_dt.inference.store import PosteriorDrawSet, save_drawset
from symphony_dt.logging import get_logger
from symphony_dt.model.observation import SUPPORTS


@dataclass(frozen=True)
class NumPyroRunConfig:
    """Configuration for NUTS scaffold inference using NumPyro."""
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    seed: int = 0


def _require_numpyro() -> Any:
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        import numpyro  # noqa: F401
        import numpyro.distributions as dist  # noqa: F401
        from numpyro.infer import MCMC, NUTS  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "NumPyro backend requires optional dependencies: jax, numpyro. "
            "Install with e.g. `pip install numpyro jax jaxlib` for your platform."
        ) from e
    return True


def _pack_c_free(C: np.ndarray) -> np.ndarray:
    # free entries: c_mm, c_cm, c_cc, c_cn, c_nm, c_nn
    return np.array([C[0, 0], C[1, 0], C[1, 1], C[1, 2], C[2, 0], C[2, 2]], dtype=np.float64)


def _unpack_c_free_jax(c_free, jnp):
    C = jnp.zeros((3, 3))
    C = C.at[0, 0].set(c_free[0])
    C = C.at[1, 0].set(c_free[1])
    C = C.at[1, 1].set(c_free[2])
    C = C.at[1, 2].set(c_free[3])
    C = C.at[2, 0].set(c_free[4])
    C = C.at[2, 2].set(c_free[5])
    return C


def _sigmoid_jax(x, jnp):
    return 1.0 / (1.0 + jnp.exp(-x))


def _softplus_jax(x, jnp):
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)


def _h_link_jax(p, a, b, jnp, eps=1e-12):
    p = jnp.clip(p, eps, 1.0 - eps)
    logit = jnp.log(p / (1.0 - p))
    return _sigmoid_jax(a + b * logit, jnp)


def _mu_from_x_jax(x, a_u, b_u, a_q, b_q, a_s, b_s, jnp):
    x_m = x[..., 0]
    x_c = x[..., 1]
    x_n = x[..., 2]
    hu = _h_link_jax(x_m, a_u, b_u, jnp)
    hq = _h_link_jax(x_c, a_q, b_q, jnp)
    hs = _h_link_jax(x_n, a_s, b_s, jnp)
    mu_u = SUPPORTS["U"].L + (SUPPORTS["U"].U - SUPPORTS["U"].L) * hu
    mu_q = SUPPORTS["Q"].L + (SUPPORTS["Q"].U - SUPPORTS["Q"].L) * (1.0 - hq)
    mu_s = SUPPORTS["S"].L + (SUPPORTS["S"].U - SUPPORTS["S"].L) * hs
    return jnp.stack([mu_u, mu_q, mu_s], axis=-1)


def _build_scaffold_arrays(clinic: pd.DataFrame, patient_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      times: (N, T_max) padded with last value
      y: (N, T_max, 3) with NaNs where missing
      mask: (N, T_max, 3) bool, True where observed
      n_visits: (N,) int
    """
    df = clinic[clinic["patient_id"].isin(list(patient_ids))].copy()
    df = df.sort_values(["patient_id", "t_days_since_baseline", "visit_id"])
    groups = [g for _, g in df.groupby("patient_id", sort=True)]
    n = len(groups)
    n_visits = np.array([len(g) for g in groups], dtype=int)
    T = int(n_visits.max()) if n_visits.size else 0
    times = np.zeros((n, T), dtype=np.float64)
    y = np.full((n, T, 3), np.nan, dtype=np.float64)
    mask = np.zeros((n, T, 3), dtype=bool)

    for i, g in enumerate(groups):
        t = g["t_days_since_baseline"].to_numpy(dtype=np.float64)
        times[i, : len(t)] = t
        if len(t) < T:
            times[i, len(t) :] = t[-1]
        # outcomes
        u = g["updrs3_total"].to_numpy(dtype=np.float64)
        q = g["moca_total"].to_numpy(dtype=np.float64)
        s = g["scopa_aut_total"].to_numpy(dtype=np.float64)

        yy = np.stack([u, q, s], axis=-1)
        y[i, : yy.shape[0], :] = yy
        mm = np.isfinite(yy)
        mask[i, : yy.shape[0], :] = mm

    return times, y, mask, n_visits


def run_scaffold_numpyro(
    *,
    clinic: pd.DataFrame,
    baseline: pd.DataFrame,
    patient_ids_train: Sequence[int],
    P: int,
    output_dir: Path,
    cfg: NumPyroRunConfig,
) -> PosteriorDrawSet:
    """
    Executes NUTS scaffold inference using NumPyro (optional dependency).
    Saves draws to output_dir using Module 9 DrawStore layout.
    """
    _require_numpyro()
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    log = get_logger(__name__)

    # Prepare arrays
    times_np, y_np, mask_np, n_visits = _build_scaffold_arrays(clinic, patient_ids_train)
    N, T = times_np.shape
    times = jnp.array(times_np)
    y = jnp.array(y_np)
    obs_mask = jnp.array(mask_np)

    # Baseline covariates d_i = (age, male)
    b = baseline[baseline["patient_id"].isin(list(patient_ids_train))].copy()
    b = b.sort_values("patient_id")
    age = jnp.array(pd.to_numeric(b["age_baseline_years"], errors="coerce").to_numpy(dtype=np.float64))
    male = jnp.array((b["sex"].astype(str).str.upper() == "M").to_numpy(dtype=np.float64))
    d = jnp.stack([age, male], axis=-1)  # (N,2)

    # Omics z_i are expected to be precomputed elsewhere and passed in later modules; here set zeros.
    z = jnp.zeros((N, P), dtype=jnp.float32)

    def model():
        # Priors
        mu_r = numpyro.sample("mu_r", dist.Normal(0.0, 1.0).expand((3,)))
        Gamma = numpyro.sample("Gamma", dist.Normal(0.0, 0.5).expand((3, 2)))
        A = numpyro.sample("A", dist.Normal(0.0, 0.25).expand((3, P)))
        c_free = numpyro.sample("C_free", dist.Normal(0.0, 0.25).expand((6,)))
        C = _unpack_c_free_jax(c_free, jnp)

        sigma_r = numpyro.sample("sigma_r", dist.HalfNormal(0.5).expand((3,)))
        sigma_xi = numpyro.sample("sigma_xi", dist.HalfNormal(0.15).expand((3,)))

        a_u = numpyro.sample("a_u", dist.Normal(0.0, 1.0))
        b_u = numpyro.sample("b_u", dist.LogNormal(0.0, 0.5))
        a_q = numpyro.sample("a_q", dist.Normal(0.0, 1.0))
        b_q = numpyro.sample("b_q", dist.LogNormal(0.0, 0.5))
        a_s = numpyro.sample("a_s", dist.Normal(0.0, 1.0))
        b_s = numpyro.sample("b_s", dist.LogNormal(0.0, 0.5))

        sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(0.1 * (SUPPORTS["U"].U - SUPPORTS["U"].L)))
        sigma_q = numpyro.sample("sigma_q", dist.HalfNormal(0.1 * (SUPPORTS["Q"].U - SUPPORTS["Q"].L)))
        sigma_s = numpyro.sample("sigma_s", dist.HalfNormal(0.1 * (SUPPORTS["S"].U - SUPPORTS["S"].L)))

        # Per-patient random effect b_i
        b_i = numpyro.sample("b_i", dist.Normal(0.0, 1.0).expand((N, 3))) * sigma_r  # (N,3)

        r = mu_r + (d @ jnp.transpose(Gamma, (1, 0))) + (z @ jnp.transpose(A, (1, 0))) + b_i  # (N,3)

        s0 = numpyro.sample("s0", dist.Normal(0.0, 1.0).expand((N, 3)))  # (N,3)

        # Innovations epsilon per interval per patient per step (T-1,3), padded; will be ignored for beyond n_visits.
        eps = numpyro.sample("epsilon", dist.Normal(0.0, 1.0).expand((N, max(T - 1, 1), 3)))

        def step(carry, inp):
            s_prev = carry  # (N,3)
            dt, eps_t = inp  # dt: (N,), eps_t: (N,3)
            x_prev = _sigmoid_jax(s_prev, jnp)
            Cx = jnp.einsum("ij,nj->ni", C, x_prev)  # (N,3)
            drive = _softplus_jax(r + Cx, jnp)  # (N,3)

            drift = -0.5 * dt[:, None] * (sigma_xi[None, :] ** 2)
            diff = jnp.sqrt(dt)[:, None] * (sigma_xi[None, :] * eps_t)
            xi = jnp.exp(drift + diff)
            s_next = s_prev + dt[:, None] * (drive * xi)
            return s_next, s_next

        # Build dt per step: dt[:,t] for t=0..T-2 (between visit t and t+1), last unused.
        dt_mat = jnp.maximum(times[:, 1:] - times[:, :-1], 0.0)  # (N,T-1)
        # scan over t dimension
        if T > 1:
            _, s_path = jax.lax.scan(step, s0, (dt_mat.T, eps[:, : (T - 1), :].transpose((1, 0, 2))))
            s_path = jnp.concatenate([s0[None, :, :], s_path], axis=0)  # (T,N,3)
        else:
            s_path = s0[None, :, :]

        x_path = _sigmoid_jax(s_path, jnp)  # (T,N,3)
        mu_path = _mu_from_x_jax(x_path, a_u, b_u, a_q, b_q, a_s, b_s, jnp)  # (T,N,3)

        # Likelihood with truncation via clamping (approx) + masking.
        # NumPyro doesn't have a built-in stable TruncatedNormal in all versions; implement via Normal with masking and manual support check.
        sig = jnp.array([sigma_u, sigma_q, sigma_s])

        # Apply observation mask: only include logprob where observed and within bounds.
        y_obs = y  # (N,T,3) after transpose next
        mu_obs = jnp.transpose(mu_path, (1, 0, 2))  # (N,T,3)
        y_obs = y_obs
        m = obs_mask

        # bounds check; if out of bounds, likelihood still possible under truncation but would be invalid for truncated distribution.
        # assume input data are in bounds (validated earlier); still clip for safety.
        L = jnp.array([SUPPORTS["U"].L, SUPPORTS["Q"].L, SUPPORTS["S"].L])
        U = jnp.array([SUPPORTS["U"].U, SUPPORTS["Q"].U, SUPPORTS["S"].U])
        y_clip = jnp.clip(y_obs, L, U)

        with numpyro.handlers.mask(mask=m):
            numpyro.sample("y", dist.Normal(mu_obs, sig), obs=y_clip)

    kernel = NUTS(model, target_accept_prob=cfg.target_accept_prob, max_tree_depth=cfg.max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=cfg.num_warmup, num_samples=cfg.num_samples, num_chains=cfg.num_chains, progress_bar=False)
    rng_key = jax.random.PRNGKey(int(cfg.seed))
    mcmc.run(rng_key)
    samples = mcmc.get_samples(group_by_chain=False)
    extra = mcmc.get_extra_fields()

    # Build drawset arrays
    arrays: Dict[str, np.ndarray] = {}
    for k, v in samples.items():
        arrays[k] = np.asarray(v)

    metadata = {
        "backend": "numpyro",
        "num_warmup": cfg.num_warmup,
        "num_samples": cfg.num_samples,
        "num_chains": cfg.num_chains,
        "seed": cfg.seed,
        "n_train_patients": int(N),
        "P": int(P),
    }

    # Minimal diagnostics
    diagnostics: Dict[str, Any] = {
        "divergences": int(np.asarray(extra.get("diverging", np.array([]))).sum()) if "diverging" in extra else 0,
    }

    drawset = PosteriorDrawSet(arrays=arrays, metadata=metadata, diagnostics=diagnostics)
    save_drawset(output_dir, drawset, overwrite=True)
    log.info("Saved NumPyro drawset to %s", output_dir)
    return drawset
