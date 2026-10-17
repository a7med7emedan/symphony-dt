from __future__ import annotations

from typing import Any

from symphony_dt.model.observation import SUPPORTS

# Score supports in endpoint order (U, Q, S): MDS-UPDRS III, MoCA, SCOPA-AUT.
LOWER = tuple(SUPPORTS[k].L for k in ("U", "Q", "S"))
UPPER = tuple(SUPPORTS[k].U for k in ("U", "Q", "S"))


def observe_bounded(name: str, mu: Any, sigma: Any, y: Any, mask: Any) -> None:
    """
    Methods 'Clinic observation model with bounded likelihood', in NumPyro.

      y^(k) ~ Normal(mu^(k), sigma_k^2) truncated to [L_k, U_k]

    The truncation matters at the floor and ceiling of each scale (MoCA near 30 is
    common), where a plain Normal would leak probability mass outside the support.

    Missing entries are replaced by the support midpoint before scoring and then
    masked out, so they contribute neither log density nor NaN gradients.

    Shapes: mu, y, mask broadcast together with endpoints on the last axis;
    sigma has shape (3,).
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    low = jnp.asarray(LOWER)
    high = jnp.asarray(UPPER)
    mask = jnp.asarray(mask, dtype=bool)
    y = jnp.asarray(y)
    y_safe = jnp.where(mask & jnp.isfinite(y), jnp.clip(y, low, high), 0.5 * (low + high))

    with numpyro.handlers.mask(mask=mask):
        numpyro.sample(name, dist.TruncatedNormal(mu, sigma, low=low, high=high), obs=y_safe)


def chain_method(jax: Any, num_chains: int) -> str:
    """Run chains in parallel only when there is one JAX device per chain."""
    return "parallel" if jax.local_device_count() >= int(num_chains) else "sequential"
