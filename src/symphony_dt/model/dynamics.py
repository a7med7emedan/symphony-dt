from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from symphony_dt.math.core import sigmoid, softplus
from symphony_dt.model.params import CouplingMatrix, GlobalDynamicsParams

Vec3 = np.ndarray  # shape (3,)
Mat33 = np.ndarray  # shape (3,3)


def sample_epsilon(
    rng: np.random.Generator,
    size: Optional[int] = None,
) -> np.ndarray:
    """
    Sample epsilon ~ N(0, I_3), optionally batched.
    Returns:
      - shape (3,) if size is None
      - shape (size, 3) otherwise
    """
    if size is None:
        return rng.normal(loc=0.0, scale=1.0, size=3).astype(np.float64)
    return rng.normal(loc=0.0, scale=1.0, size=(int(size), 3)).astype(np.float64)


def xi_from_epsilon(
    epsilon: np.ndarray,
    dt: float,
    sigma_xi: np.ndarray,
) -> np.ndarray:
    """
    Eq. (xi_eps):
      xi = exp( -0.5*dt*sigma^2 + sqrt(dt)*(sigma ⊙ epsilon) )

    Supports epsilon shape (3,) or (...,3). sigma_xi shape (3,).
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    sigma_xi = np.asarray(sigma_xi, dtype=np.float64).reshape(3)
    if np.any(sigma_xi < 0):
        raise ValueError("sigma_xi must be >= 0")

    eps = np.asarray(epsilon, dtype=np.float64)
    if eps.shape[-1] != 3:
        raise ValueError("epsilon last dimension must be 3")

    drift = -0.5 * dt * (sigma_xi ** 2)
    diff = np.sqrt(dt) * (sigma_xi * eps)
    return np.exp(drift + diff)


def _coupling_array(C: Union[CouplingMatrix, np.ndarray]) -> Mat33:
    if isinstance(C, CouplingMatrix):
        return C.as_array()
    arr = np.asarray(C, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError("C must be shape (3,3)")
    return arr


def propagate_state(
    s_t: np.ndarray,
    r_i: np.ndarray,
    C: Union[CouplingMatrix, np.ndarray],
    sigma_xi: np.ndarray,
    dt: float,
    epsilon: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Eq. (trans_general) for one step from t to t+dt.

    s(t+dt) = s(t) + dt * ( softplus( r + C*x(t) ) ⊙ xi )
    x(t) = sigmoid(s(t))
    xi computed via Eq. (xi_eps)

    Supports s_t shape (3,) or (B,3). r_i same shape as s_t.
    Returns same shape as s_t.

    Determinism:
      - provide epsilon explicitly to make the step fully deterministic
      - otherwise provide rng to sample epsilon
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")

    s = np.asarray(s_t, dtype=np.float64)
    r = np.asarray(r_i, dtype=np.float64)
    if s.shape != r.shape:
        raise ValueError("s_t and r_i must have same shape")
    if s.shape[-1] != 3:
        raise ValueError("last dim must be 3")

    C_arr = _coupling_array(C)
    sx = np.asarray(sigma_xi, dtype=np.float64).reshape(3)

    if epsilon is None:
        if rng is None:
            raise ValueError("Provide either epsilon or rng")
        eps = sample_epsilon(rng, size=None if s.ndim == 1 else s.shape[0])
    else:
        eps = np.asarray(epsilon, dtype=np.float64)

    xi = xi_from_epsilon(eps, dt=dt, sigma_xi=sx)

    x = sigmoid(s)  # (..,3)

    # Compute C*x(t)
    if s.ndim == 1:
        Cx = C_arr @ x
    else:
        Cx = x @ C_arr.T

    drive = softplus(r + Cx)  # phi(...)
    delta = dt * (drive * xi)
    return s + delta


def induce_visit_time_trajectory(
    times_days: Sequence[float],
    s0: np.ndarray,
    r_i: np.ndarray,
    dyn: GlobalDynamicsParams,
    epsilons: Optional[Sequence[np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Deterministic construction on observed visit times (Methods).
    Given:
      - times_days: [t0, t1, ..., tN] (must be increasing, t0 can be 0)
      - s0: s(t0) shape (3,)
      - r_i: progression drive shape (3,)
      - dyn: global dynamics params (C topology + sigma_xi)
      - epsilons: optional list of epsilon vectors for each interval (length N, each shape (3,))
                  If provided, rng is ignored and recursion is fully deterministic.
      - rng: if epsilons not provided, used to sample epsilon per interval.

    Returns:
      array S of shape (N+1, 3) with S[n] = s(t_n)
    """
    if len(times_days) < 1:
        raise ValueError("times_days must be non-empty")
    t = np.asarray(times_days, dtype=np.float64)
    if (np.diff(t) <= 0).any():
        raise ValueError("times_days must be strictly increasing")

    s = np.asarray(s0, dtype=np.float64).reshape(3)
    r = np.asarray(r_i, dtype=np.float64).reshape(3)

    sx = dyn.sigma_xi_array()
    C_arr = dyn.C.as_array()

    n_intervals = len(t) - 1
    if epsilons is not None and len(epsilons) != n_intervals:
        raise ValueError("epsilons length must be len(times_days)-1")

    S = np.zeros((len(t), 3), dtype=np.float64)
    S[0] = s

    for k in range(n_intervals):
        dt = float(t[k + 1] - t[k])
        eps = None if epsilons is None else np.asarray(epsilons[k], dtype=np.float64).reshape(3)
        s = propagate_state(
            s_t=s,
            r_i=r,
            C=C_arr,
            sigma_xi=sx,
            dt=dt,
            epsilon=eps,
            rng=rng,
        ).reshape(3)
        S[k + 1] = s

    return S
