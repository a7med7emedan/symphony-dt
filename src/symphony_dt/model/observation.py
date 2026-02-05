from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

from symphony_dt.math.core import logit, sigmoid
from symphony_dt.math.truncnorm import truncnorm_cdf, truncnorm_rvs

ArrayLike = Union[float, np.ndarray]


@dataclass(frozen=True)
class EndpointSupport:
    name: str
    L: float
    U: float


SUPPORTS: Dict[str, EndpointSupport] = {
    "U": EndpointSupport(name="UPDRS3", L=0.0, U=132.0),
    "Q": EndpointSupport(name="MoCA", L=0.0, U=30.0),
    "S": EndpointSupport(name="SCOPA_AUT", L=0.0, U=69.0),
}


@dataclass(frozen=True)
class ObservationParams:
    """
    Observation model parameters (Methods, Section 'Clinic observation model with bounded likelihood').

    Link:
      h(p;a,b) = sigmoid(a + b*logit(p)), with b>0.

    Means:
      mu_U = L_U + (U_U-L_U)*h(x_M; a_U, b_U)
      mu_Q = L_Q + (U_Q-L_Q)*(1 - h(x_C; a_Q, b_Q))
      mu_S = L_S + (U_S-L_S)*h(x_N; a_S, b_S)

    Likelihood:
      y^(k) ~ Normal(mu^(k), sigma_k^2) truncated to [L_k, U_k]
    """
    a_u: float
    b_u: float
    a_q: float
    b_q: float
    a_s: float
    b_s: float
    sigma_u: float
    sigma_q: float
    sigma_s: float

    def validate(self) -> None:
        if not (self.b_u > 0 and self.b_q > 0 and self.b_s > 0):
            raise ValueError("b_u, b_q, b_s must be > 0")
        if not (self.sigma_u > 0 and self.sigma_q > 0 and self.sigma_s > 0):
            raise ValueError("sigma_u, sigma_q, sigma_s must be > 0")


def h_link(p: ArrayLike, a: float, b: float, eps: float = 1e-12) -> ArrayLike:
    """h(p;a,b)=sigmoid(a+b*logit(p))"""
    return sigmoid(a + b * logit(p, eps=eps))


def mu_from_x(
    x: np.ndarray,
    obs: ObservationParams,
) -> np.ndarray:
    """
    Map latent x(t) in (0,1)^3 to endpoint means (mu_U, mu_Q, mu_S).

    Input:
      x: shape (3,) or (B,3), dims correspond to (M,C,N)

    Output:
      mu: shape (3,) or (B,3) in order (U,Q,S)
    """
    obs.validate()
    xx = np.asarray(x, dtype=np.float64)
    if xx.shape[-1] != 3:
        raise ValueError("x last dimension must be 3 (M,C,N)")

    x_m = xx[..., 0]
    x_c = xx[..., 1]
    x_n = xx[..., 2]

    sup_u = SUPPORTS["U"]
    sup_q = SUPPORTS["Q"]
    sup_s = SUPPORTS["S"]

    hu = h_link(x_m, obs.a_u, obs.b_u)
    hq = h_link(x_c, obs.a_q, obs.b_q)
    hs = h_link(x_n, obs.a_s, obs.b_s)

    mu_u = sup_u.L + (sup_u.U - sup_u.L) * hu
    mu_q = sup_q.L + (sup_q.U - sup_q.L) * (1.0 - hq)
    mu_s = sup_s.L + (sup_s.U - sup_s.L) * hs

    mu = np.stack([mu_u, mu_q, mu_s], axis=-1)
    return mu.astype(np.float64)


def sample_observations(
    mu: np.ndarray,
    obs: ObservationParams,
    rng: np.random.Generator,
    size: Optional[int] = None,
) -> np.ndarray:
    """
    Sample from truncated Gaussian likelihood for each endpoint.

    Input:
      mu: shape (3,) or (B,3) endpoint means in order (U,Q,S)
      size: optional number of iid samples per row of mu
        - if mu is (3,) and size is None => returns (3,)
        - if mu is (3,) and size is S => returns (S,3)
        - if mu is (B,3) and size is None => returns (B,3)
        - if mu is (B,3) and size is S => returns (B,S,3)
    """
    obs.validate()
    m = np.asarray(mu, dtype=np.float64)
    if m.shape[-1] != 3:
        raise ValueError("mu last dimension must be 3 (U,Q,S)")

    sigmas = np.array([obs.sigma_u, obs.sigma_q, obs.sigma_s], dtype=np.float64)
    Ls = np.array([SUPPORTS["U"].L, SUPPORTS["Q"].L, SUPPORTS["S"].L], dtype=np.float64)
    Us = np.array([SUPPORTS["U"].U, SUPPORTS["Q"].U, SUPPORTS["S"].U], dtype=np.float64)

    def _draw_row(mu_row: np.ndarray) -> np.ndarray:
        if size is None:
            out = np.zeros(3, dtype=np.float64)
            for k in range(3):
                out[k] = truncnorm_rvs(mu_row[k], sigmas[k], Ls[k], Us[k], size=1, rng=rng)[0]
            return out
        out = np.zeros((int(size), 3), dtype=np.float64)
        for k in range(3):
            out[:, k] = truncnorm_rvs(mu_row[k], sigmas[k], Ls[k], Us[k], size=int(size), rng=rng)
        return out

    if m.ndim == 1:
        return _draw_row(m)

    if m.ndim == 2:
        B = m.shape[0]
        if size is None:
            out = np.zeros((B, 3), dtype=np.float64)
            for i in range(B):
                out[i] = _draw_row(m[i])
            return out
        out = np.zeros((B, int(size), 3), dtype=np.float64)
        for i in range(B):
            out[i] = _draw_row(m[i])
        return out

    raise ValueError("mu must be shape (3,) or (B,3)")


def pit_u(
    y: np.ndarray,
    mu: np.ndarray,
    obs: ObservationParams,
) -> np.ndarray:
    """
    PIT under truncation: u = F_trunc(y).

    Input shapes:
      y and mu: either (3,) or (...,3) with order (U,Q,S)
    Output:
      u: same leading shape, last dim (3)
    """
    obs.validate()
    yy = np.asarray(y, dtype=np.float64)
    mm = np.asarray(mu, dtype=np.float64)
    if yy.shape != mm.shape:
        raise ValueError("y and mu must have same shape")
    if yy.shape[-1] != 3:
        raise ValueError("last dimension must be 3 (U,Q,S)")

    sigmas = np.array([obs.sigma_u, obs.sigma_q, obs.sigma_s], dtype=np.float64)
    Ls = np.array([SUPPORTS["U"].L, SUPPORTS["Q"].L, SUPPORTS["S"].L], dtype=np.float64)
    Us = np.array([SUPPORTS["U"].U, SUPPORTS["Q"].U, SUPPORTS["S"].U], dtype=np.float64)

    out = np.zeros_like(yy, dtype=np.float64)
    for k in range(3):
        out[..., k] = truncnorm_cdf(yy[..., k], mm[..., k], sigmas[k], Ls[k], Us[k])
    return out
