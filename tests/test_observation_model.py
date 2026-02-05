import numpy as np

from symphony_dt.model.observation import ObservationParams, SUPPORTS, mu_from_x, pit_u, sample_observations


def _default_obs() -> ObservationParams:
    # Reasonable defaults consistent with priors (b>0, sigma>0)
    return ObservationParams(
        a_u=0.0, b_u=1.0,
        a_q=0.0, b_q=1.0,
        a_s=0.0, b_s=1.0,
        sigma_u=5.0, sigma_q=1.5, sigma_s=3.0,
    )


def test_mu_within_supports_and_directions() -> None:
    obs = _default_obs()
    # Compare low vs high severity in each domain
    x_low = np.array([0.1, 0.1, 0.1])
    x_high = np.array([0.9, 0.9, 0.9])

    mu_low = mu_from_x(x_low, obs)
    mu_high = mu_from_x(x_high, obs)

    # Within bounds
    assert SUPPORTS["U"].L <= mu_low[0] <= SUPPORTS["U"].U
    assert SUPPORTS["Q"].L <= mu_low[1] <= SUPPORTS["Q"].U
    assert SUPPORTS["S"].L <= mu_low[2] <= SUPPORTS["S"].U

    assert SUPPORTS["U"].L <= mu_high[0] <= SUPPORTS["U"].U
    assert SUPPORTS["Q"].L <= mu_high[1] <= SUPPORTS["Q"].U
    assert SUPPORTS["S"].L <= mu_high[2] <= SUPPORTS["S"].U

    # Directionality per Methods:
    # UPDRS increases with motor severity
    assert mu_high[0] > mu_low[0]
    # MoCA decreases with cognitive severity
    assert mu_high[1] < mu_low[1]
    # SCOPA-AUT increases with non-motor severity
    assert mu_high[2] > mu_low[2]


def test_sampling_in_bounds_and_pit_range() -> None:
    obs = _default_obs()
    rng = np.random.default_rng(0)
    x = np.array([0.6, 0.4, 0.7])
    mu = mu_from_x(x, obs)

    y = sample_observations(mu, obs, rng=rng)
    assert SUPPORTS["U"].L <= y[0] <= SUPPORTS["U"].U
    assert SUPPORTS["Q"].L <= y[1] <= SUPPORTS["Q"].U
    assert SUPPORTS["S"].L <= y[2] <= SUPPORTS["S"].U

    u = pit_u(y, mu, obs)
    assert np.all(u >= 0.0)
    assert np.all(u <= 1.0)
