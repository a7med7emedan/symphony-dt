"""Constants of the pd_v2 model of record, fixed in one place.

The values come from the SYMPHONY v2 Stan data builder. The endpoint order is load-bearing: the
Stan program keys the MoCA sign flip on endpoint index 2, so the order must never change.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

__all__ = [
    "AGE_DIVISOR_YEARS",
    "CENSORED_PRIMARY",
    "DAYS_PER_YEAR",
    "ENDPOINTS",
    "ENDPOINT_INDEX",
    "H_MAX_YEARS",
    "LEDD_IN_MU_U_PRIMARY",
    "MASTER_SEED",
    "PRIOR_CORRECTED",
    "PRIOR_PUBLISHED",
    "PRODUCTION_GRAINSIZE",
    "SCALE_BOUNDS",
    "SEX_LABELS",
    "SIGMA_XBAR_PRIMARY",
    "STATE_DEPENDENT_SIGMA_PRIMARY",
    "STATE_DEPENDENT_SIGMA_SENSITIVITY",
]

#: Endpoint order in every Stan array: MDS-UPDRS Part III, MoCA, SCOPA-AUT.
ENDPOINTS: Final[tuple[str, str, str]] = ("updrs3", "moca", "scopa_aut")

#: One-based Stan index of each endpoint.
ENDPOINT_INDEX: Final[Mapping[str, int]] = MappingProxyType({name: k + 1 for k, name in enumerate(ENDPOINTS)})

#: Scale support of each instrument. SCOPA-AUT is prorated to the 23-item scale, so its support is 0 to 69.
SCALE_BOUNDS: Final[Mapping[str, tuple[float, float]]] = MappingProxyType(
    {"updrs3": (0.0, 132.0), "moca": (0.0, 30.0), "scopa_aut": (0.0, 69.0)}
)

#: Days per year used to convert integer days to years.
DAYS_PER_YEAR: Final[float] = 365.25

#: Maximum sub-step of the transition grid, in years. One kernel serves fitting and forecasting.
H_MAX_YEARS: Final[float] = 0.5

#: Centred age is expressed in decades.
AGE_DIVISOR_YEARS: Final[float] = 10.0

#: The corrected prior hyperparameters, calibrated by prior predictive checks.
PRIOR_CORRECTED: Final[Mapping[str, Any]] = MappingProxyType(
    {"mu_r_mean": -2.6, "mu_r_sd": 0.6, "a_mean": (-2.0, -2.2, -1.8), "a_sd": 0.75}
)

#: The published prior hyperparameters, kept so both priors can be run from one source.
PRIOR_PUBLISHED: Final[Mapping[str, Any]] = MappingProxyType(
    {"mu_r_mean": 0.0, "mu_r_sd": 1.0, "a_mean": (0.0, 0.0, 0.0), "a_sd": 1.0}
)

#: A03, the primary censoring switch per endpoint: MoCA censored, the other two truncated.
CENSORED_PRIMARY: Final[Mapping[str, int]] = MappingProxyType({"updrs3": 0, "moca": 1, "scopa_aut": 0})

#: A01 off is the primary arm.
STATE_DEPENDENT_SIGMA_PRIMARY: Final[int] = 0

#: A01 on is the pre-declared sensitivity arm.
STATE_DEPENDENT_SIGMA_SENSITIVITY: Final[int] = 1

#: A02 on: LEDD enters the UPDRS-III observation mean only.
LEDD_IN_MU_U_PRIMARY: Final[int] = 1

#: A01 centring constant, the prior predictive mean of the latent state, written as a literal.
SIGMA_XBAR_PRIMARY: Final[tuple[float, float, float]] = (0.53, 0.54, 0.55)

#: Production grainsize. With reduce_sum_static, byte identity holds only at a fixed grainsize.
PRODUCTION_GRAINSIZE: Final[int] = 64

#: Master seed of the study.
MASTER_SEED: Final[int] = 20260729

#: Sex code book: 0 female, 1 male.
SEX_LABELS: Final[Mapping[int, str]] = MappingProxyType({0: "Female", 1: "Male"})
