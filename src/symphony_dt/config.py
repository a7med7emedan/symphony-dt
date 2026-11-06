"""Model configuration: the observation-model switches of pd_v2, named rather than written as literals."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from symphony_dt import constants as C
from symphony_dt.labels import validation

__all__ = ["ModelConfig"]


@validation("VALIDATED")
class ModelConfig(BaseModel):
    """The switches and grid settings of one pd_v2 arm.

    `censored` gives each endpoint the tobit density (1) or the truncated density (0).
    `state_dependent_sigma` is A01, `ledd_in_mu_U` is A02 and `sigma_xbar` is the A01 centring constant.
    Use `ModelConfig.primary()` for the study's primary arm.

    Validation: VALIDATED. The switch values of `primary()` and `a01_sensitivity()` are the study's.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    censored: tuple[int, int, int] = (
        C.CENSORED_PRIMARY["updrs3"],
        C.CENSORED_PRIMARY["moca"],
        C.CENSORED_PRIMARY["scopa_aut"],
    )
    state_dependent_sigma: int = C.STATE_DEPENDENT_SIGMA_PRIMARY
    ledd_in_mu_U: int = C.LEDD_IN_MU_U_PRIMARY
    sigma_xbar: tuple[float, float, float] = C.SIGMA_XBAR_PRIMARY
    prior: Literal["corrected", "published"] = "corrected"
    h_max_years: float = C.H_MAX_YEARS

    @field_validator("censored")
    @classmethod
    def _binary(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(v not in (0, 1) for v in value):
            raise ValueError(f"censored must hold 0 or 1 per endpoint, got {value}")
        return value

    @field_validator("state_dependent_sigma", "ledd_in_mu_U")
    @classmethod
    def _switch(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError(f"a switch must be 0 or 1, got {value}")
        return value

    @field_validator("sigma_xbar")
    @classmethod
    def _unit(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(not 0.0 <= v <= 1.0 for v in value):
            raise ValueError(f"sigma_xbar entries must lie in [0, 1], got {value}")
        return value

    @field_validator("h_max_years")
    @classmethod
    def _positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"h_max_years must be positive, got {value}")
        return value

    @classmethod
    def primary(cls) -> ModelConfig:
        """The study's primary arm: A01 off, A02 on, A03 on for MoCA only.

        Validation: VALIDATED.
        """
        return cls()

    @classmethod
    def a01_sensitivity(cls) -> ModelConfig:
        """The study's pre-declared sensitivity arm: the primary arm with A01 switched on.

        Validation: VALIDATED.
        """
        return cls(state_dependent_sigma=C.STATE_DEPENDENT_SIGMA_SENSITIVITY)

    def hyperparameters(self) -> dict[str, Any]:
        """Return the prior hyperparameters this configuration selects.

        Validation: VALIDATED.
        """
        hyper = C.PRIOR_CORRECTED if self.prior == "corrected" else C.PRIOR_PUBLISHED
        return dict(hyper)

    def switches(self) -> dict[str, Any]:
        """Return the switch entries of the Stan data block.

        Validation: VALIDATED.
        """
        return {
            "censored": list(self.censored),
            "state_dependent_sigma": int(self.state_dependent_sigma),
            "ledd_in_mu_U": int(self.ledd_in_mu_U),
            "sigma_xbar": [float(v) for v in self.sigma_xbar],
        }
