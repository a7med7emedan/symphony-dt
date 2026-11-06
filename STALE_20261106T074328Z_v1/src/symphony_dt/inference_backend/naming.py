from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ParameterNamingSpec:
    """
    Canonical naming spec for posterior draw arrays (Module 9 store) and downstream engines.

    This spec prevents drift between notebook variable names and package draw keys.
    In later modules, samplers must emit arrays using these keys.
    """
    # global parameters
    global_params: Tuple[str, ...]
    # per-patient parameters
    patient_params: Tuple[str, ...]
    # per-interval innovations (optional)
    innovation_params: Tuple[str, ...]


def default_naming_spec() -> ParameterNamingSpec:
    # Matches LaTeX symbol set; exact key names can be mapped from notebook in Module 26.
    global_params = (
        "mu_r",           # (L,3)
        "Gamma",          # (L,3,2)
        "A",              # (L,3,P)
        "C_free",         # (L,6) packed coupling entries
        "sigma_r",        # (L,3)
        "sigma_xi",       # (L,3)
        "a_u", "b_u", "a_q", "b_q", "a_s", "b_s",  # (L,)
        "sigma_u", "sigma_q", "sigma_s",           # (L,)
    )
    patient_params = (
        "s0",             # (L_train,N_train,3) scaffold; conditional will be (S,3)
        "r",              # (L_train,N_train,3)
    )
    innovation_params = (
        "epsilon",        # optional packed innovations per interval
    )
    return ParameterNamingSpec(global_params=global_params, patient_params=patient_params, innovation_params=innovation_params)
