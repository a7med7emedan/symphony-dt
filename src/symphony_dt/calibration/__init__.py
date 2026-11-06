"""Calibration: simulation-based calibration and posterior predictive checks."""

from symphony_dt.calibration.ppc import boundary_statistics, pit_values, ppc_coverage, ppc_summary
from symphony_dt.calibration.sbc import (
    CHI2_ALPHA,
    N_BINS,
    RANK_DRAWS,
    SBCDesign,
    SBCResult,
    SBCVerdict,
    chi_squared_uniformity,
    judge,
    ks_uniformity,
    pd_v2_prior_design,
    rank_histogram_bands,
    run_sbc,
    sbc_rank,
    thin_for_rank,
)

__all__ = [
    "CHI2_ALPHA",
    "N_BINS",
    "RANK_DRAWS",
    "SBCDesign",
    "SBCResult",
    "SBCVerdict",
    "boundary_statistics",
    "chi_squared_uniformity",
    "judge",
    "ks_uniformity",
    "pd_v2_prior_design",
    "pit_values",
    "ppc_coverage",
    "ppc_summary",
    "rank_histogram_bands",
    "run_sbc",
    "sbc_rank",
    "thin_for_rank",
]
