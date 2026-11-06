"""Posterior storage as netCDF through ArviZ InferenceData."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from symphony_dt.labels import validation

__all__ = ["read_posterior", "write_posterior"]


@validation("TESTED")
def write_posterior(idata: Any, path: str | Path) -> Path:
    """Write an ArviZ InferenceData to netCDF and return the path.

    Validation: TESTED.
    """
    target = Path(path)
    idata.to_netcdf(str(target), engine="h5netcdf")
    return target


@validation("TESTED")
def read_posterior(path: str | Path) -> Any:
    """Read an ArviZ InferenceData from netCDF.

    Validation: TESTED.
    """
    import arviz as az

    return az.from_netcdf(str(path), engine="h5netcdf")
