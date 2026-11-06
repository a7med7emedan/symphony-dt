"""Harmonisation hooks for instrument versions and units.

No crosswalk ships with the package. A crosswalk is a user-supplied conversion between two versions
of an instrument, and it enters only with the citation that justifies it. Unit converters work the
same way for exposures and covariates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from symphony_dt.data.dataset import TwinDataset
from symphony_dt.labels import validation

__all__ = ["Crosswalk", "UnitConverter", "convert_units", "harmonise"]


@validation("EXPERIMENTAL")
@dataclass(frozen=True)
class Crosswalk:
    """A cited conversion of one instrument from a source version to a target version.

    `convert` maps an array of source scores to target scores. `citation` must name the published
    source of the conversion; an empty citation raises ValueError.

    Validation: EXPERIMENTAL.
    """

    endpoint: str
    source_version: str
    target_version: str
    convert: Callable[[np.ndarray], np.ndarray]
    citation: str

    def __post_init__(self) -> None:
        if not self.citation.strip():
            raise ValueError(
                f"the crosswalk for {self.endpoint} {self.source_version} to {self.target_version} "
                "has no citation; an unpublished crosswalk may not enter the package"
            )


@validation("EXPERIMENTAL")
def harmonise(dataset: TwinDataset, crosswalks: Iterable[Crosswalk]) -> TwinDataset:
    """Apply crosswalks to observations whose `instrument_version` matches a crosswalk source.

    The converted rows take the target version. The citations used are recorded in
    `metadata["harmonisation"]`. Observations without `instrument_version` are left unchanged.

    Validation: EXPERIMENTAL.
    """
    obs = dataset.observations.copy()
    applied: list[dict[str, str]] = []
    if "instrument_version" in obs.columns:
        for cw in crosswalks:
            rows = (obs["endpoint"] == cw.endpoint) & (
                obs["instrument_version"].astype(str) == cw.source_version
            )
            if rows.any():
                obs.loc[rows, "value"] = cw.convert(obs.loc[rows, "value"].to_numpy(float))
                obs.loc[rows, "instrument_version"] = cw.target_version
                applied.append(
                    {
                        "endpoint": cw.endpoint,
                        "from": cw.source_version,
                        "to": cw.target_version,
                        "citation": cw.citation,
                        "rows": str(int(rows.sum())),
                    }
                )
    metadata = {**dataset.metadata, "harmonisation": applied}
    return TwinDataset(
        dataset.participants, obs, dataset.visits, dataset.markers, dataset.instruments, metadata
    )


@validation("TESTED")
@dataclass(frozen=True)
class UnitConverter:
    """A unit conversion for a visit-level exposure column, for example a dose in other units.

    Validation: TESTED.
    """

    column: str
    factor: float
    source_unit: str
    target_unit: str


@validation("TESTED")
def convert_units(dataset: TwinDataset, converters: Iterable[UnitConverter]) -> TwinDataset:
    """Multiply visit-level columns by their conversion factors and record the conversion.

    Validation: TESTED.
    """
    if dataset.visits is None:
        return dataset
    visits = dataset.visits.copy()
    done = []
    for conv in converters:
        if conv.column not in visits.columns:
            raise KeyError(f"visits has no column {conv.column!r}")
        visits[conv.column] = pd.to_numeric(visits[conv.column], errors="coerce") * conv.factor
        done.append(f"{conv.column}: {conv.source_unit} to {conv.target_unit} x {conv.factor}")
    metadata = {**dataset.metadata, "unit_conversions": done}
    return TwinDataset(
        dataset.participants, dataset.observations, visits, dataset.markers, dataset.instruments, metadata
    )
