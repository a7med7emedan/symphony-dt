"""symphony-dt: Bayesian hierarchical state-space digital twins for longitudinal clinical cohorts.

Importing the package prints nothing and loads no heavy dependency. Subpackages load on first use:
`symphony_dt.data`, `symphony_dt.io`, `symphony_dt.models`, `symphony_dt.fit`, `symphony_dt.diagnostics`,
`symphony_dt.forecast`, `symphony_dt.evaluate`, `symphony_dt.calibration`, `symphony_dt.governance`,
`symphony_dt.report`, `symphony_dt.plots`. Call `symphony_dt.about()` for the version and citation.
"""

from __future__ import annotations

import importlib
from typing import Any

from symphony_dt._version import __version__

__all__ = ["__version__", "about", "validation_status"]

_SUBMODULES = {
    "data",
    "io",
    "models",
    "fit",
    "diagnostics",
    "forecast",
    "evaluate",
    "calibration",
    "governance",
    "report",
    "plots",
    "cli",
    "constants",
    "config",
    "labels",
    "errors",
}


def about(file: Any = None) -> str:
    """Print and return the logo, version, author, citation and documentation address.

    Validation: TESTED.
    """
    from symphony_dt.about import about as _about

    return _about(file=file)


def validation_status() -> dict[str, str]:
    """Return the validation label of every public function and class.

    Validation: TESTED.
    """
    from symphony_dt.labels import validation_status as _status

    return _status()


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f"symphony_dt.{name}")
    raise AttributeError(f"module 'symphony_dt' has no attribute {name!r}")
