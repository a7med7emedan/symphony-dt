"""Validation labels for every public function and class.

Three labels exist. VALIDATED marks the pd_v2 model and the evaluation code of record, as used in
the SYMPHONY v2 study and ported with golden tests. TESTED marks code that is correct on synthetic
data and unit tests but has not been validated on a clinical cohort. EXPERIMENTAL marks code whose
design is not yet settled.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

__all__ = ["ValidationLabel", "registry", "validation", "validation_status"]

T = TypeVar("T")


class ValidationLabel(str, Enum):
    """The validation status of a public function or class.

    Validation: TESTED.
    """

    VALIDATED = "VALIDATED"
    TESTED = "TESTED"
    EXPERIMENTAL = "EXPERIMENTAL"


_REGISTRY: dict[str, ValidationLabel] = {}


def validation(label: ValidationLabel | str) -> Callable[[T], T]:
    """Attach a validation label to a function or class and record it in the registry.

    The docstring of the decorated object must state the same label as `Validation: <LABEL>.`.
    A test checks the two agree for every public object.

    Validation: TESTED.
    """
    value = ValidationLabel(label)

    def decorate(obj: T) -> T:
        target: Any = obj
        target.__validation__ = value
        name = f"{target.__module__}.{target.__qualname__}"
        _REGISTRY[name] = value
        return obj

    return decorate


def registry() -> dict[str, ValidationLabel]:
    """Return a copy of the label registry, keyed by qualified name.

    Validation: TESTED.
    """
    return dict(_REGISTRY)


def validation_status() -> dict[str, str]:
    """Import every public module and return the label of each public object.

    Validation: TESTED.
    """
    import importlib

    for module in (
        "symphony_dt.data",
        "symphony_dt.io",
        "symphony_dt.models",
        "symphony_dt.fit",
        "symphony_dt.diagnostics",
        "symphony_dt.forecast",
        "symphony_dt.evaluate",
        "symphony_dt.calibration",
        "symphony_dt.governance",
        "symphony_dt.report",
        "symphony_dt.plots",
        "symphony_dt.about",
    ):
        importlib.import_module(module)
    return {name: label.value for name, label in sorted(_REGISTRY.items())}
