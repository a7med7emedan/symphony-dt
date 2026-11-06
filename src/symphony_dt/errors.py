"""Exceptions and warnings raised by symphony-dt.

Every package error subclasses `SymphonyError`, so a caller can catch the whole family at once.
"""

from __future__ import annotations

__all__ = [
    "CmdStanNotFoundError",
    "DataValidationError",
    "GateError",
    "MappingError",
    "ModelIntegrityError",
    "SymphonyError",
    "SymphonyWarning",
    "UnfrozenSpecificationError",
]


class SymphonyError(Exception):
    """Base class of every error raised by symphony-dt."""


class SymphonyWarning(UserWarning):
    """Base class of every warning issued by symphony-dt."""


class DataValidationError(SymphonyError, ValueError):
    """A dataset failed schema validation.

    The `code` attribute names the rule that failed, for example `DV003`.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.detail = message


class ModelIntegrityError(SymphonyError):
    """A frozen model file does not match its recorded sha256."""


class GateError(SymphonyError):
    """A convergence gate could not be evaluated, or a gate override is malformed."""


class UnfrozenSpecificationError(SymphonyError):
    """Scoring was requested against a specification that is not frozen, or has changed."""


class MappingError(SymphonyError, ValueError):
    """An adapter mapping file is malformed or does not match the input."""


class CmdStanNotFoundError(SymphonyError):
    """CmdStan is not installed or cannot be found."""
