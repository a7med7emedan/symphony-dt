"""The scoring specification and its pre-registration freeze.

A specification is frozen by hashing its canonical JSON. Scoring refuses an unfrozen specification,
and refuses a specification whose hash no longer matches its freeze, unless a deviation record
names the change, the reason and the author. The record travels with the scores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from symphony_dt.errors import UnfrozenSpecificationError
from symphony_dt.labels import validation

__all__ = ["DeviationRecord", "FrozenSpec", "ScoringSpec", "freeze_spec", "require_frozen"]

_INF = 1e300  # JSON has no infinity; this stands for it in the canonical form.


@validation("TESTED")
class ScoringSpec(BaseModel):
    """Every choice the scoring makes, fixed before any target is scored.

    `ScoringSpec.default()` reproduces the SYMPHONY v2 code of record: anchor 2.0 years, nine
    horizon bins, 30 residuals per bin, windows next, 1y, 2y and all, events of +5 UPDRS-III and
    -2 MoCA, B = 1000 bootstrap replicates and no cell from fewer than 10 participants.

    Validation: TESTED.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "SCORING_SPEC_2026-10-21"
    anchor_years: float = 2.0
    bins: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, float("inf"))
    min_bin: int = 30
    windows: dict[str, tuple[float, float] | None] = {
        "next": None,
        "1y": (0.75, 1.25),
        "2y": (1.75, 2.25),
        "all": (float("-inf"), float("inf")),
    }
    event_windows: dict[str, tuple[float, float]] = {"2y": (1.75, 2.25), "1y": (0.75, 1.25)}
    events: dict[int, tuple[str, float]] = {1: ("up", 5.0), 2: ("down", 2.0)}
    endpoints: dict[int, str] = {1: "UPDRS-III", 2: "MoCA", 3: "SCOPA-AUT"}
    levels: tuple[float, float] = (0.5, 0.95)
    bootstrap_replicates: int = 1000
    seed_bootstrap: int = 20261021
    seed_lmm: int = 20261022
    lmm_samples: int = 2000
    min_cell_participants: int = 10

    @classmethod
    def default(cls) -> ScoringSpec:
        """Return the code-of-record specification.

        Validation: TESTED.
        """
        return cls()

    def canonical(self) -> str:
        """Return the canonical JSON used for hashing: sorted keys, infinities as 1e300.

        Validation: TESTED.
        """

        def clean(value: Any) -> Any:
            if isinstance(value, float):
                if value == float("inf"):
                    return _INF
                if value == float("-inf"):
                    return -_INF
                return value
            if isinstance(value, dict):
                return {str(k): clean(v) for k, v in value.items()}
            if isinstance(value, list | tuple):
                return [clean(v) for v in value]
            return value

        return json.dumps(clean(self.model_dump()), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        """Return the sha256 of the canonical JSON.

        Validation: TESTED.
        """
        return hashlib.sha256(self.canonical().encode()).hexdigest()


@validation("TESTED")
@dataclass(frozen=True)
class FrozenSpec:
    """A specification hash with the time it was frozen.

    Validation: TESTED.
    """

    sha256: str
    frozen_at: str
    name: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form.

        Validation: TESTED.
        """
        return dict(self.__dict__)


@validation("TESTED")
@dataclass(frozen=True)
class DeviationRecord:
    """An explicit, signed record of a change to a frozen specification.

    Validation: TESTED.
    """

    reason: str
    author: str
    changes: dict[str, Any] = field(default_factory=dict)
    recorded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if not self.reason.strip() or not self.author.strip():
            raise ValueError("a deviation record needs a reason and an author")


@validation("TESTED")
def freeze_spec(spec: ScoringSpec, path: str | Path | None = None) -> FrozenSpec:
    """Freeze a specification: hash it and, when `path` is given, write spec and hash to disk.

    The file holds the canonical specification and its sha256, so a reader can check the freeze
    without the package.

    Validation: TESTED.
    """
    frozen = FrozenSpec(
        sha256=spec.sha256(),
        frozen_at=datetime.now(UTC).isoformat(),
        name=spec.name,
        path=str(path) if path is not None else None,
    )
    if path is not None:
        payload = {"specification": json.loads(spec.canonical()), **frozen.to_dict()}
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return frozen


@validation("TESTED")
def load_frozen_spec(path: str | Path) -> tuple[ScoringSpec, FrozenSpec]:
    """Read a frozen specification file and check its hash.

    Validation: TESTED.
    """
    payload = json.loads(Path(path).read_text())
    raw = payload["specification"]

    def restore(value: Any) -> Any:
        if isinstance(value, float) and abs(value) >= _INF:
            return float("inf") if value > 0 else float("-inf")
        if isinstance(value, list):
            return [restore(v) for v in value]
        if isinstance(value, dict):
            return {k: restore(v) for k, v in value.items()}
        return value

    data = restore(raw)
    data["events"] = {int(k): tuple(v) for k, v in data["events"].items()}
    data["endpoints"] = {int(k): v for k, v in data["endpoints"].items()}
    spec = ScoringSpec(**data)
    frozen = FrozenSpec(sha256=payload["sha256"], frozen_at=payload["frozen_at"], name=payload["name"], path=str(path))
    if spec.sha256() != frozen.sha256:
        raise UnfrozenSpecificationError(f"{path}: the stored specification does not match its hash")
    return spec, frozen


@validation("TESTED")
def require_frozen(
    spec: ScoringSpec, frozen: FrozenSpec | None, deviation: DeviationRecord | None = None
) -> dict[str, Any]:
    """Check `spec` against its freeze and return the record that travels with the scores.

    Raises UnfrozenSpecificationError when `frozen` is None, or when the hashes differ and no
    deviation record is given.

    Validation: TESTED.
    """
    if frozen is None:
        raise UnfrozenSpecificationError("scoring needs a frozen specification; call freeze_spec(spec) before scoring")
    current = spec.sha256()
    if current != frozen.sha256:
        if deviation is None:
            raise UnfrozenSpecificationError(
                f"the specification changed after it was frozen ({frozen.sha256[:12]} to "
                f"{current[:12]}); write a DeviationRecord to proceed"
            )
        return {"frozen_sha256": frozen.sha256, "used_sha256": current, "deviation": deviation.__dict__}
    return {"frozen_sha256": frozen.sha256, "used_sha256": current, "deviation": None}
