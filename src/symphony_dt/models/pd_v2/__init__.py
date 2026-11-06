"""pd_v2, the SYMPHONY v2 model of record, frozen.

`model.stan` is byte-identical to the analysis repository's `symphony_population.stan`. Its sha256
is checked when this module is imported; a mismatch raises ModelIntegrityError and nothing runs.
`conditional.stan` is byte-identical to `symphony_conditional.stan`. `forecast_gq.stan` is a new
generated-quantities program written for the package.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from symphony_dt.errors import ModelIntegrityError

__all__ = ["CARD", "CONDITIONAL_SHA256", "FORECAST_GQ_SHA256", "HERE", "MODEL_SHA256", "verify"]

HERE = Path(__file__).resolve().parent

#: sha256 of model.stan, the model of record. Never edit the file; never edit this value.
MODEL_SHA256 = "061c1a770c14c13ba9e2e0728938142279949bfd0091c061e0f5b16b75b5c42c"

#: sha256 of conditional.stan, byte-identical to the analysis repository's conditional program.
CONDITIONAL_SHA256 = "466fde1a32c81dce28cb1706825cd4c53f10562b8d0316fc7b49ccc6365d5d6f"

#: sha256 of forecast_gq.stan, the package's generated-quantities program.
FORECAST_GQ_SHA256 = "f06b1b0ba0d08a1069881b7aa375d8681b96ce04547071221f8529d1f58d0a41"

CARD = HERE / "card.md"

_EXPECTED = {
    "model.stan": MODEL_SHA256,
    "conditional.stan": CONDITIONAL_SHA256,
    "forecast_gq.stan": FORECAST_GQ_SHA256,
}


def verify() -> dict[str, str]:
    """Check every pd_v2 Stan file against its recorded sha256 and return the digests.

    Raises ModelIntegrityError naming the file when a digest differs.
    """
    found: dict[str, str] = {}
    for name, expected in _EXPECTED.items():
        digest = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ModelIntegrityError(
                f"pd_v2/{name} has sha256 {digest}, expected {expected}. The frozen model files "
                "must never be edited; reinstall symphony-dt."
            )
        found[name] = digest
    return found


verify()
