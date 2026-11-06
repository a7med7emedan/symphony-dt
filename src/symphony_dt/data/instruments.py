"""The instrument registry: scale range, direction of worsening and boundary handling per instrument."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from symphony_dt.labels import validation

__all__ = ["Instrument", "InstrumentRegistry", "default_registry"]


@validation("TESTED")
class Instrument(BaseModel):
    """One clinical instrument.

    `worsening` says whether higher scores mean worse disease (`increase`) or better (`decrease`).
    `boundary` says how the model treats values at the scale ends: `truncated` or `censored`.

    Validation: TESTED.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str = "1"
    lower: float
    upper: float
    worsening: Literal["increase", "decrease"]
    boundary: Literal["truncated", "censored"] = "truncated"
    integer: bool = True
    citation: str = ""

    @model_validator(mode="after")
    def _range(self) -> Instrument:
        if not self.upper > self.lower:
            raise ValueError(f"instrument {self.name}: upper {self.upper} must exceed lower {self.lower}")
        return self

    def contains(self, value: float) -> bool:
        """Return True when `value` lies inside the scale range, ends included.

        Validation: TESTED.
        """
        return self.lower <= value <= self.upper


@validation("TESTED")
class InstrumentRegistry:
    """A name-keyed collection of instruments.

    Validation: TESTED.
    """

    def __init__(self, instruments: Iterable[Instrument] = ()) -> None:
        self._items: dict[str, Instrument] = {}
        for instrument in instruments:
            self.add(instrument)

    def add(self, instrument: Instrument) -> None:
        """Register an instrument. A second instrument with the same name raises ValueError.

        Validation: TESTED.
        """
        if instrument.name in self._items:
            raise ValueError(f"instrument {instrument.name} is already registered")
        self._items[instrument.name] = instrument

    def get(self, name: str) -> Instrument:
        """Return the instrument called `name`, or raise KeyError naming the known instruments.

        Validation: TESTED.
        """
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items))
            raise KeyError(f"unknown instrument {name!r}; registered: {known}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[Instrument]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def names(self) -> tuple[str, ...]:
        """Return the registered names in insertion order.

        Validation: TESTED.
        """
        return tuple(self._items)


@validation("TESTED")
def default_registry() -> InstrumentRegistry:
    """Return the three instruments of the pd_v2 model with their published scale ranges.

    MDS-UPDRS Part III runs 0 to 132 and worsens upwards. MoCA runs 0 to 30 and worsens downwards.
    SCOPA-AUT is prorated to the 23-item scale, 0 to 69, and worsens upwards. Boundary handling
    follows the pd_v2 primary arm: MoCA censored, the other two truncated.

    Validation: TESTED.
    """
    return InstrumentRegistry(
        [
            Instrument(
                name="updrs3",
                version="MDS-UPDRS 2008",
                lower=0,
                upper=132,
                worsening="increase",
                boundary="truncated",
                citation="Goetz CG et al. Mov Disord. 2008;23(15):2129-2170.",
            ),
            Instrument(
                name="moca",
                version="8.1",
                lower=0,
                upper=30,
                worsening="decrease",
                boundary="censored",
                citation="Nasreddine ZS et al. J Am Geriatr Soc. 2005;53(4):695-699.",
            ),
            Instrument(
                name="scopa_aut",
                version="23-item prorated",
                lower=0,
                upper=69,
                worsening="increase",
                boundary="truncated",
                integer=False,
                citation="Visser M et al. Mov Disord. 2004;19(11):1306-1312.",
            ),
        ]
    )
