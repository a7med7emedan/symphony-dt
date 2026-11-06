"""Calibration monitoring and online recalibration for living twins.

`CalibrationMonitor` tests, after every outcome, whether the twin's one-step-ahead probability integral
transforms (PITs) are still uniform. Under a correctly specified sequential forecaster these PITs are independent
and uniform (Dawid's prequential principle), so any departure is evidence of miscalibration or drift. The test is
an e-process: a mixture of test martingales against bias and dispersion alternatives. By Ville's inequality the
probability that it ever exceeds 1/alpha under calibration is at most alpha, so it may be checked after every
visit with no correction (Arnold, Henzi and Ziegel; Vovk and Wang; Ramdas and colleagues).

`AdaptiveConformal` keeps the long-run coverage of the twin's intervals at target by moving the interval level
online after each outcome (Gibbs and Candes, 2021).

`RelearnPolicy` decides when the population model must be refitted: on monitor evidence, or after enough new
outcomes have accumulated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from symphony_dt.labels import validation

__all__ = ["AdaptiveConformal", "CalibrationMonitor", "RelearnDecision", "RelearnPolicy"]

# Alternatives: f(u) = 1 + lam * g(u), with g integrating to zero on [0, 1] and f non-negative.
# Bias: g = 2u - 1, valid for |lam| <= 1. Dispersion: g = 3(2u - 1)^2 - 1 in [-1, 2], valid for -0.5 <= lam <= 1.
_BIAS = (-0.6, -0.3, 0.3, 0.6)
_DISP = (-0.4, -0.2, 0.3, 0.6)


def _alternatives(u: float) -> np.ndarray:
    b = 2 * u - 1
    d = 3 * b * b - 1
    return np.concatenate([1 + np.asarray(_BIAS) * b, 1 + np.asarray(_DISP) * d])


@validation("EXPERIMENTAL")
@dataclass
class CalibrationMonitor:
    """An anytime-valid e-process monitor of PIT uniformity, overall and per endpoint.

    `update(pit, endpoint)` adds one PIT. `evidence` is the current e-value of the mixture; `alarm` is true once
    it has reached 1/alpha. Each stream starts at e = 1. The mixture weights are uniform over eight bias and
    dispersion alternatives, fixed before any data, which keeps the mixture a test martingale.

    Validation: EXPERIMENTAL.
    """

    alpha: float = 0.05
    log_wealth: dict[str, np.ndarray] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    alarms: dict[str, int | None] = field(default_factory=dict)
    peak: dict[str, float] = field(default_factory=dict)

    def _stream(self, name: str) -> np.ndarray:
        if name not in self.log_wealth:
            self.log_wealth[name] = np.zeros(len(_BIAS) + len(_DISP))
            self.counts[name] = 0
            self.alarms[name] = None
            self.peak[name] = 1.0
        return self.log_wealth[name]

    def update(self, pit: float, endpoint: str | None = None) -> float:
        """Add one PIT value to the overall stream and to its endpoint stream; return the overall e-value.

        Validation: EXPERIMENTAL.
        """
        u = float(np.clip(pit, 1e-9, 1 - 1e-9))
        for name in ("all", *([endpoint] if endpoint else [])):
            lw = self._stream(name)
            lw += np.log(_alternatives(u))
            self.counts[name] += 1
            e = self.evidence(name)
            self.peak[name] = max(self.peak[name], e)
            if self.alarms[name] is None and e >= 1 / self.alpha:
                self.alarms[name] = self.counts[name]
        return self.evidence("all")

    def evidence(self, stream: str = "all") -> float:
        """Return the e-value of a stream: the average wealth over the alternatives.

        Validation: EXPERIMENTAL.
        """
        lw = self._stream(stream)
        m = float(lw.max())
        return float(np.exp(m) * np.mean(np.exp(lw - m)))

    def alarm(self, stream: str = "all") -> bool:
        """True once the stream's e-value has reached 1/alpha at any time.

        Validation: EXPERIMENTAL.
        """
        self._stream(stream)
        return self.alarms[stream] is not None

    def state(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of every stream.

        Validation: EXPERIMENTAL.
        """
        return {
            name: {
                "n": self.counts[name],
                "evidence": self.evidence(name),
                "peak": self.peak[name],
                "alarm_at": self.alarms[name],
                "threshold": 1 / self.alpha,
            }
            for name in sorted(self.log_wealth)
        }

    def reset(self, stream: str | None = None) -> None:
        """Restart one stream, or all streams, at e = 1, for example after relearning.

        Validation: EXPERIMENTAL.
        """
        names = [stream] if stream else list(self.log_wealth)
        for name in names:
            self.log_wealth.pop(name, None)
            self._stream(name)


@validation("EXPERIMENTAL")
@dataclass
class AdaptiveConformal:
    """Adaptive conformal inference on the twin's central intervals, one level per endpoint.

    After each outcome, `alpha_t` moves by `gamma * (target - miss)`: a miss lowers alpha (wider intervals) and
    a hit raises it. Over time the miss rate approaches the target whatever the data do.

    Validation: EXPERIMENTAL.
    """

    target: float = 0.1
    gamma: float = 0.01
    level: dict[str, float] = field(default_factory=dict)
    misses: dict[str, list[int]] = field(default_factory=dict)

    def alpha(self, endpoint: str) -> float:
        """Return the current miscoverage level for an endpoint, clipped to [0.001, 0.999].

        Validation: EXPERIMENTAL.
        """
        return float(np.clip(self.level.get(endpoint, self.target), 0.001, 0.999))

    def interval(self, samples: np.ndarray, endpoint: str) -> tuple[float, float]:
        """Return the central interval of `samples` at the endpoint's current level.

        Validation: EXPERIMENTAL.
        """
        a = self.alpha(endpoint)
        lo, hi = np.quantile(np.asarray(samples, float), [a / 2, 1 - a / 2])
        return float(lo), float(hi)

    def update(self, samples: np.ndarray, y: float, endpoint: str) -> bool:
        """Score one outcome against the current interval, move the level, and return whether it was covered.

        Validation: EXPERIMENTAL.
        """
        lo, hi = self.interval(samples, endpoint)
        miss = int(not lo <= y <= hi)
        self.level[endpoint] = self.level.get(endpoint, self.target) + self.gamma * (self.target - miss)
        self.misses.setdefault(endpoint, []).append(miss)
        return not miss

    def state(self) -> dict[str, Any]:
        """Return the current levels and the realised miss rates.

        Validation: EXPERIMENTAL.
        """
        return {
            e: {"alpha": self.alpha(e), "miss_rate": float(np.mean(m)) if m else None, "n": len(m)}
            for e, m in self.misses.items()
        }


@validation("EXPERIMENTAL")
@dataclass(frozen=True)
class RelearnDecision:
    """Whether to refit the population model, and why.

    Validation: EXPERIMENTAL.
    """

    refit: bool
    reason: str
    evidence: float
    new_outcomes: int


@validation("EXPERIMENTAL")
@dataclass(frozen=True)
class RelearnPolicy:
    """Refit when the calibration monitor alarms, or when `max_new_outcomes` outcomes have arrived since the fit.

    The monitor's alarm carries anytime-valid error control: a calibrated twin triggers it with probability at
    most alpha over its whole life. The count rule refreshes the population as data accumulate, with no claim
    of miscalibration.

    Validation: EXPERIMENTAL.
    """

    max_new_outcomes: int = 500
    streams: tuple[str, ...] = ("all",)

    def decide(self, monitor: CalibrationMonitor, new_outcomes: int) -> RelearnDecision:
        """Return the decision for the current monitor state and count of new outcomes.

        Validation: EXPERIMENTAL.
        """
        evidence = max(monitor.evidence(s) for s in self.streams)
        for s in self.streams:
            if monitor.alarm(s):
                return RelearnDecision(
                    True,
                    f"calibration alarm on stream {s!r} (e >= {1 / monitor.alpha:g})",
                    evidence,
                    new_outcomes,
                )
        if new_outcomes >= self.max_new_outcomes:
            return RelearnDecision(True, f"{new_outcomes} new outcomes since the last fit", evidence, new_outcomes)
        return RelearnDecision(False, "calibrated and below the refresh count", evidence, new_outcomes)
