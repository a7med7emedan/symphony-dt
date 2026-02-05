from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


def sign_probability(samples: np.ndarray) -> float:
    """Compute Pr(T > 0) empirically from posterior samples."""
    s = np.asarray(samples, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float(np.mean(s > 0))


@dataclass(frozen=True)
class ExplanationGateConfig:
    sign_prob_threshold: float = 0.95


def should_report_explanation(samples: np.ndarray, cfg: ExplanationGateConfig = ExplanationGateConfig()) -> bool:
    p = sign_probability(samples)
    return bool(np.isfinite(p) and p >= cfg.sign_prob_threshold)
