from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class InferenceBackend(str, Enum):
    PYMC = "pymc"
    NUMPYRO = "numpyro"
    STAN = "stan"
    PYRO = "pyro"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class InferenceBackendConfig:
    backend: InferenceBackend
    notes: str
    evidence: Dict[str, int]  # token -> count


_KEYWORDS: Dict[InferenceBackend, Tuple[str, ...]] = {
    InferenceBackend.PYMC: ("import pymc", "pymc as pm", "pm.sample(", "pm.Model(", "pytensor", "aesara"),
    InferenceBackend.NUMPYRO: ("import numpyro", "numpyro.infer", "NUTS(", "MCMC(", "numpyro.sample", "jax.numpy"),
    InferenceBackend.STAN: ("cmdstanpy", "pystan", "stan.build", "CmdStanModel", "sampling(", "cmdstan"),
    InferenceBackend.PYRO: ("import pyro", "pyro.infer", "pyro.sample", "pyro.contrib"),
}


def _extract_code_cells(nb: dict) -> List[str]:
    cells = nb.get("cells", [])
    out: List[str] = []
    for c in cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", [])
        if isinstance(src, str):
            out.append(src)
        else:
            out.append("".join(map(str, src)))
    return out


def detect_backend_from_notebook(notebook_path: Path) -> InferenceBackendConfig:
    """
    Deterministically detect likely inference backend used by a reference notebook.
    Uses keyword counts over code cells; returns UNKNOWN if no evidence.

    This is used to lock backend choice for subsequent NUTS implementation modules.
    """
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    text = "\n".join(_extract_code_cells(nb)).lower()

    evidence: Dict[str, int] = {}
    scores: Dict[InferenceBackend, int] = {}
    for b, keys in _KEYWORDS.items():
        s = 0
        for k in keys:
            cnt = text.count(k.lower())
            if cnt:
                evidence[k] = cnt
                s += cnt
        scores[b] = s

    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] == 0:
        return InferenceBackendConfig(backend=InferenceBackend.UNKNOWN, notes="No backend keywords detected.", evidence={})

    # If multiple backends tie, label unknown (conservative)
    top = [b for b, sc in scores.items() if sc == best[1] and sc > 0]
    if len(top) > 1:
        return InferenceBackendConfig(
            backend=InferenceBackend.UNKNOWN,
            notes=f"Ambiguous backend evidence (ties): {top}.",
            evidence=evidence,
        )

    return InferenceBackendConfig(
        backend=best[0],
        notes=f"Detected backend={best[0].value} with score={best[1]}",
        evidence=evidence,
    )
