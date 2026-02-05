from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ConditionalRunArtifacts:
    """
    On-disk representation of a conditional run for (patient_id, visit_index, global_draw_index).

    Layout:
      <dir>/
        samples_x_tn.npy         # (S,3) x(t_n) samples
        diagnostics.json         # scalar diagnostics used by gating rules
        metadata.json            # provenance: patient_id, visit_index, global_draw_index, seed, backend, etc.
    """
    samples_x_tn: np.ndarray
    diagnostics: Dict[str, Any]
    metadata: Dict[str, Any]


def save_conditional_run(dir_path: Path, art: ConditionalRunArtifacts, overwrite: bool = False) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    x_path = dir_path / "samples_x_tn.npy"
    if x_path.exists() and not overwrite:
        raise FileExistsError(f"Conditional run exists and overwrite=false: {dir_path}")
    np.save(x_path, np.asarray(art.samples_x_tn, dtype=np.float64))
    (dir_path / "diagnostics.json").write_text(json.dumps(art.diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    (dir_path / "metadata.json").write_text(json.dumps(art.metadata, indent=2, sort_keys=True), encoding="utf-8")


def load_conditional_run(dir_path: Path) -> ConditionalRunArtifacts:
    x_path = dir_path / "samples_x_tn.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing samples_x_tn.npy: {x_path}")
    samples_x = np.load(x_path)
    diagnostics = json.loads((dir_path / "diagnostics.json").read_text(encoding="utf-8")) if (dir_path / "diagnostics.json").exists() else {}
    metadata = json.loads((dir_path / "metadata.json").read_text(encoding="utf-8")) if (dir_path / "metadata.json").exists() else {}
    return ConditionalRunArtifacts(samples_x_tn=samples_x, diagnostics=diagnostics, metadata=metadata)
