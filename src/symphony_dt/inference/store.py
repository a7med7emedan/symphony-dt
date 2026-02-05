from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class PosteriorDrawSet:
    """
    Lightweight, dependency-minimal posterior draw container.

    Conventions:
      - arrays are stored as float64 unless inherently integer/bool
      - axis 0 is draw index (L draws)
      - metadata holds provenance (fold_id, sampler, seed, etc.)
      - diagnostics holds sampler diagnostics (rhat, ess_bulk, divergences, accept_prob, etc.)

    Designed to be produced by external inference backends and then loaded by this package.
    """
    arrays: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    diagnostics: Dict[str, Any]

    def n_draws(self) -> int:
        if not self.arrays:
            return 0
        n = None
        for k, a in self.arrays.items():
            if a.ndim == 0:
                raise ValueError(f"Draw array {k} must have at least 1 dimension (draw axis)")
            if n is None:
                n = int(a.shape[0])
            elif int(a.shape[0]) != n:
                raise ValueError(f"Inconsistent draw axis for {k}: {a.shape[0]} != {n}")
        return int(n or 0)


class DrawStore:
    """
    Directory-based draw store.

    Layout:
      <dir>/
        arrays.npz
        metadata.json
        diagnostics.json
    """
    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def arrays_path(self) -> Path:
        return self.root / "arrays.npz"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    @property
    def diagnostics_path(self) -> Path:
        return self.root / "diagnostics.json"

    def save(self, drawset: PosteriorDrawSet, overwrite: bool = False) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.arrays_path.exists() and not overwrite:
            raise FileExistsError(f"arrays.npz exists and overwrite=false: {self.arrays_path}")
        np.savez_compressed(self.arrays_path, **drawset.arrays)
        self.metadata_path.write_text(json.dumps(drawset.metadata, indent=2, sort_keys=True), encoding="utf-8")
        self.diagnostics_path.write_text(json.dumps(drawset.diagnostics, indent=2, sort_keys=True), encoding="utf-8")

    def load(self) -> PosteriorDrawSet:
        if not self.arrays_path.exists():
            raise FileNotFoundError(f"Missing arrays.npz: {self.arrays_path}")
        arrays_npz = np.load(self.arrays_path, allow_pickle=False)
        arrays = {k: arrays_npz[k] for k in arrays_npz.files}

        metadata = {}
        diagnostics = {}
        if self.metadata_path.exists():
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.diagnostics_path.exists():
            diagnostics = json.loads(self.diagnostics_path.read_text(encoding="utf-8"))

        ds = PosteriorDrawSet(arrays=arrays, metadata=metadata, diagnostics=diagnostics)
        ds.n_draws()
        return ds


def save_drawset(path: Path, drawset: PosteriorDrawSet, overwrite: bool = False) -> None:
    DrawStore(path).save(drawset, overwrite=overwrite)


def load_drawset(path: Path) -> PosteriorDrawSet:
    return DrawStore(path).load()
