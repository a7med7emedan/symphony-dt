from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RNGStreams:
    seed: int
    _root: np.random.SeedSequence

    @classmethod
    def from_seed(cls, seed: int) -> "RNGStreams":
        return cls(seed=seed, _root=np.random.SeedSequence(int(seed)))

    def global_rng(self) -> np.random.Generator:
        return np.random.default_rng(self._root.spawn(1)[0])

    def cv_rng(self) -> np.random.Generator:
        return np.random.default_rng(self._root.spawn(1)[0].spawn(1)[0])

    def shard_rng(self, shard_id: int) -> np.random.Generator:
        ss = np.random.SeedSequence([self.seed, 9991, int(shard_id)])
        return np.random.default_rng(ss)

    def patient_rng(self, patient_id: int, fold_id: Optional[int] = None) -> np.random.Generator:
        if fold_id is None:
            ss = np.random.SeedSequence([self.seed, 4242, int(patient_id)])
        else:
            ss = np.random.SeedSequence([self.seed, 4242, int(patient_id), int(fold_id)])
        return np.random.default_rng(ss)
