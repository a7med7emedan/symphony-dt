from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CouplingMatrix:
    """
    Fixed sparse topology per Methods:

        [ c_MM   0     0  ]
    C = [ c_CM  c_CC  c_CN]
        [ c_NM   0    c_NN]

    Indices: (M,C,N) -> (0,1,2)

    This structure prevents cyclic feedback and limits degrees of freedom.
    """
    c_mm: float
    c_cm: float
    c_cc: float
    c_cn: float
    c_nm: float
    c_nn: float

    def as_array(self) -> np.ndarray:
        C = np.zeros((3, 3), dtype=np.float64)
        C[0, 0] = float(self.c_mm)
        C[1, 0] = float(self.c_cm)
        C[1, 1] = float(self.c_cc)
        C[1, 2] = float(self.c_cn)
        C[2, 0] = float(self.c_nm)
        C[2, 2] = float(self.c_nn)
        return C


@dataclass(frozen=True)
class GlobalDynamicsParams:
    """
    Global dynamics parameters needed for Eq. (trans_general) and (xi_eps).
    """
    C: CouplingMatrix
    sigma_xi: Tuple[float, float, float]  # (sigma_xi_M, sigma_xi_C, sigma_xi_N)

    def sigma_xi_array(self) -> np.ndarray:
        sx = np.asarray(self.sigma_xi, dtype=np.float64).reshape(3)
        if np.any(sx < 0):
            raise ValueError("sigma_xi must be componentwise >= 0")
        return sx
