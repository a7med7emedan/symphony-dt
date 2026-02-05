from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class PatientVisit:
    patient_id: int
    visit_id: int
    t_days_since_baseline: float
    updrs3_total: Optional[float]
    moca_total: Optional[float]
    scopa_aut_total: Optional[float]
    ledd_total: Optional[float]
    miss_updrs3: bool
    miss_moca: bool
    miss_scopa: bool


def extract_patient_visits(clinic: pd.DataFrame, patient_id: int) -> List[PatientVisit]:
    """
    Extract ordered visits for a patient.

    Assumes clinic is schema-validated (Module 3 contracts), contains canonical columns.
    """
    required = [
        "patient_id",
        "visit_id",
        "t_days_since_baseline",
        "updrs3_total",
        "moca_total",
        "scopa_aut_total",
        "ledd_total",
        "miss_updrs3",
        "miss_moca",
        "miss_scopa",
    ]
    missing = [c for c in required if c not in clinic.columns]
    if missing:
        raise ValueError(f"clinic missing required columns for visit extraction: {missing}")

    df = clinic[clinic["patient_id"] == int(patient_id)].sort_values("t_days_since_baseline").copy()
    visits: List[PatientVisit] = []
    for _, r in df.iterrows():
        visits.append(
            PatientVisit(
                patient_id=int(r["patient_id"]),
                visit_id=int(r["visit_id"]),
                t_days_since_baseline=float(r["t_days_since_baseline"]),
                updrs3_total=None if pd.isna(r["updrs3_total"]) else float(r["updrs3_total"]),
                moca_total=None if pd.isna(r["moca_total"]) else float(r["moca_total"]),
                scopa_aut_total=None if pd.isna(r["scopa_aut_total"]) else float(r["scopa_aut_total"]),
                ledd_total=None if pd.isna(r["ledd_total"]) else float(r["ledd_total"]),
                miss_updrs3=bool(r["miss_updrs3"]),
                miss_moca=bool(r["miss_moca"]),
                miss_scopa=bool(r["miss_scopa"]),
            )
        )
    return visits
