from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


_CANONICAL: Dict[str, Tuple[str, ...]] = {
    "patient_id": ("patient_id", "PATNO", "patno", "patientid", "participant_id", "subject_id"),
    "visit_id": ("visit_id", "visit", "visitnum", "event_id", "VISIT_ID"),
    "t_days_since_baseline": ("t_days_since_baseline", "t", "days_since_baseline", "time_days", "t_days"),
    "updrs3_total": ("updrs3_total", "UPDRS3", "mds_updrs3", "updrs_part_iii_total"),
    "moca_total": ("moca_total", "MOCA", "moca"),
    "scopa_aut_total": ("scopa_aut_total", "SCOPA_AUT", "scopa_aut"),
    "ledd_total": ("ledd_total", "LEDD", "ledd"),
    "miss_updrs3": ("miss_updrs3", "missing_updrs3", "updrs3_missing"),
    "miss_moca": ("miss_moca", "missing_moca", "moca_missing"),
    "miss_scopa": ("miss_scopa", "missing_scopa", "scopa_missing"),
    "sex": ("sex", "SEX", "gender"),
    "age_baseline_years": ("age_baseline_years", "age", "baseline_age_years", "age_at_baseline"),
    "is_pd_like": ("is_pd_like", "pd_like", "is_pdlike"),
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    col_map: Dict[str, str] = {}
    reverse: Dict[str, str] = {}
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}

    for canonical, variants in _CANONICAL.items():
        found = None
        for v in variants:
            if v in cols:
                found = v
                break
            if v.lower() in cols_lower:
                found = cols_lower[v.lower()]
                break
        if found is None:
            continue
        if canonical in reverse:
            raise ValueError(f"Ambiguous mapping: {reverse[canonical]} and {found} both map to {canonical}")
        reverse[canonical] = found
        col_map[found] = canonical

    return df.rename(columns=col_map).copy()
