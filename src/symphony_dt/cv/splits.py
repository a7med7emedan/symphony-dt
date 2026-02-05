from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldAssignment:
    n_folds: int
    assignment: pd.DataFrame  # patient_id, sex, fold_id


def make_stratified_folds(
    baseline: pd.DataFrame,
    patient_ids: Sequence[int],
    n_folds: int,
    seed: int,
) -> FoldAssignment:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    req = {"patient_id", "sex"}
    missing = sorted([c for c in req if c not in baseline.columns])
    if missing:
        raise ValueError(f"baseline missing required columns for CV split: {missing}")

    ids = sorted(set(int(x) for x in patient_ids))
    b = baseline[baseline["patient_id"].isin(ids)].copy()
    if b.empty:
        return FoldAssignment(n_folds=n_folds, assignment=pd.DataFrame(columns=["patient_id", "sex", "fold_id"]))

    b["sex"] = b["sex"].astype(str).str.upper()
    bad = sorted(set(b["sex"].dropna().unique().tolist()) - {"M", "F"})
    if bad:
        raise ValueError(f"Unexpected sex values in baseline for CV split: {bad}")

    rng = np.random.default_rng(seed)

    def assign_for_sex(sex: str) -> Dict[int, int]:
        sub = b[b["sex"] == sex]["patient_id"].astype(int).unique().tolist()
        sub = sorted(sub)
        rng.shuffle(sub)
        mapping: Dict[int, int] = {}
        for idx, pid in enumerate(sub):
            mapping[pid] = int(idx % n_folds)
        return mapping

    m_map = assign_for_sex("M")
    f_map = assign_for_sex("F")

    rows = []
    for pid in ids:
        sex_row = b[b["patient_id"] == pid]["sex"]
        if sex_row.empty:
            continue
        sex = str(sex_row.iloc[0]).upper()
        fold_id = m_map[int(pid)] if sex == "M" else f_map[int(pid)]
        rows.append((int(pid), sex, int(fold_id)))

    df = pd.DataFrame(rows, columns=["patient_id", "sex", "fold_id"]).sort_values("patient_id").reset_index(drop=True)
    return FoldAssignment(n_folds=n_folds, assignment=df)


def fold_train_test_ids(assignment: FoldAssignment, fold_id: int) -> Tuple[List[int], List[int]]:
    if not (0 <= fold_id < assignment.n_folds):
        raise ValueError("fold_id out of range")
    test_ids = assignment.assignment.loc[assignment.assignment["fold_id"] == fold_id, "patient_id"].astype(int).tolist()
    train_ids = assignment.assignment.loc[assignment.assignment["fold_id"] != fold_id, "patient_id"].astype(int).tolist()
    return train_ids, test_ids
