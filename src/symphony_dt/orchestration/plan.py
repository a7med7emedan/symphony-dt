from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass(frozen=True)
class FoldRunPlan:
    n_folds: int
    folds: List[Dict[str, Any]]


def build_fold_plan(assignment_df: pd.DataFrame, n_folds: int) -> FoldRunPlan:
    if assignment_df.empty:
        return FoldRunPlan(n_folds=n_folds, folds=[{"fold_id": k, "train_n": 0, "test_n": 0} for k in range(n_folds)])

    folds: List[Dict[str, Any]] = []
    for k in range(n_folds):
        test_n = int((assignment_df["fold_id"] == k).sum())
        train_n = int((assignment_df["fold_id"] != k).sum())
        folds.append({"fold_id": k, "train_n": train_n, "test_n": test_n})
    return FoldRunPlan(n_folds=n_folds, folds=folds)
