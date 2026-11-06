from __future__ import annotations

from typing import List, Literal, Sequence

from symphony_dt.utils.hashing import stable_int_hash


def select_demo_patient_ids(
    patient_ids: Sequence[int],
    n_patients: int,
    strategy: Literal["sorted_first", "stable_hash"],
    seed: int,
) -> List[int]:
    ids = sorted(set(int(x) for x in patient_ids))
    if not ids:
        return []
    if n_patients >= len(ids):
        return ids

    if strategy == "sorted_first":
        return ids[:n_patients]

    if strategy == "stable_hash":
        scored = [(stable_int_hash(f"{seed}:{pid}", 10_000_000), pid) for pid in ids]
        scored.sort()
        return [pid for _, pid in scored[:n_patients]]

    raise ValueError(f"Unknown strategy: {strategy}")
