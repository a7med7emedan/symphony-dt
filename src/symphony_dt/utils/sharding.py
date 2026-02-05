from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

from symphony_dt.utils.hashing import stable_int_hash


@dataclass(frozen=True)
class ShardAssignment:
    shard_id: int
    num_shards: int
    patient_ids: List[int]


def assign_patients_to_shard(
    patient_ids: Sequence[int],
    shard_id: int,
    num_shards: int,
    method: Literal["hash", "stride"] = "hash",
) -> ShardAssignment:
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0")
    if not (0 <= shard_id < num_shards):
        raise ValueError("shard_id must be in [0, num_shards-1]")

    ids = sorted(set(int(x) for x in patient_ids))

    if method == "stride":
        assigned = ids[shard_id::num_shards]
        return ShardAssignment(shard_id=shard_id, num_shards=num_shards, patient_ids=assigned)

    if method == "hash":
        assigned = [pid for pid in ids if stable_int_hash(str(pid), num_shards) == shard_id]
        return ShardAssignment(shard_id=shard_id, num_shards=num_shards, patient_ids=assigned)

    raise ValueError(f"Unknown sharding method: {method}")
