from symphony_dt.utils.sharding import assign_patients_to_shard


def test_hash_sharding_partition() -> None:
    ids = list(range(1, 101))
    num = 7
    assigned = []
    for s in range(num):
        a = assign_patients_to_shard(ids, shard_id=s, num_shards=num, method="hash").patient_ids
        assigned.extend(a)
    assert sorted(assigned) == sorted(set(ids))


def test_stride_sharding_partition() -> None:
    ids = list(range(1, 30))
    num = 4
    assigned = []
    for s in range(num):
        a = assign_patients_to_shard(ids, shard_id=s, num_shards=num, method="stride").patient_ids
        assigned.extend(a)
    assert sorted(assigned) == sorted(set(ids))
