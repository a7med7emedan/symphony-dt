# Slurm usage (symphony-dt)

## Sharded run (job array)

Example for 20 shards:

```bash
sbatch slurm/ppmi_array.sbatch configs/config.demo.yaml 20
```

Each array task runs:

- `symphony-dt run --mode shard --shard-id $SLURM_ARRAY_TASK_ID --num-shards <N>`

Sharding is deterministic and based on config `sharding.method` (default: hash).

## Cohort aggregation

After array completion:

```bash
symphony-dt aggregate-all --runs-root outputs --out-dir outputs/COHORT_RESULTS
```

This writes cohort-level versions of all standardized outputs currently produced by the pipeline.
