# symphony-dt

Production-grade, HPC-friendly implementation of the SYMPHONY clinic-updated digital twin pipeline for PPMI exportables.

This repository supports:
- local runs (`demo`, `full`)
- HPC/Slurm sharded runs (`shard`) with deterministic sharding
- cohort-level aggregation of shard outputs
- schema validation + preprocessing reports + reproducibility artifacts

> Note: The repo includes both baseline forecast engines (LOCF) and optional NumPyro-based NUTS components for scaffold/conditioning/forecasting. Turn on the inference backend only after installing the `inference` extras.

## Install

```bash
pip install -e .[dev]
```

Optional (NumPyro/JAX inference):
```bash
pip install -e .[dev,inference]
```

## Inputs

Set `io.input_dir` in `configs/config.demo.yaml` to the extracted `ppmi.zip` directory.

## Run (demo)

```bash
symphony-dt run --config configs/config.demo.yaml --mode demo
```

Outputs:
- `outputs/<run_id>/inputs/` (baseline/clinic/omics)
- `outputs/<run_id>/preprocessing_report*.json`, `data_dictionary.md`
- `outputs/<run_id>/splits/` (CV fold artifacts)
- `outputs/<run_id>/results/` standardized tables

## Run (HPC shard mode)

```bash
sbatch slurm/ppmi_array.sbatch configs/config.demo.yaml 20
```

Aggregate all tables:

```bash
symphony-dt aggregate-all --runs-root outputs --out-dir outputs/COHORT_RESULTS
```

## Parity / schema checks

```bash
symphony-dt parity-check --run-dir outputs/<run_id>
```

## Reproducibility

Each run writes:
- config snapshot
- run metadata (UTC timestamp, seed, mode, shard)
- deterministic CV split artifacts

## Development

```bash
pytest
```
