# SYMPHONY — Systematic Monitoring and Personalised Hybrid Optimised Neurological digital twin with clinic interpretability

`symphony-dt` is a production-grade, reproducible, HPC-friendly Python package that generates **standardized SYMPHONY clinic-updated digital twin result tables** from **PPMI-style preprocessed exports**.

This repository supports:
- **demo** mode: small deterministic subset for fast validation
- **full** mode: entire cohort in a single run
- **shard** mode: deterministic sharding (Slurm array or local loop), shard-safe output writes
- **aggregation**: merge shard outputs into cohort-level tables
- **parity / schema checks**: validate presence + schemas of standardized outputs
- **pytest** unit + integration tests
- **GitLab CI** example to run tests on every push

> Default pipeline execution uses baseline forecast engines (LOCF) for operational stability and HPC scaling.  
> Optional NumPyro-based NUTS components exist (extras) and can be integrated into orchestration separately.

---

## Installation

### Recommended: create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### Install package (editable)

```bash
pip install -e ".[dev]"
```

Optional inference extras (NumPyro/JAX):

```bash
pip install -e ".[dev,inference]"
```

### If the CLI command is not found

If you see `zsh: command not found: symphony-dt`, activate your venv:

```bash
source .venv/bin/activate
which symphony-dt
```

---

## Inputs

The pipeline expects a directory of **preprocessed** CSV tables (e.g., extracted from your `ppmi.zip`).  
Set that directory in `configs/config.demo.yaml`:

```yaml
io:
  input_dir: /path/to/preprocessed_unzipped
```

### Typical input files

- `baseline_table_PDLiKE.csv`
- `clinic_visit_table_PDLiKE.csv`
- `omics_table_optional.csv` (optional)
- `preprocessing_audit.json` (optional)
- `data_dictionary.md` (optional)

### Minimum required columns

**Baseline** (minimum):
- `patient_id`
- `sex` (`M`/`F`)
- `age_baseline_years`
- `is_pd_like`

**Clinic visits** (minimum):
- `patient_id`
- `visit_id`  
  - supports string codes like `SC`, `BL`, `V01`… and deterministically encodes them
- `t_days_since_baseline`  
  - ties/out-of-order are coerced to **strictly increasing** per patient (stable ordering)
- outcomes: `updrs3_total`, `moca_total`, `scopa_aut_total`
- LEDD (stratification only): `ledd_total`
- missingness flags: `miss_updrs3`, `miss_moca`, `miss_scopa`
- plus baseline covariates duplicated in visits: `sex`, `age_baseline_years`, `is_pd_like`

**Omics** (optional):
- `patient_id` plus feature columns

---

## Run modes

### Demo mode

```bash
symphony-dt run --config configs/config.demo.yaml --mode demo
```

### Full mode

```bash
symphony-dt run --config configs/config.demo.yaml --mode full
```

### Shard mode

```bash
symphony-dt run --config configs/config.demo.yaml --mode shard --shard-id 0 --num-shards 200
```

Sharding method is configured by:

```yaml
sharding:
  method: hash   # hash|stride
```

> **Shard-stability note:** state binning and trajectory clustering are implemented with **fixed cutpoints** (not `qcut`) to avoid shard-dependent failures and to keep labels consistent across shards.

---

## Sharded execution

### Local sharded run (no Slurm)

Use on a laptop/workstation where `sbatch` is unavailable:

```bash
set -euo pipefail

for i in $(seq 0 199); do
  symphony-dt run --config configs/config.demo.yaml --mode shard --shard-id "$i" --num-shards 200
done

symphony-dt aggregate-all --runs-root outputs --out-dir outputs/COHORT_RESULTS
```

### Slurm job array (HPC)

Run on an HPC login node where Slurm is available:

```bash
which sbatch
sbatch slurm/ppmi_array.sbatch configs/config.demo.yaml 200
```

After all array tasks finish:

```bash
symphony-dt aggregate-all --runs-root outputs --out-dir outputs/COHORT_RESULTS
```

---

## Outputs

Each run writes:
- `outputs/<run_id>/config_snapshot.yaml`
- `outputs/<run_id>/metadata.json`

And subdirectories:
- `outputs/<run_id>/inputs/` (baseline/clinic/omics)
- `outputs/<run_id>/data_quality/` (missingness profiles)
- `outputs/<run_id>/splits/` (CV fold artifacts)
- `outputs/<run_id>/results/` (**standardized result tables**)

### Standardized result tables (produced by default pipeline)

- `forecast_one_step_predictions.csv`
- `forecast_horizon_long.csv`
- `forecast_horizon_wide.csv`
- `forecast_horizon_long_uq.csv`
- `forecast_horizon_long_uq_refined.csv`
- `forecast_risk_curves.csv`
- `forecast_horizon_states.csv`
- `states_long.csv`
- `state_occupancy.csv`
- `trajectory_clusters.csv`
- `transitions_long.csv`
- `transitions_edge_list.csv`
- `clinical_event_rates_by_state.csv`
- `clinical_event_rates_by_trajectory.csv`
- `clinical_event_rates_by_trust.csv`
- `clinical_event_pairs.csv`
- `evaluation_mae.csv`
- `evaluation_stratified_mae.csv`
- `evaluation_coverage.csv`
- `personalization_diagnostics.csv`
- `clinician_summary_risk_table.csv`
- `patient_panel.csv`

### Input/export tables (per run)

- `inputs/baseline.csv`
- `inputs/clinic_visits.csv`
- `inputs/omics_prebaseline.csv`

### Data quality + preprocessing artifacts (per run)

- `data_quality/data_quality_missingness.csv`
- `preprocessing_report.json`
- `preprocessing_report.validated.json`
- `data_dictionary.md`

---

## Aggregation (cohort-level)

Aggregate all standardized outputs across shard runs:

```bash
symphony-dt aggregate-all --runs-root outputs --out-dir outputs/COHORT_RESULTS
```

---

## Validation / parity checks

Validate required files exist and key schemas validate:

```bash
symphony-dt parity-check --run-dir outputs/<run_id>
```

---

## Testing

Run unit/integration tests:

```bash
pytest
```

---

## GitLab CI

A minimal `.gitlab-ci.yml` (example) runs `pytest` on every push.

Typical CI behavior:
- install editable package with dev dependencies
- run `pytest`

---

## Troubleshooting

### `sbatch: command not found`
You are not on a Slurm machine. Use the local shard loop or run Slurm commands on the HPC login node.

### `symphony-dt: command not found`
Activate your virtual environment:
```bash
source .venv/bin/activate
```

### Shard runs sometimes fail
If you see intermittent failures, it is usually due to shard-dependent quantile binning. This repo uses fixed cutpoints for shard-stability; ensure your branch includes the latest changes.

---

## Reproducibility

Each run stores:
- config snapshot (exact YAML used)
- run metadata (timestamp, seed, mode, shard id)
- deterministic sharding behavior (config-driven)
- deterministic CV splits (seeded and saved)
