# SYMPHONY: Systematic Monitoring and Personalised Hybrid Optimised Neurological digital twin with clinic interpretability

`symphony-dt` is a production-grade, reproducible, HPC-friendly Python package that generates **standardized SYMPHONY clinic-updated digital twin results** from **PPMI clinical data**.

The repository is designed for **reuse, large-scale execution, and validation** rather than demonstration. It supports deterministic execution locally and on HPC systems, with strict output schemas to ensure reproducibility and downstream compatibility.

## Key capabilities

- Deterministic **demo** mode for fast validation
- **Full-cohort** execution in a single run
- **Shard-based execution** for HPC (Slurm arrays or local loops)
- Shard-safe output writing with reproducible aggregation
- Cohort-level aggregation into standardized result tables
- Output **parity and schema validation**
- Unit and integration testing via `pytest`
- Example **GitLab CI** configuration

> By default, the pipeline uses stable baseline forecasting engines (e.g. LOCF) to ensure robustness and scalability on HPC systems.  
> Optional NumPyro/JAX-based NUTS components are provided as extras and can be integrated separately when required.

---

## Installation

### Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### Install the package

```bash
pip install -e ".[dev]"
```

### Optional inference extras (NumPyro / JAX)

```bash
pip install -e ".[dev,inference]"
```


---

## Inputs

The pipeline expects a directory of **processed CSV tables** (e.g. extracted from a `ppmi.zip` clinical data).  
Configure the input location in `configs/config.demo.yaml`:

```yaml
io:
  input_dir: /path/to/preprocessed_unzipped
```

---

## Run modes

### Demo mode

```bash
symphony-dt run --config configs/config.demo.yaml --mode demo
```

### Full cohort mode

```bash
symphony-dt run --config configs/config.demo.yaml --mode full
```

### Shard mode

```bash
symphony-dt run   --config configs/config.demo.yaml   --mode shard   --shard-id 0   --num-shards 200
```

Shard assignment is controlled via configuration:

```yaml
sharding:
  method: hash   # hash | stride
```

---

## Sharded execution

### Local sharded execution (no Slurm)

```bash
set -euo pipefail

for i in $(seq 0 199); do
  symphony-dt run     --config configs/config.demo.yaml     --mode shard     --shard-id "$i"     --num-shards 200
done

symphony-dt aggregate-all --runs-root outputs --out-dir outputs/COHORT_RESULTS
```

### Slurm job array (HPC)

```bash
which sbatch
sbatch slurm/ppmi_array.sbatch configs/config.demo.yaml 200
```

After all array jobs complete:

```bash
symphony-dt aggregate-all --runs-root outputs --out-dir outputs/COHORT_RESULTS
```

---

## Aggregation

Merge shard-level outputs into cohort-level standardized result tables:

```bash
symphony-dt aggregate-all --runs-root outputs --out-dir outputs/COHORT_RESULTS
```

---

## Validation and parity checks

Verify that required outputs exist and conform to expected schemas:

```bash
symphony-dt parity-check --run-dir outputs/<run_id>
```

---

## Testing

Run the full test suite:

```bash
pytest
```

---

## Continuous integration

A minimal `.gitlab-ci.yml` example is included to:

- Install the package in editable mode with development dependencies
- Execute `pytest` on every push

This ensures schema stability, execution integrity, and reproducibility across environments.
