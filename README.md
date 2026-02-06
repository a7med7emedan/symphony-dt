# SYMPHONY: Systematic Monitoring and Personalised Hybrid Orchestrated Neurological digital twin with clinic interpretability

`symphony-dt` is a production-grade, reproducible, HPC-friendly Python package that generates **standardized SYMPHONY clinic-updated digital twin results** from **PPMI clinical data**.

The repository is designed for **reuse, large-scale execution, and validation**. It supports deterministic execution locally and on HPC systems, with schema-validated outputs to ensure reproducibility and downstream compatibility.

## Key capabilities

- Deterministic **demo** mode for fast validation
- **Full-cohort** execution in a single run
- **Shard-based execution** for HPC (Slurm arrays or local loops)
- Shard-safe execution + reproducible aggregation
- Output **parity and schema validation**
- Unit and integration testing via `pytest`
- Example **GitLab CI** configuration
- Optional **Bayesian personalization / posterior simulation** via NumPyro + JAX (extras)

> By default, the pipeline uses stable baseline engines (e.g. LOCF) for robustness and HPC scaling.  
> NumPyro/JAX-based NUTS workflows are **opt-in** and do not affect baseline runs unless explicitly invoked.

---

## Installation

### Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### Install the package (development)

```bash
pip install -e ".[dev]"
```

### Optional inference extras (NumPyro / JAX)

Install this only if you intend to run the `symphony-dt nuts ...` commands.

```bash
pip install -e ".[dev,inference]"
```

---

## Inputs

The pipeline expects a directory containing the processed CSV tables (e.g. extracted from a preprocessing step).

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
symphony-dt run --config configs/config.demo.yaml --mode shard --shard-id 0 --num-shards 200
```

Shard assignment is controlled via configuration:

```yaml
sharding:
  method: hash   # hash | stride
```

---

## Bayesian personalization (NUTS / NumPyro)

This is intended for posterior conditioning + posterior predictive simulation.
It requires the inference extras (`pip install -e ".[dev,inference]"`).

### 1) Fit a scaffold (global posterior)

```bash
symphony-dt nuts fit-scaffold   --config configs/config.demo.yaml   --fold-id 0   --out-dir outputs/scaffold_fold0_smoke   --seed 0   --num-warmup 100   --num-samples 100   --num-chains 2
```

### 2) Condition per patient/visit (personalization artifacts)

Choose a `--visit-index` that exists for each patient. In many PPMI-style exports, early visit indices may not have all endpoints recorded.
For a demo smoke run, `--visit-index 5` is typically a good default.

```bash
symphony-dt nuts condition   --config configs/config.demo.yaml   --drawset-dir outputs/scaffold_fold0_smoke   --out-dir outputs/cond_fold0_v5_smoke   --mode demo   --visit-index 5   --l-cond 10   --seed 0   --num-warmup 50   --num-samples 50   --num-chains 1
```

### 3) Write posterior predictive forecasts

```bash
symphony-dt nuts forecast   --config configs/config.demo.yaml   --drawset-dir outputs/scaffold_fold0_smoke   --conditional-dir outputs/cond_fold0_v5_smoke   --out-results-dir outputs/posterior_results_demo_v5   --mode demo   --visit-index 5   --l-cond 10   --n-draws-predictive 200   --seed 0
```

---

## Sharded execution

### Local sharded execution (no Slurm)

Use this when you do **not** have Slurm available (e.g., on a laptop).

```bash
set -euo pipefail

for i in $(seq 0 199); do
  symphony-dt run --config configs/config.demo.yaml --mode shard --shard-id "$i" --num-shards 200
done

symphony-dt aggregate-all --runs-root outputs --out-dir outputs/COHORT_RESULTS
```

### Slurm job array (HPC)

> `sbatch` exists only on a Slurm login/submit node. If you run this on a laptop, `sbatch: command not found` is expected.

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
RUN_DIR=$(ls -dt outputs/symphony-dt__* | head -n 1)
symphony-dt parity-check --run-dir "$RUN_DIR"
```

---

## Testing

Run the full test suite:

```bash
pytest
```

Recommended smoke checks after changes:

```bash
symphony-dt run --config configs/config.demo.yaml --mode demo
symphony-dt run --config configs/config.demo.yaml --mode full
```

---

## Continuous integration

A minimal `.gitlab-ci.yml` example is included to:

- Install the package (editable) with development dependencies
- Execute `pytest` on every push

---

## License

Apache-2.0 (see `LICENSE` and `NOTICE`).
