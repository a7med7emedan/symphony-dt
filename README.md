# SYMPHONY: Systematic Monitoring and Personalised Hybrid Orchestrated Neurological digital twin with clinic interpretability

[![ci](https://github.com/a7med7emedan/symphony-dt/actions/workflows/ci.yml/badge.svg)](https://github.com/a7med7emedan/symphony-dt/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
![docker](https://img.shields.io/badge/docker-non--root%2C%20no%20data-informational)

`symphony-dt` is a reproducible, HPC-friendly Python package that produces **standardized SYMPHONY clinic-updated digital twin results** for Parkinson's disease from **PPMI-shaped clinical tables**: latent motor, cognitive and autonomic states, multi-horizon forecasts, clinical event risk curves, and confidence gating that refuses to forecast when the evidence is too thin.

It runs deterministically on a laptop, in a container, or as a Slurm array, and every run is schema-validated.

## Try it in one minute (synthetic data)

No PPMI access needed. The package ships a generator for invented, PPMI-shaped tables.

```bash
pip install "git+https://github.com/a7med7emedan/symphony-dt.git"
symphony-dt quickstart --workdir symphony-quickstart
```

This writes synthetic inputs, runs the demo pipeline and parity-checks the result. Tables land under `symphony-quickstart/outputs/<run_id>/results/`.

Or with Docker:

```bash
docker build -t symphony-dt .
docker run --rm -v "$PWD/qs:/work" symphony-dt quickstart --workdir /work/run
```

## Data governance

PPMI data is governed by the PPMI Data Use Agreement. This repository therefore:

- contains **no PPMI data**, raw or derived, and no per-participant results;
- builds a container image that holds **code only** (CI checks that no data files are inside);
- ignores `data/`, `outputs/` and `*.csv` in git so local tables are not committed by accident.

Use your own approved PPMI export locally. The synthetic generator exists so that tests, CI and demos never need real data.

## Key capabilities

- Deterministic **demo** mode for fast validation
- **Full-cohort** execution in a single run
- **Shard-based execution** for HPC (Slurm arrays or local loops)
- Shard-safe execution + reproducible aggregation
- Output **parity and schema validation**
- Unit and integration testing via `pytest`
- **GitHub Actions** CI: lint, tests on Python 3.10 to 3.12, inference extras, synthetic end-to-end run, Docker build and in-container run
- Synthetic, PPMI-shaped data generator (`symphony-dt synth`, `symphony-dt quickstart`)
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

The pipeline expects a directory containing the processed CSV tables (e.g. extracted from a preprocessing step):
`baseline_table_PDLiKE.csv` and `clinic_visit_table_PDLiKE.csv` (see `src/symphony_dt/synthetic.py` for the exact columns).

Configure the input location in `configs/config.demo.yaml`, or override it per run:

```yaml
io:
  input_dir: data/ppmi_preprocessed   # local only, never committed
```

```bash
symphony-dt run --config configs/config.demo.yaml --mode demo --input-dir /path/to/preprocessed_unzipped
```

To generate synthetic tables with the same layout:

```bash
symphony-dt synth --out-dir data/synthetic --n-patients 120
symphony-dt run --config configs/config.synthetic.yaml --mode demo
```

### Docker with your own data

```bash
docker build --build-arg INFERENCE=1 -t symphony-dt:inference .
docker run --rm -v /path/to/preprocessed:/data:ro -v "$PWD/outputs:/work/outputs" \
  symphony-dt:inference run --config /opt/symphony-dt/configs/config.demo.yaml --mode demo --input-dir /data
```

Data is mounted read-only at runtime and never copied into the image.

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

Patient-level tables are de-duplicated on their keys. Rate tables are averaged across run directories without weighting by shard size, so point `--runs-root` at one set of shard runs only. The command warns if it finds shard runs mixed with demo or full runs.

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

`.github/workflows/ci.yml` runs on every push and pull request:

- `ruff` lint
- `pytest` with coverage on Python 3.10, 3.11 and 3.12
- the full suite again with the NumPyro/JAX extras, including an end-to-end NUTS fit, condition and forecast on synthetic data
- `symphony-dt quickstart` on a clean install
- Docker build (with and without inference extras), non-root check, no-data check, and the quickstart inside the container

A matching `.gitlab-ci.yml` is kept for GitLab mirrors.

## Results explorer notebook

`notebooks/explorer/` holds an interactive Jupyter/Colab dashboard for a SYMPHONY results bundle. The bundle is not distributed here; see `notebooks/explorer/README.md`.

---

## Citation

If you use this software, please cite it using `CITATION.cff`.

## License

Apache-2.0 (see `LICENSE` and `NOTICE`). PPMI data used with this software remains subject to the PPMI Data Use Agreement.
