<p align="center">
  <img src="https://raw.githubusercontent.com/a7med7emedan/symphony-dt/main/docs/assets/symphony-wordmark.svg" alt="SYMPHONY" width="640">
</p>

<p align="center">
  <a href="https://github.com/a7med7emedan/symphony-dt/actions/workflows/ci.yml"><img src="https://github.com/a7med7emedan/symphony-dt/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/a7med7emedan/symphony-dt/actions/workflows/docker.yml"><img src="https://github.com/a7med7emedan/symphony-dt/actions/workflows/docker.yml/badge.svg" alt="docker"></a>
  <a href="https://github.com/a7med7emedan/symphony-dt/actions/workflows/codeql.yml"><img src="https://github.com/a7med7emedan/symphony-dt/actions/workflows/codeql.yml/badge.svg" alt="codeql"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776ab" alt="python 3.10 to 3.12">
  <a href="https://github.com/a7med7emedan/symphony-dt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2e7d32" alt="Apache-2.0"></a>
</p>

# SYMPHONY

**Systematic Monitoring and Personalised Hybrid Orchestrated Neurological digital twin, with clinic interpretability.**

`symphony-dt` is a Python package for a clinic-updated Bayesian digital twin of Parkinson's disease. It reads PPMI-shaped clinical tables and returns latent motor, cognitive and autonomic states. It also returns multi-horizon forecasts with intervals, clinical event risk curves and calibration tables. A confidence gate withholds a forecast when the evidence behind it is too thin.

Every run is deterministic for a given seed and schema-validated. It runs the same way on a laptop, in a container or as a Slurm array.

<p align="center">
  <img src="https://raw.githubusercontent.com/a7med7emedan/symphony-dt/main/docs/assets/symphony-architecture.svg" alt="SYMPHONY method: data, hierarchical latent dynamics, bounded observation model, NUTS inference, confidence gate and outputs" width="100%">
</p>

## Quick start

No PPMI access is needed. The package ships a generator for invented, PPMI-shaped tables.

```bash
pip install "git+https://github.com/a7med7emedan/symphony-dt.git"
symphony-dt quickstart --workdir symphony-quickstart
```

This writes synthetic inputs, runs the demo pipeline and parity-checks the result:

```text
Loaded baseline rows=120, clinic rows=933, omics rows=0
OK Run complete. Results: symphony-quickstart/outputs/symphony-dt__<UTC>__seed20240205__demo/results
OK Quickstart complete.
```

The `results/` folder then holds 22 standardised tables, among them:

| Table | Content |
| --- | --- |
| `states_long.csv`, `state_occupancy.csv` | latent domain states per visit and their occupancy |
| `forecast_one_step_predictions.csv` | next-visit forecasts with central 90% intervals (5th to 95th percentile) |
| `forecast_horizon_long_uq_refined.csv` | 180, 365 and 730 day forecasts with 90% intervals and their widths |
| `forecast_risk_curves.csv` | probability of a clinically meaningful change per endpoint and horizon |
| `evaluation_coverage.csv`, `evaluation_mae.csv` | coverage of the 90% interval (`cov90_*`) and error, overall and by sex and age quartile |
| `personalization_diagnostics.csv`, `clinician_summary_risk_table.csv` | gating diagnostics and a clinician-facing risk summary |

With Docker:

```bash
docker build -t symphony-dt .
docker run --rm -v "$PWD/qs:/work" symphony-dt quickstart --workdir /work/run
```

## The model

SYMPHONY is a hierarchical, nonlinear state-space model observed at irregular clinic visits. The implementation lives in [`model/`](src/symphony_dt/model) (NumPy) and [`inference_backend/`](src/symphony_dt/inference_backend) (NumPyro and JAX).

**Latent state.** Each participant $i$ carries a state $s_i(t) \in \mathbb{R}^3$ for the motor (M), cognitive (C) and autonomic (N) domains, read on the unit scale as $x_i(t) = \sigma(s_i(t))$.

**Hierarchical progression drive.** Baseline covariates $d_i$ (age, sex) and optional omics scores $z_i$ set a per-participant drive, with partial pooling across the cohort:

$$
r_i = \mu_r + \Gamma d_i + A z_i + b_i, \qquad b_i \sim \mathcal{N}\left(0, \mathrm{diag}(\sigma_r^2)\right).
$$

**Monotone stochastic dynamics.** Between visits separated by $\Delta$ days, the state can only progress, at a rate shaped by the other domains:

$$
s_i(t+\Delta) = s_i(t) + \Delta \cdot \mathrm{softplus}\left(r_i + C\,x_i(t)\right) \odot \xi_{i,t},
\qquad
\xi_{i,t} = \exp\left(-\tfrac{1}{2}\Delta\,\sigma_\xi^2 + \sqrt{\Delta}\,\sigma_\xi \odot \varepsilon_{i,t}\right),
$$

with $\varepsilon_{i,t} \sim \mathcal{N}(0, I_3)$. The multiplicative noise has mean one, so it widens uncertainty with elapsed time without biasing the drift. The coupling matrix $C$ has a fixed sparse topology: motor drives cognition and autonomic function, autonomic function drives cognition, and each domain has a self term. Nothing feeds back into the motor domain. This rules out cyclic feedback and limits the model to six free couplings.

**Bounded observation model.** MDS-UPDRS Part III (0 to 132), MoCA (0 to 30, reversed) and SCOPA-AUT (0 to 69) are linked to the latent scale by a monotone link $h(p; a, b) = \sigma(a + b\,\mathrm{logit} p)$ with $b > 0$. Each score then follows a Gaussian truncated to its own support:

$$
y^{(k)}_{i,t} \sim \mathcal{N}_{[L_k, U_k]}\left(L_k + (U_k - L_k)\,h\left(x^{(k)}_i(t); a_k, b_k\right),\ \sigma_k^2\right).
$$

The truncation matters at the floor and ceiling of each scale, where many participants sit (MoCA near 30 is common). Missing endpoints are masked rather than imputed.

## Inference

The Bayesian path is opt-in and needs the inference extras (`pip install "symphony-dt[inference] @ git+https://github.com/a7med7emedan/symphony-dt.git"`).

1. **Global scaffold.** `symphony-dt nuts fit-scaffold` samples the population posterior with the No-U-Turn Sampler (NumPyro on JAX). All innovations $\varepsilon$ and random effects are non-centred, which avoids funnel geometry. Draws are stored as a versioned drawset.
2. **Per-patient conditioning.** `symphony-dt nuts condition` fixes global parameters at $L_{\text{cond}}$ scaffold draws. It then re-samples each participant's initial state, drive and innovations from their visits up to an anchor. Every conditional run records divergences, mean acceptance probability and $\hat{R}$.
3. **Posterior predictive forecasts.** `symphony-dt nuts forecast` pools the conditional runs into a mixture. It simulates the dynamics forward to each horizon and writes predictive intervals and event probabilities.

The default `run` command uses a baseline engine: last observation carried forward, with intervals of plus or minus 10% of each scale's range. It is fast enough for cohort-wide HPC sweeps. It writes the same table schemas, so Bayesian results can be compared with it like for like.

## Confidence gating

A forecast is shown only when it can be trusted ([`personalization/gating.py`](src/symphony_dt/personalization/gating.py)). It is suppressed when:

1. the participant has fewer than two clinic visits;
2. any endpoint is missing at the anchor visit;
3. latent uncertainty $\Delta_{i,n} = \max_j \mathrm{IQR}\left(x_j(t_{i,n})\right)$ exceeds $\tau_{\text{conf}}$;
4. fewer than $\max\left(10, \lceil 0.4\,L_{\text{cond}} \rceil\right)$ conditional runs pass their diagnostics.

A fifth rule keeps intervals but withholds threshold-event probabilities when an anchor score sits on a scale boundary.

`symphony-dt nuts select-tau-conf` chooses $\tau_{\text{conf}}$ on a training fold ([`trust/tau_conf.py`](src/symphony_dt/trust/tau_conf.py)). For each training-fold anchor, it computes $\Delta_{i,n}$ and the one-step central 90% interval from the conditioning outputs, with rule 3 switched off. Anchors suppressed by rules 1, 2 or 4 are left out, since no threshold would show them.

On a grid from 0.05 to 0.60, it picks the value whose empirical coverage is closest to 0.90, while keeping at least half of the anchors. The target matches the interval actually produced: every engine writes the 5th and 95th percentiles. If no grid value keeps half of the anchors, the value that keeps the most is used and the file records `"feasible": false`.

The command writes the chosen value, with the coverage and fraction kept at each grid point, to JSON. `nuts forecast --tau-conf-json` reads that file, and `--tau-conf` sets a value directly. With neither, the threshold is 1.0 and a warning is printed. That value leaves rule 3 inactive, because the interquartile range on the unit scale cannot exceed 1. Posterior forecasts record $\Delta_{i,n}$ for each anchor in a `delta_iqr` column.

## Calibration and evaluation

Forecasts are scored by mean absolute error and interval coverage, overall and stratified by sex and age quartile. Coverage is the share of observed scores inside the central 90% interval, from the 5th to the 95th percentile, in columns `cov90_updrs3`, `cov90_moca`, `cov90_scopa` and `cov90_overall`.

Releases up to 0.2.0 wrote the same 90% coverage under the name `cov95_*`. Those columns are still written, with identical values, as deprecated aliases for one release. The table contract, `parity-check` and `aggregate-all` accept either name, so older run directories can be validated and merged with new ones. For posterior draws, the package also provides CRPS and probability integral transform (PIT) values under truncation, with a Kolmogorov distance to uniformity. Trajectory clusters and state binning use fixed cut points, so results do not depend on how patients were sharded.

## Data governance

PPMI data is governed by the PPMI Data Use Agreement. This repository therefore:

- contains **no PPMI data**, raw or derived, and no per-participant results;
- builds a container image that holds **code only**, and CI checks that no data files are inside;
- ignores `data/`, `outputs/` and `*.csv` in git, so local tables are not committed by accident.

Use your own approved PPMI export locally. The synthetic generator exists so that tests, CI and demos never need real data.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"               # pipeline, tests and tooling
pip install -e ".[dev,inference]"     # adds NumPyro and JAX for the nuts commands
```

## Inputs

The pipeline expects a directory of processed tables: `baseline_table_PDLiKE.csv` and `clinic_visit_table_PDLiKE.csv`. [`synthetic.py`](src/symphony_dt/synthetic.py) documents the exact columns. Set the location in `configs/config.demo.yaml` or per run:

```bash
symphony-dt run --config configs/config.demo.yaml --mode demo --input-dir /path/to/preprocessed
```

To generate synthetic tables with the same layout:

```bash
symphony-dt synth --out-dir data/synthetic --n-patients 120
symphony-dt run --config configs/config.synthetic.yaml --mode demo
```

Docker with your own data, mounted read-only and never copied into the image:

```bash
docker build --build-arg INFERENCE=1 -t symphony-dt:inference .
docker run --rm -v /path/to/preprocessed:/data:ro -v "$PWD/outputs:/work/outputs" \
  symphony-dt:inference run --config /opt/symphony-dt/configs/config.demo.yaml --mode demo --input-dir /data
```

## Run modes

| Mode | Command | Use |
| --- | --- | --- |
| demo | `symphony-dt run --config configs/config.demo.yaml --mode demo` | fast validation on a fixed subset |
| full | `symphony-dt run --config configs/config.demo.yaml --mode full` | the whole cohort in one process |
| shard | `symphony-dt run --config configs/config.demo.yaml --mode shard --shard-id 0 --num-shards 200` | one slice of the cohort, for arrays |

Shard assignment is set by `sharding.method` (`hash` or `stride`).

### Bayesian personalisation end to end

```bash
symphony-dt nuts fit-scaffold --config configs/config.demo.yaml --fold-id 0 \
  --out-dir outputs/scaffold_fold0 --seed 0 --num-warmup 100 --num-samples 100 --num-chains 2

symphony-dt nuts condition --config configs/config.demo.yaml --drawset-dir outputs/scaffold_fold0 \
  --out-dir outputs/cond_fold0_v5 --mode demo --visit-index 5 --l-cond 10 --seed 0 \
  --num-warmup 50 --num-samples 50 --num-chains 1

symphony-dt nuts select-tau-conf --config configs/config.demo.yaml --drawset-dir outputs/scaffold_fold0 \
  --conditional-dir outputs/cond_fold0_v5 --out-json outputs/tau_conf_fold0_v5.json \
  --fold-id 0 --visit-index 5 --l-cond 10 --n-draws-predictive 200 --seed 0

symphony-dt nuts forecast --config configs/config.demo.yaml --drawset-dir outputs/scaffold_fold0 \
  --conditional-dir outputs/cond_fold0_v5 --out-results-dir outputs/posterior_v5 --mode demo \
  --visit-index 5 --l-cond 10 --n-draws-predictive 200 --seed 0 \
  --tau-conf-json outputs/tau_conf_fold0_v5.json
```

`select-tau-conf` uses the training-fold participants that have conditioning outputs. For a real analysis, condition with `--mode full` or in shards before selecting. Use the same `--fold-id` as `fit-scaffold`, and the same `--visit-index` and `--l-cond` throughout.

Choose a `--visit-index` that exists for each patient. Participants with too few visits are skipped, not failed. The values above are smoke-test budgets. Use at least 4 chains and several hundred warm-up and sampling iterations for real analyses.

### Sharded execution

Local loop, no scheduler needed:

```bash
for i in $(seq 0 199); do
  symphony-dt run --config configs/config.demo.yaml --mode shard --shard-id "$i" --num-shards 200
done
symphony-dt aggregate-all --runs-root outputs --out-dir outputs/COHORT_RESULTS
```

Slurm job array, from a login node:

```bash
sbatch slurm/ppmi_array.sbatch configs/config.demo.yaml 200
symphony-dt aggregate-all --runs-root outputs --out-dir outputs/COHORT_RESULTS   # after the array finishes
```

`aggregate-all` de-duplicates patient-level tables on their keys. Rate tables are averaged across run directories without weighting by shard size. Point `--runs-root` at one set of shard runs only; the command warns if shard runs are mixed with demo or full runs.

### Validation

```bash
RUN_DIR=$(ls -dt outputs/symphony-dt__* | head -n 1)
symphony-dt parity-check --run-dir "$RUN_DIR"
```

## Quality and automation

- **Tests.** `pytest` covers the dynamics, observation model, truncated normal, gating, $\tau_{\text{conf}}$ selection and its command, the coverage column aliases, sharding and aggregation. It also runs an end-to-end NUTS fit, condition and forecast on synthetic data.
- **CI** ([`ci.yml`](.github/workflows/ci.yml)). This runs ruff, then tests on Linux (Python 3.10 to 3.12), macOS and Windows, then the full suite with NumPyro and JAX, then a clean-install quickstart.
- **Container** ([`docker.yml`](.github/workflows/docker.yml)). This builds both images with layer caching and checks they run as non-root and contain no data files. It then runs the quickstart inside each one and publishes to GHCR on version tags.
- **Security.** CodeQL analysis, Dependabot updates for pip, Actions and Docker, and a [security policy](SECURITY.md).
- **Releases** ([`release.yml`](.github/workflows/release.yml)). Builds and checks the distributions, verifies the tag against the package version and publishes release notes from the changelog.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Results explorer

[`notebooks/explorer/`](notebooks/explorer) holds an interactive Jupyter and Colab dashboard for a SYMPHONY results bundle. The bundle is not distributed here; see its README.

## Citation

If you use this software, please cite it using [CITATION.cff](CITATION.cff).

## Licence

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). PPMI data used with this software remains subject to the PPMI Data Use Agreement.
