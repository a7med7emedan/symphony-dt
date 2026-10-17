# Changelog

## Unreleased

### Fixed
- The NumPyro scaffold and conditioning models now use the truncated Gaussian likelihood of the Methods. Before, they scored clipped scores under an untruncated Normal, which understated the density near each scale's floor and ceiling. Posterior draws change as a result.
- Missing endpoints are replaced by a safe value before scoring and then masked, so they contribute neither log density nor NaN gradients.
- Conditioning used `log1p(exp(z))` for the softplus drive, which overflows for large `z`. It now uses the stable form, matching the scaffold.
- `symphony_dt.__version__` reported 0.1.0 while the package was 0.2.0. It is now read from the installed package metadata.
- `select_tau_conf` targeted 95% coverage while every engine produces central 90% intervals. The target is now 0.90. Missing target scores are skipped rather than counted as misses.

### Changed
- Coverage columns are renamed to `cov90_updrs3`, `cov90_moca`, `cov90_scopa` and `cov90_overall`, matching the central 90% interval (5th to 95th percentile) they score. The values are unchanged: numbers reported as `cov95_*` by 0.2.0 and earlier are 90% coverage.
- **Deprecated:** `cov95_*` is still written alongside `cov90_*`, with identical values, for one release and will then be removed. The `evaluation_coverage` contract, `parity-check`, `aggregate-all` and `aggregate-evaluation` accept either name, so old and new run directories can be merged. `parity-check` warns when a table or notebook uses only the old names.
- `nuts forecast --tau-conf` has no silent default any more. Without it or `--tau-conf-json`, 1.0 is used and a warning says that gating rule 3 is inactive.
- The `tau_conf` grid diagnostics are keyed `cov90_le_*` (were `cov95_le_*`).

### Added
- `symphony-dt nuts select-tau-conf`: computes Delta and the one-step 90% interval for each training-fold anchor from the conditioning outputs, selects `tau_conf`, and writes it with the coverage and fraction kept at every grid point to JSON.
- `nuts forecast --tau-conf-json` reads that file. Posterior forecasts gain a `delta_iqr` column with the latent uncertainty that rule 3 compares with `tau_conf`.
- `ColumnRule.aliases` in the schema contracts, an `evaluation_coverage` contract checked by `parity-check`, and coverage column references in notebook introspection.
- Tests for the coverage aliases, and for `select-tau-conf` and `--tau-conf-json` on quickstart data.
- `symphony-dt --version`, and help text for the top-level command and the `nuts` group.
- Chains run in parallel only when JAX has one device per chain, otherwise sequentially, without a warning.
- Tests for the bounded likelihood against the closed-form truncated normal density, and for the version flag.
- Logo, wordmark and method figure under `docs/assets/`.
- `codemeta.json`, a richer `CITATION.cff` abstract, and expanded package keywords and project links.

## 0.2.0

### Fixed
- `nuts forecast` no longer crashes on every conditioned run. NumPyro only records divergences by default, so `accept_prob` was never collected, its mean was `None`, and the gate raised `TypeError`. Conditioning now requests `accept_prob` (draws are unchanged), and the gate treats any missing or non-finite diagnostic as a failed run instead of crashing.
- Conditioning `diagnostics.json` could contain bare `NaN`, which is not valid JSON. Non-finite values are now written as `null`.
- Removed three unused patch modules and a hard-coded local input path from `configs/config.demo.yaml`.

### Added
- `symphony-dt synth`: synthetic, PPMI-shaped input tables. Every value is invented.
- `symphony-dt quickstart`: synthetic data, demo run and parity check in one command.
- `symphony-dt run --input-dir`: override the configured input directory per run.
- `aggregate-all` warns when shard runs are mixed with demo or full runs.
- Dockerfile (non-root, code only, optional NumPyro/JAX), `.dockerignore`.
- GitHub Actions CI: lint, tests on Python 3.10 to 3.12, inference extras, synthetic end-to-end, Docker build and run.
- End-to-end NUTS test on synthetic data, quickstart and diagnostics tests.
- Results explorer notebook unpacked into `notebooks/explorer/`. The results bundle is opt-in via `SYMPHONY_BUNDLE_GDRIVE_ID` instead of a hard-coded file ID.

## 0.1.0
- Initial pipeline: demo, full and shard modes, aggregation, parity checks, optional NumPyro inference.
