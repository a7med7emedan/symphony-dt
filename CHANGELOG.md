# Changelog

## Unreleased

### Fixed
- The NumPyro scaffold and conditioning models now use the truncated Gaussian likelihood of the Methods. Before, they scored clipped scores under an untruncated Normal, which understated the density near each scale's floor and ceiling. Posterior draws change as a result.
- Missing endpoints are replaced by a safe value before scoring and then masked, so they contribute neither log density nor NaN gradients.
- Conditioning used `log1p(exp(z))` for the softplus drive, which overflows for large `z`. It now uses the stable form, matching the scaffold.
- `symphony_dt.__version__` reported 0.1.0 while the package was 0.2.0. It is now read from the installed package metadata.
- The coverage docstring now states the nominal level (90%, from the 5th and 95th percentiles). Column names are unchanged.

### Added
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
