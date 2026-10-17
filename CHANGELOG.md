# Changelog

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
