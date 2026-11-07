# Changelog

All notable changes are recorded here. The project follows semantic versioning.

## [2.1.1] - 2026-11-06

### Fixed
- A fit with exactly one chain crashed after sampling, so every task of a Slurm array failed. cmdstanpy names a
  one-chain file without its chain id. `fit` now adds the id to the file name, and the chain readers take it from
  the CmdStan header when a name lacks it, so chain folders written by 2.1.0 still load.
- Resume and `load_fit` no longer skip one-chain files.

### Added
- `diagnostics.chain_id` returns the chain id of a CmdStan CSV file.
- A container image for every release at `ghcr.io/a7med7emedan/symphony-dt`, for `linux/amd64` and `linux/arm64`.
  It holds the wheel published on PyPI, CmdStan 2.39.0 and a compiled pd_v2, runs as an unprivileged user and
  carries a signed build provenance. The first image is 2.1.0.

## [2.1.0] - 2026-11-06

### Added
- A study file and one command: `symphony-dt init` and `symphony-dt run study.yaml`, with ten resumable stages,
  a manifest per stage and a dry run.
- Eligibility rules applied in sequence, with an OHDSI-style attrition table, per-rule gain and a flow chart.
- Readers for Stata, SAS, SPSS, JSON and JSON Lines; SQL queries on DuckDB, SQLite or any connection; a MEDS reader
  and writer; a REDCap mapping template.
- A data profile with coverage, visits, intervals and boundary shares.
- Tidy output tables written as CSV, Parquet and one Excel workbook.
- Plots: trace, rank, energy, predictive intervals, coverage calibration, event reliability, cohort trajectories,
  attrition flow, data profile and a participant report; `save_figure` writes several formats at once.
- The living twin (EXPERIMENTAL): a tempered SMC sampler with preconditioned Crank-Nicolson moves on the
  non-centred latent vector; uncertainty split; milestone curves; next-visit suggestions by expected information;
  an anytime-valid calibration e-process; adaptive conformal recalibration; a relearning registry with lineage.
- Running anywhere: run bundles with digests, Slurm job arrays with one chain per task, Docker and Apptainer
  recipes, JSON Schemas, and a JSON-lines twin service.
- An open synthetic benchmark of five scenarios, with LOCF and the mixed model as baselines.

### Fixed
- `fit` with `resume` now touches only the chains it was asked to run, so parallel Slurm tasks cannot clobber one
  another.

## [2.0.1] - 2026-11-06

### Changed
- The brand returns to the SYMPHONY twin tulip: a red tulip, the symbol of Parkinson's disease, whose solid half is
  the patient and whose dotted half is the digital twin. Refined mark, dark variant, app icon, wordmark, favicons,
  social card and an ASCII tulip for `symphony-dt about`.
- Two figures: the twin concept in Parkinson's disease, and the method overview updated for v2.

## [2.0.0] - 2026-11-06

A complete rewrite. Version 1 is preserved on the `v1` branch and the `v1.0-preprint` tag.

### Added
- pd_v2, the SYMPHONY v2 model of record, byte-identical to the study and checked by sha256 at import.
- `TwinDataset`, the canonical long-format data model, with coded validation errors and fingerprints.
- Instrument registry, validity envelope, harmonisation hooks and unit conversion.
- Readers and writers for CSV, TSV, Parquet, Excel, Arrow and polars; netCDF posteriors.
- Adapters with YAML mappings: wide tables, a generic long layout, OMOP CDM, HL7 FHIR R4, CDISC SDTM and the
  PPMI download layout.
- The Stan data builder and the simulator, ported with golden tests against the study code.
- `fit()` with chain-level checkpoint and resume, run manifests and ArviZ output.
- The study's convergence gates with logged overrides.
- Exact-route forecasting with a generated-quantities program, and an experimental fast route.
- The scoring code of record (coverage, CRPS, LOCF, LMM by EM, events, bootstrap, strata), golden-tested.
- SBC toolkit with Bonferroni and Benjamini-Hochberg judges; posterior predictive checks.
- Governance: manifests, provenance, pre-registration freezes, deviation records, model and data cards,
  TRIPOD+AI helper, reproduction command and HTML reports.
- `symphony-dt` command line, documentation site, tutorials and logo.

### Changed
- Licence from Apache-2.0 (v1) to BSD-3-Clause.
