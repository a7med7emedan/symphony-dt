# Changelog

All notable changes are recorded here. The project follows semantic versioning.

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
