# symphony-dt 2.0 design contract

This file fixes the architecture and the public API before code is written. Code that disagrees with it
is a defect in one of the two, and the fix updates both in the same commit.

## 1. Architecture

```
symphony_dt
  constants        endpoint order, scale bounds, priors, switches, grid, seeds (one place)
  labels           ValidationLabel and the @validation decorator; validation_status()
  data             TwinDataset, instruments, schema validation, envelope, harmonisation,
                   Stan data builder, simulator
  io               tabular readers and writers, posterior netCDF, adapters + YAML mappings
  models           registry, ModelConfig, pd_v2 (frozen model.stan, forecast_gq.stan, conditional.stan, card)
  fit              fit(), FitResult, chain-level checkpoint and resume
  diagnostics      gate constants, GateOverride, chain exclusion, assess() -> GateReport
  forecast         split_at_anchor, thin_draws, forecast_exact, forecast_fast, Forecast
  evaluate         coverage, CRPS, MAE, LOCF, LMM by EM, events, bootstrap, score_fold, pool_folds
  calibration      SBC (ranks, uniformity, judge with multiplicity control, driver), PPC
  governance       RunManifest, fingerprints, pre-registration, provenance, cards, TRIPOD+AI
  report           HTML report from manifests and result objects
  plots            matplotlib figures
  cli              typer application `symphony-dt`
  about            about(), ASCII logo
```

Dependencies flow downwards only: `constants` and `labels` import nothing from the package; `data`
imports `constants`; `models` imports `data`; `fit` and `diagnostics` import `models`; `forecast` imports
`fit`; `evaluate` and `calibration` import `data` and `diagnostics`; `governance`, `report`, `plots` and
`cli` may import everything. Importing `symphony_dt` imports no heavy dependency and prints nothing.

## 2. Validation labels

`VALIDATED`: the pd_v2 model and the evaluation code of record, as used in the SYMPHONY v2 study, ported
with golden tests that require identical output. `TESTED`: correct on synthetic data and unit tests, not
validated on a clinical cohort. `EXPERIMENTAL`: works, is tested, but its design is not settled.

Every public function and class carries `Validation: <LABEL>.` in its docstring and the matching
`@validation(<LABEL>)` decorator. `tests/test_labels.py` fails if any public callable lacks either or if
they disagree. `symphony_dt.validation_status()` returns the full table; the docs page is generated from it.

## 3. Data model

`TwinDataset` holds pandas frames, validated on construction:

| frame | required columns | optional columns |
|---|---|---|
| `participants` | `participant_id` (str), `cohort` (str) | `age_baseline_years`, `sex` (0 female, 1 male), `genotype_assay_available`, any covariate |
| `observations` | `participant_id`, `time_days` (int, days since baseline), `endpoint` (str), `value` (float) | `instrument_version` |
| `visits` | `participant_id`, `time_days` | `ledd` and other exposures; visits without scores are nodes too |
| `markers` | `participant_id`, `name`, `value` | pathway or biomarker scores, one value per participant and name |

Plus an `InstrumentRegistry` (name, version, lower, upper, worsening `increase` or `decrease`, boundary
handling `truncated` or `censored`, integer flag, citation) and free metadata. Validation raises
`DataValidationError` with a code (`DV001` to `DV0xx`) and a message that names the column and the rows.
The dataset fingerprint is the sha256 of a canonical, sorted serialisation; the data never leaves.

## 4. Public API (signature, errors, label)

### constants (no callables)
`ENDPOINTS`, `ENDPOINT_INDEX`, `SCALE_BOUNDS`, `DAYS_PER_YEAR`, `H_MAX_YEARS`, `PRIOR_CORRECTED`,
`PRIOR_PUBLISHED`, `CENSORED_PRIMARY`, `STATE_DEPENDENT_SIGMA_PRIMARY`, `STATE_DEPENDENT_SIGMA_SENSITIVITY`,
`LEDD_IN_MU_U_PRIMARY`, `SIGMA_XBAR_PRIMARY`, `PRODUCTION_GRAINSIZE`, `MASTER_SEED`.

### data
- `Instrument(name, version, lower, upper, worsening, boundary="truncated", integer=True, citation="")` TESTED
- `InstrumentRegistry(instruments)`, `.get(name)`, `default_registry() -> InstrumentRegistry` TESTED
- `TwinDataset(participants, observations, visits=None, markers=None, instruments=None, metadata=None)`;
  `.validate()`, `.fingerprint() -> str`, `.subset(ids)`, `.endpoints`, `.participant_ids`,
  `.visit_table()`, `.summary()`; raises `DataValidationError` TESTED
- `check_envelope(dataset, model="pd_v2") -> EnvelopeReport` (warnings with codes `ENV001` to `ENV010`) TESTED
- `Crosswalk(source, target, convert, citation)`; `harmonise(dataset, crosswalks) -> TwinDataset`; a
  crosswalk without a citation raises `ValueError` EXPERIMENTAL
- `Standardisation.fit(dataset, train_ids, *, pathway=()) -> Standardisation` TESTED (port of fold scope)
- `substeps_days(dt_days, h_max_years=0.5) -> ndarray`, `h_max_in_days(h_max_years=0.5) -> tuple[int, int]` VALIDATED
- `build_stan_data(dataset, *, config=ModelConfig.primary(), standardisation=None, pathway=(),
  grainsize=PRODUCTION_GRAINSIZE) -> StanData` VALIDATED (golden against `stan_data.build`)
- `StanData`: `.data`, `.audit`, `.participant_ids`, `.sha256()`, `.to_json(path)`, `StanData.from_json(path)` TESTED
- `draw_prior(rng, n_pathway, hyper=None) -> dict` VALIDATED; `simulate_replicate(patterns, n, theta, rng, *,
  censored, state_dependent_sigma, ledd_in_mu_U, sigma_xbar, h_max=0.5, grainsize=1, latents=None) -> dict`
  VALIDATED; `initial_values(theta_like, rng, n=0, n_steps=0, jitter=0.1) -> dict` VALIDATED;
  `flatten_theta(theta) -> dict[str, float]` VALIDATED
- `synthetic_visit_patterns(n, rng, *, ...) -> list[VisitPattern]` TESTED; `visit_patterns(dataset)` TESTED
- `simulate_cohort(n_participants, *, seed=MASTER_SEED, config=ModelConfig.primary(), patterns=None,
  theta=None, n_pathway=0, cohort_share_pd=0.4) -> SimulatedCohort` (`.dataset`, `.stan_data`, `.theta`,
  `.latents`) TESTED

### io
- `read_table(path, format=None) -> DataFrame`, `write_table(frame, path, format=None)` (csv, tsv, parquet,
  xlsx, arrow/feather; pandas or polars in) TESTED
- `read_dataset(directory) -> TwinDataset`, `write_dataset(dataset, directory, format="parquet")` TESTED
- `write_posterior(idata, path)`, `read_posterior(path)` (netCDF via ArviZ) TESTED
- `load_mapping(name_or_path) -> Mapping`, `list_mappings()` TESTED
- adapters, each `-> TwinDataset`: `from_wide(frame, mapping)`, `from_mapping(frames, mapping)`,
  `from_ppmi_export(directory, mapping="ppmi")`, `from_omop(tables, mapping="omop")`,
  `from_fhir(bundles, mapping="fhir")`, `from_sdtm(qs, dm, mapping="sdtm")` TESTED (synthetic only)

### models
- `ModelConfig(censored, state_dependent_sigma, ledd_in_mu_U, sigma_xbar, prior="corrected",
  h_max_years=0.5)`; `.primary()`, `.a01_sensitivity()` VALIDATED (the study's switch values)
- `ModelSpec`, `get_model(name="pd_v2") -> ModelSpec` (raises `ModelIntegrityError` on sha mismatch),
  `list_models()`, `register_model(spec)` TESTED
- `compile_model(name="pd_v2", program="model", *, force=False) -> CmdStanModel` TESTED
- `model_card(name="pd_v2") -> str` TESTED

### fit
- `fit(stan_data, *, model="pd_v2", chains=4, iter_warmup=1000, iter_sampling=1000, seed=MASTER_SEED,
  chain_ids=None, threads_per_chain=1, parallel_chains=None, inits="prior", output_dir=None, resume=False,
  adapt_delta=0.8, max_treedepth=10) -> FitResult` VALIDATED design, TESTED code
- `FitResult`: `.csv_files`, `.draws(name)`, `.to_inference_data()`, `.gates()`, `.manifest`, `.save()`;
  `load_fit(output_dir) -> FitResult` TESTED

### diagnostics
- `C2_MAX_RHAT = 1.01`, `C3_MIN_ESS_BULK = 400.0`, `MIN_RETAINED_CHAINS = 4`, `TRAP_MULTIPLE = 5.0`
- `GateOverride(reason, author, c2_max_rhat=None, c3_min_ess_bulk=None, min_retained_chains=None,
  trap_multiple=None)` TESTED; never silent, always written to the report and the manifest
- `retained_chains(lp, *, override=None) -> dict` VALIDATED (port of `sbc.retained_chains`)
- `summary_on_chains(csv_files, keep=None) -> DataFrame` VALIDATED (CmdStan `stansummary`)
- `is_global(name) -> bool` VALIDATED
- `assess(csv_files, *, override=None) -> GateReport` VALIDATED (port of `combine_fold`);
  `GateReport.converged` derives only from C2, C3 and the retained-chain count; divergences, tree depth
  saturation, E-BFMI and tail ESS are reported beside it

### forecast
- `split_at_anchor(stan_data, test_index, *, anchor_years=2.0, zero_dose_std=0.0) -> HoldoutSplit` TESTED
  (reimplements the cluster-only `fit_holdout.split` from its written design)
- `thin_draws(csv_files, output, *, total_keep=400) -> Path` VALIDATED (port of `thin_fold`)
- `forecast_exact(fit, split, *, total_keep=400, seed=20261021) -> Forecast` TESTED (the study design; the
  GQ program is a rewrite)
- `forecast_fast(population, history, horizons_years, *, n_particles=2000, n_draws=400, seed=0) -> Forecast`
  EXPERIMENTAL (importance sampling of the participant's latent path under each population draw)
- `Forecast`: `.draws`, `.targets`, `.quantiles(q)`, `.interval(level)`, `.event_probability(...)`,
  `.to_frame()`, `.to_csv(path)`, `.to_netcdf(path)`, `.plot()` TESTED

### evaluate
- `crps_ensemble(samples, y) -> float`, `interval_coverage(samples, y, level) -> float`,
  `central_interval(samples, level)`, `mae_median(samples, y)` VALIDATED
- `fit_lmm_em(groups) -> LMMFit`, `lmm_condition(fit, t, y)` VALIDATED
- `locf_residuals(obs, *, anchor_years, bins, min_bin)` VALIDATED
- `event_indicator`, `event_probability(samples, last_value, endpoint, thresholds=EVENT_THRESHOLDS)`,
  `brier`, `brier_skill`, `auc_with_ties(p, e, w=None)` VALIDATED
- `cluster_bootstrap_weights(n, B, seed)`, `bootstrap_mean(...)` VALIDATED
- `ScoringSpec` (anchor, bins, min_bin, windows, event thresholds, B, seeds, min_cell) TESTED;
  `ScoringSpec.default()` reproduces the code of record
- `score_fold(fit_data, target_data, targets_meta, twin_draws, test_index, *, spec, age=None, fold=1)
  -> FoldScores` VALIDATED (golden against `score_fold.py --test`)
- `pool_folds(fold_scores, *, spec) -> PooledScores` VALIDATED (golden against `pool_folds.py --roots`)

### calibration
- `sbc_rank(draws, truth)`, `thin_for_rank(draws, target)`, `chi_squared_uniformity(ranks, n_draws, bins=20,
  alpha=None)` VALIDATED; `ks_uniformity(ranks, n_draws)` TESTED;
  `rank_histogram_bands(n_ranks, bins, level=0.99)` TESTED
- `judge(per_parameter_p, *, alpha=0.01, method="bonferroni") -> SBCVerdict` (also `"bh"` and `"none"`;
  reports `fwer_uncorrected = 1 - (1 - alpha) ** m`) TESTED
- `SBCDesign`, `run_sbc(design, *, n_replicates, seed)` TESTED; posterior regime EXPERIMENTAL
- `pit_values`, `boundary_statistics`, `ppc_coverage`, `ppc_summary` TESTED

### governance
- `fingerprint(obj) -> str` TESTED; `RunManifest.create(kind, ...)`, `.write(path)`, `.to_html()`,
  `RunManifest.read(path)` TESTED
- `FrozenSpec`, `freeze_spec(spec, path=None)`, `DeviationRecord(reason, author, changes)`,
  `require_frozen(spec, frozen, deviation=None)`; raises `UnfrozenSpecificationError` TESTED
- `ProvenanceDocument`, `.entity`, `.activity`, `.agent`, `.used`, `.generated`, `.write(path)` TESTED
- `reproduce(manifest_path, *, workdir=None) -> ReproduceReport` TESTED
- `model_card_markdown(name)`, `data_card_markdown(dataset, *, min_cell=10)` TESTED
- `tripod_ai_items()`, `tripod_ai_report(answers)` TESTED

### report, plots, cli, about
- `html_report(manifests, *, title, sections=None) -> str`, `write_html_report(path, ...)` TESTED
- `plot_trajectory`, `plot_coverage_by_window`, `plot_crps_differences`, `plot_rank_histogram`,
  `plot_pit`, `plot_effects` TESTED
- CLI commands: `about check validate-data simulate fit gates forecast evaluate sbc ppc report reproduce
  envelope` TESTED
- `about(file=None) -> str` TESTED

## 5. Errors

All package errors subclass `SymphonyError`: `DataValidationError` (with `.code`), `ModelIntegrityError`,
`GateError`, `UnfrozenSpecificationError`, `MappingError`, `CmdStanNotFoundError`.

## 6. Extension points

New models register a `ModelSpec` with their own Stan file and sha. New instruments enter the
`InstrumentRegistry`. New adapters are YAML mappings handled by `from_mapping`, or functions returning a
`TwinDataset`. New crosswalks carry a citation. New scoring metrics take `(samples, y)` arrays.

## 7. Versioning and deprecation

Semantic versioning. The model of record is versioned by name (`pd_v2`) and sha, never edited; a new model
is a new name. A public name is deprecated with a `DeprecationWarning` for at least one minor release before
removal, and the changelog names it. Gate constants change only with a major release.
