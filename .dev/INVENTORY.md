# Inventory: SYMPHONY v2 analysis code to symphony-dt 2.0

Sources read: analysis repo `symphony_v2/` at 38d812d (read only) and the code of record in
`SYMPHONY_v2_PAPER/04_CODE_OF_RECORD/scripts/` (read only). No participant-level file was opened.

Columns: **PPMI** says whether the module needs PPMI files to run. **Target** is the package module.
**Action** is PORT (same algorithm, same numbers on the same input, golden-tested), REWRITE (same
purpose, new code on the canonical data model), REFERENCE (informs docs or tests, no code shipped) or
OMIT (study-specific operations with no general use).

## Model files

| file | purpose | PPMI | target | action |
|---|---|---|---|---|
| `stan/symphony_population.stan` | population posterior, the model of record | no | `models/pd_v2/model.stan` | SHIP byte-identical, sha-checked |
| `stan/symphony_conditional.stan` | one participant conditioned on fixed globals | no | `models/pd_v2/conditional.stan` | SHIP byte-identical, sha recorded; used by the EXPERIMENTAL fast route |
| cluster `symphony_ppc_gq.stan` (c149ae44...4f85) | standalone GQ, `y_rep_c` and `y_rep_p` | no | `models/pd_v2/forecast_gq.stan` | REWRITE: not on this machine; new program, TESTED |

## `symphony_v2/src`

| module | purpose | PPMI | target | action |
|---|---|---|---|---|
| `stan_data.py` | ragged Stan data block, integer-day grid D39, switches, priors | yes (reads fold files) | `data.stan_data` | PORT on TwinDataset; golden test against `stan_data.build` on synthetic files |
| `simulate.py` | numpy generative model, prior draws, inits | yes (visit patterns) | `data.simulate` | PORT; patterns come from a synthetic visit generator or a user dataset; golden test |
| `sbc.py` | SBC campaign, chain exclusion, gates, rank uniformity, Bonferroni | no for the statistics | `diagnostics`, `calibration.sbc` | PORT the statistics (retained_chains, ranks, thinning, chi-squared, Bonferroni); REWRITE the driver; OMIT cluster checkpoint plumbing |
| `fold_guard.py` | fold-scoped constants, leakage refusal | yes | `data.stan_data.Standardisation` | REWRITE as a training-only standardisation object with a participant digest |
| `leakage_gate.py` | mutation-tested leakage harness | yes | none | REFERENCE (the package refuses test rows in training statistics) |
| `covariate_manifest.py` | declared role of every column | yes | `data.schema` covariate roles | REWRITE as declared roles in TwinDataset |
| `check_privacy.py`, `check_privacy_history.py` | privacy gate for the repo and its history | no | `tools/privacy_gate.py` | REWRITE for this repository |
| `provenance.py`, `measurements.py` | hash-pinned artefacts, provenance ledger | yes | `governance.provenance`, `governance.manifest` | REWRITE (PROV-style JSON, data fingerprints) |
| `prior_spec.py` | emits the prior set, diffs against Table S6 | no | `models.pd_v2` model card and `PriorSpec` | REWRITE |
| `stan_toolchain.py`, `stan_determinism.py` | toolchain proof, byte identity across runs | no | `fit`, `cli check` | REWRITE (`symphony-dt check`, determinism test in slow suite) |
| `liveness.py`, `recover_diagnostics.py` | fit liveness, diagnostics from CSVs on disk | no | `fit` resume, `diagnostics.gates_from_csv` | REWRITE |
| `compute_probe.py` | timing probe on simulated data | no | none | OMIT (cluster sizing) |
| `paths.py` | paths, seeds, supports, ban lists | yes | `constants` | REWRITE (supports, days per year, seeds as defaults) |
| `splits.py` | M35 folds, internal-external arms | yes | `forecast.split_at_anchor`, `evaluate.folds` | REWRITE (participant-level k-fold on TwinDataset) |
| `cohort.py`, `eligibility.py`, `enrolment.py`, `attrition.py` | PPMI cohort selection ledgers | yes | `io.adapters.ppmi` (layout only) | OMIT logic; the adapter maps the user's own export |
| `endpoints.py` | UPDRS-III, MoCA, SCOPA-AUT derivation, SCOPA proration | yes | `data.instruments`, `io.adapters.ppmi` | REWRITE the instrument registry; proration documented, not shipped as a crosswalk |
| `ledd.py` | LEDD harmonisation from medication logs | yes | `data.schema` medication exposure | REWRITE as a user-supplied exposure column; no conversion table shipped |
| `omics.py` | pathway scores, M18 missing means z = 0 | yes | `data.schema` pathway markers | REWRITE (markers are user columns; M18 rule kept) |
| `covariates.py`, `diagnosis.py`, `descriptives.py` | strata, time-varying diagnosis, Table S5 | yes | `evaluate` strata, `governance.cards` | REWRITE generically |
| `time_align.py`, `visit_index.py`, `visit_protocol.py` | dates to elapsed time, master visit index | yes | `io.adapters` (time in days since baseline) | REWRITE |
| `reconcile.py`, `build_block1..4.py`, `anchor_sensitivity.py` | study orchestration and reconciliation | yes | none | OMIT |
| `block5b_*.py`, `block5c_*.py`, `block5d_*.py`, `block5f_*.py`, `probe_*.py` | sampler pathology studies | no | docs (model card known limits) | REFERENCE |

## Code of record, `04_CODE_OF_RECORD/scripts`

| script | purpose | PPMI | target | action |
|---|---|---|---|---|
| `five_fold/score_fold.py` | twin, LOCF, LMM scoring of one fold | yes (`--test` mode runs on synthetic files) | `evaluate.scoring.score_fold` | PORT; golden test in `--test` mode |
| `five_fold/pool_folds.py` | pooled metrics, events, strata, small-cell suppression | yes (`--roots` test mode) | `evaluate.scoring.pool_folds` | PORT; golden test |
| `o30/fold_events.py` | per-fold event metrics and pooled LOCF bins | yes | `evaluate.events` | PORT the event metrics |
| `five_fold/combine_fold.py` | chain exclusion and C2/C3 over all chains of a fold | yes | `diagnostics.assess` | PORT |
| `five_fold/thin_fold.py`, `fold1_O20_fig5/thin8.py` | thin to 400 draws for GQ | no | `forecast.thin_draws` | PORT |
| `five_fold/fit_fold.py` | fold data build and sampling | yes | `fit.fit`, `forecast.split_at_anchor` | REWRITE |
| `five_fold/make_jobs.py`, `*.sbatch`, `*_PASTE.sh` | Slurm job generation | no | none | OMIT |
| `fold1_O20_fig5/score_holdout.py` | fold 1 scorer, ancestor of score_fold | yes | none | REFERENCE (score_fold is its generalisation) |
| `fold1_O20_fig5/fig5_export.py`, `o20_misses.py`, `o30/q37_medians.py` | figure export, post hoc analyses | yes | `plots` | REWRITE generic plots |
| `sensitivity_D-bf/*` | sensitivity pool | yes | none | REFERENCE |
| `numbers/make_numbers.py` | paper numbers | yes | none | OMIT |
| `bridge_g1/*` | bridge gate drafts | yes | none | OMIT |
| `post_hoc/o26_sex.py` | post hoc sex analysis | yes | `evaluate` strata | REFERENCE |

## Golden-test strategy

The code of record imports cluster paths and study modules. The golden tests therefore run it locally in
its own test modes on synthetic inputs written by the package simulator, with study modules replaced by
small stubs that only supply constants and file locations. The outputs are frozen as small synthetic
JSON fixtures under `tests/data/golden/`. CI compares the package against those fixtures. A local test,
`tests/golden/test_regenerate.py`, reruns the code of record from the read-only paper folder when it is
present and checks that the fixtures still match it.
