# symphony-dt 2.0.0: final report

## 1. Preflight

| item | value | verdict |
|---|---|---|
| analysis repo HEAD | 38d812d | MATCH |
| stan/symphony_population.stan | 061c1a770c14c13ba9e2e0728938142279949bfd0091c061e0f5b16b75b5c42c | MATCH |
| src/simulate.py | 602b54871f5576bb85efe4fce5cf440174c640672ea71dcb4f074fc2d6e73d6e | MATCH |
| src/sbc.py | f05d98b05c8bfdaa5b33ba872156ec19169fbc6eb0dfe834b02b46045b6c0b0f | MATCH |
| src/stan_data.py | 065819419fe5aed922103ca84972858c57d217846b913bc8a1118c8a7cd6acc7 | MATCH |
| CmdStan | 2.39.0, project-local `.cmdstan/` | the study's version |

## 2. Modules and validation labels (152 public objects)

| module | VALIDATED | TESTED | EXPERIMENTAL |
|---|---|---|---|
| data | 9 | 17 | 2 |
| io | 0 | 23 | 0 |
| models + config | 1 | 9 | 0 |
| fit | 0 | 4 | 0 |
| diagnostics | 7 | 4 | 0 |
| forecast | 2 | 6 | 1 |
| evaluate | 21 | 6 | 0 |
| calibration | 3 | 12 | 0 |
| governance | 0 | 10 | 0 |
| report, plots, labels, about | 0 | 15 | 0 |
| total | 43 | 106 | 3 |

## 3. Quality gates

See the CI run on the merged pull request. Local: ruff PASS; mypy --strict PASS (41 files); pytest 100 passed,
6 slow skipped; coverage 93% (fit.py excluded, Stan-only); slow suite 6 of 6 PASS; mkdocs --strict PASS; four
tutorials executed without error; privacy gate 0 findings; model-sha gate MATCH; prose check 0 findings.

## 4. Golden tests

Code of record run read-only on synthetic cases (tests/golden/generate.py): stan_data.build, simulate_replicate
(draw for draw, latents, flatten_theta, initial_values), sbc.retained_chains, chi_squared_uniformity, _thin, _rank,
score_fold.py folds 1 to 3 (score_fold.py sha 946eb88b...), pool_folds.py (ed133786...). 9 of 9 PASS, exact
equality on Darwin arm64. On Linux the EM fit agrees to about 1e-9; CI uses 1e-6 relative there.
