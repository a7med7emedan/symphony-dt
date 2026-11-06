# symphony-dt 2.0.0 build progress

Resume rule: read this file first, then continue at the first step whose status is not DONE.

## Step 0. Preflight: DONE (2026-11-06)

Commands: `git -C symphony_v2 log -1`, `git -C symphony_v2 status`, `shasum -a 256` on four files,
`gh auth status`, `git clone`, `git shortlog -sne --all`, `cmdstanpy.install_cmdstan(version="2.39.0", dir=".cmdstan")`.

| item | value | verdict |
|---|---|---|
| analysis repo HEAD | 38d812d "CI: GitHub Actions runs every test that needs no PPMI data" | MATCH |
| working tree status | clean | OK |
| stan/symphony_population.stan | 061c1a770c14c13ba9e2e0728938142279949bfd0091c061e0f5b16b75b5c42c | MATCH |
| src/simulate.py | 602b54871f5576bb85efe4fce5cf440174c640672ea71dcb4f074fc2d6e73d6e | MATCH |
| src/sbc.py | f05d98b05c8bfdaa5b33ba872156ec19169fbc6eb0dfe834b02b46045b6c0b0f | MATCH |
| src/stan_data.py | 065819419fe5aed922103ca84972858c57d217846b913bc8a1118c8a7cd6acc7 | MATCH (prompt prefix 06581941) |
| stan/symphony_conditional.stan | 466fde1a32c81dce28cb1706825cd4c53f10562b8d0316fc7b49ccc6365d5d6f | recorded |
| gh auth | a7med7emedan, scopes gist, read:org, repo; git over ssh | OK |
| CmdStan | 2.39.0, project-local `.cmdstan/` (the study's version) | OK |

Repository before v2: default branch `main` at 9b8410a, no tags, licence Apache-2.0, 30 commits.
Contributors (`git shortlog -sne --all`): Ahmed Hemedan only. As sole author he may relicense, so v2 is
BSD-3-Clause throughout. v1 keeps Apache-2.0 on branch `v1` and tag `v1.0-preprint`.

Rule 6 done: annotated tag `v1.0-preprint` and branch `v1` at 9b8410a, both pushed. Work continues on branch `v2`.
Rule 5: the v1 tree was moved with `git mv` into `STALE_20261106T074328Z_v1/`; nothing deleted.

Findings that shape later steps:
- The study's GQ program `symphony_ppc_gq.stan` (sha c149ae44...4f85) and `fit_holdout.py` live only on the
  LIH cluster. Not available here. The package ships its own GQ program beside pd_v2, labelled TESTED.
- No `24_STATION.md` exists in this folder; the global CLAUDE.md roadmap rule has no source here.

## Step 1. Inventory and design: DONE (2026-11-06)

Wrote `.dev/INVENTORY.md` and `.dev/DESIGN.md`.

## Steps 2 to 6. Data, model, fit, gates, forecast, evaluation, governance, CLI: DONE (2026-11-06)

Written so far (2026-11-06, not yet committed): pyproject, constants, config, labels, errors, data
(dataset, instruments, envelope, harmonise, stan_data, simulate), models (registry, pd_v2 with
model.stan, conditional.stan, new forecast_gq.stan sha f06b1b0b...0a41, card.md), diagnostics, fit,
forecast, evaluate (metrics, spec, scoring), governance (manifest, provenance, cards, tripod, reproduce).
Smoke test: simulate_cohort(40) -> build_stan_data reproduces the generating data block exactly
(obs_k, obs_node, K, h, y equal). forecast_gq.stan compiles with CmdStan 2.39.0.
Remaining for Steps 2 to 6: io (readers, adapters, YAML mappings), calibration (SBC, PPC), plots,
report, cli, about; then tests, golden fixtures, privacy gate.

Commits on v2: 07302b8 (package), d47668e (tests and logo).
Checks at d47668e: ruff PASS; mypy --strict PASS (41 files); fast suite 97 passed, 6 slow skipped;
coverage 93% with fit.py excluded (Stan-only); privacy gate 0 findings; model-sha gate MATCH.
Golden tests (exact equality against the code of record run on synthetic cases): stan_data.build,
simulate_replicate (draw for draw, latents, flatten_theta, initial_values), sbc.retained_chains,
chi_squared_uniformity, _thin, _rank, score_fold.py folds 1 to 3, pool_folds.py: 9 of 9 PASS.
Code-of-record digests used: score_fold.py 946eb88b..., pool_folds.py ed133786...
Decisions: thinned GQ input is run by calling the CmdStan binary directly (cmdstanpy rejects a pooled
thin CSV); `about` module renamed `_about` so `symphony_dt.about()` stays a function.

## Step 7. Docs, logo, tutorials: DONE (2026-11-06)
Logo done: docs/assets/logo (tools/make_logo.py; needs DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib).
Docs: mkdocs-material site, 40 pages incl. API reference, validation status (generated), model card
(generated), four executed tutorials (no errors; stderr stripped by tools/clean_notebooks.py).
mkdocs build --strict PASS. Prose check (no dashes, sentences under 25 words) 0 findings.
Slow suite (--run-slow): 6 of 6 PASS in 4 min (fit, gates, resume, ArviZ, exact vs fast, CLI fit/gates/ppc).

## Step 8. Quality gates and repository files: DONE locally (2026-11-06)
ruff PASS, mypy --strict PASS, pytest 100 passed / 6 slow skipped, coverage 93%, privacy gate 0, model-sha MATCH.
Workflows: ci.yml (gates, lint, test matrix ubuntu/macos x 3.11-3.13, docs, package), slow.yml, docs.yml,
release.yml (TestPyPI for rc tags, PyPI for final, attestations, SBOM, GitHub release), codeql.yml,
scorecard.yml, dependabot.yml. Golden tests: exact on the generating platform (Darwin-arm64), 1e-12 elsewhere.

## Step 9. GitHub: NEXT
