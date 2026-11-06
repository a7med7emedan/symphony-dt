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

## Step 9. GitHub: IN PROGRESS
v2 pushed; PR https://github.com/a7med7emedan/symphony-dt/pull/2 (v2 -> main) opened.
Repo description, topics (7), homepage set; Pages enabled with build_type workflow; main protected
(required checks: gates, lint, tests ubuntu/macos 3.12, docs, package; no force push, no deletion);
environments testpypi and pypi created. Social preview image needs the web UI (human step).
Local wheel built, twine check PASS, clean-venv install ran about + quickstart (fit, gates, GQ, score).


CI round 1 (597263c): gates, lint, docs, package PASS; Ubuntu tests FAILED 2 golden tests (EM fit agrees
to about 1e-9 across platforms; one comparison had missed the tolerant helper). Fixed in af0f0d4 (exact on
the generating platform, 1e-6 relative elsewhere). Pushed; CI round 2 running.

NEXT (resume here): when PR #2 CI is green, merge it (merge commit, no force). Then on main: tag v2.0.0rc1,
push the tag; release.yml publishes to TestPyPI (environment testpypi). Install from TestPyPI in a clean venv,
run quickstart and `symphony-dt about`. Then tag v2.0.0 for PyPI and the GitHub release. If trusted
publishing is not configured, the publish job fails: leave the tag unpublished and report the settings.

CI round 2 (af0f0d4): golden LMM comparisons still failed across platforms (EM stops a few iterations
apart; values move in the 6th digit) and ArviZ 0.x test path on Python 3.11. Fixed in 44adc23 (1e-4
relative, 1% iteration window off-platform; exact on the generating platform). Verified locally on
Python 3.11 + ArviZ 0.23: 100 passed. 950dfca sets version 2.0.0rc1 (release.yml requires tag = version).
Final release needs a small PR bumping to 2.0.0 before tagging v2.0.0.

## Steps 9 and 10. DONE (2026-11-06)
PR #2 merged by Ahmed (beb182d). v2.0.0rc1 tag -> TestPyPI with attestations; installed from TestPyPI, about +
quickstart PASS. Rule from Ahmed: all PRs, merges and releases under his account, no bots. Dependabot PRs #3 to #7
closed with branches removed, 24 bot-triggered workflow runs deleted, dependabot.yml and the bot GitHub-release job
removed (PR #8, merged by Ahmed, 66db8cc, version 2.0.0). v2.0.0 tag -> PyPI with attestations; GitHub release
created by Ahmed with wheel, sdist and SBOM; digests equal PyPI's. pip install symphony-dt PASS in a clean venv.
Docs live at https://a7med7emedan.github.io/symphony-dt/ (coverage badge 93%).
Brand: Ahmed asked to restore the v1 twin tulip and the method figure. tools/make_brand.py and make_figures.py;
PR #9 (brand branch) open, version 2.0.1.dev0 on main after merge.
Human steps: upload docs/assets/brand/social-preview.png in repo settings; Zenodo integration for a DOI.
2.0.1 requested by Ahmed: release the tulip brand and figures so PyPI and `symphony-dt about` show them.

## Release 2.1 (branch v2.1-modeller): IN PROGRESS
Plan: docs/roadmap.md (sections 1 to 8). Ahmed asked for: modeller inputs/outputs/plots/control, formats,
eligibility; a mature interacting self-learning twin; interoperability on any machine; real algorithmic novelty;
whatever makes it the best tool in the domain. Order of work:
1 eligibility (src/symphony_dt/data/eligibility.py written) 2 formats + profile 3 study.yaml runner
4 outputs + plots 5 ParticipantTwin (SIR update, VoI next visit, milestones, uncertainty split, audit log) +
TwinMonitor (e-process on PIT, adaptive conformal, relearn policy) 6 compute (bundle, Slurm, Apptainer, Docker,
JSON Schemas, JSON stdio twin interface) 7 benchmark scenarios 8 tests, docs, CI, PR under Ahmed (no bots).
All new features EXPERIMENTAL or TESTED; novelty claims hedged; prior SYMPHONY preprint credited.
Done on v2.1-modeller (uncommitted at time of writing): eligibility.py, io formats + read_query + meds.py +
redcap.yaml, data/profile.py, plots (save_figure, profile, attrition, trace, rank, energy, cohort trajectories,
ppc intervals, coverage calibration, reliability, participant report), outputs.py, study.py (dry run verified,
stage skipping verified). Next: src/symphony_dt/twin/ (core, lifecycle, monitor, registry), then compute, bench.
Twin core finding (2026-11-06): SIR + Liu-West collapses when an instrument is very precise (MoCA sigma_obs
~0.3): tempering hits its cap, rate spread collapses, long-horizon coverage 0.2-0.4 in the oracle benchmark.
Rewriting the core as a tempered SMC sampler with pCN moves on the non-centred latent vector (s0, b_raw, eps),
which is exactly N(0, I) a priori; MH acceptance uses the full-history likelihood ratio (Del Moral et al. 2006).
SMC-pCN twin core DONE: oracle benchmark (n=100, 35 test, 5 scenarios, 500 particles, 14 s): twin 95% coverage
0.92-0.99, 50% 0.40-0.74; best or tied CRPS in most cells; largest gains on sparse and dropout. Next: CLI, tests
(monitor false-alarm rate, ACI coverage, twin calibration), docs pages, commit, PR under Ahmed.
