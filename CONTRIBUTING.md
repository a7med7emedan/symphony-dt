# Contributing to symphony-dt

Thank you for helping. Please open an issue before a large change, so we can agree the design first.

## Ground rules

1. **No participant data.** Never commit real or derived participant data, identifiers or results from restricted
   cohorts. Use `symphony_dt.data.simulate_cohort` for examples and tests. The privacy gate enforces this.
2. **The model of record is frozen.** Never edit `src/symphony_dt/models/pd_v2/model.stan`. Add new models beside it.
3. **Validation labels.** Every public function and class needs `@validation(...)` and a matching
   `Validation: <LABEL>.` line in its docstring. Never label new code VALIDATED.
4. **Prose.** Plain English, short sentences, British spelling, no em or en dashes.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
pytest                      # fast suite
pytest --run-slow           # compiles Stan and fits small models
mkdocs build --strict
```

## Pull requests

Keep each pull request to one change. Add tests. Update `CHANGELOG.md`. CI must pass, including the privacy gate,
the model-sha gate, ruff, mypy and the docs build.
