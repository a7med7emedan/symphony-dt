# Contributing to symphony-dt

Thank you for helping. This file covers setting up a development environment,
the checks a change must pass, and the rules on data.

## Development setup

Python 3.10 to 3.12 is supported.

```bash
git clone https://github.com/a7med7emedan/symphony-dt.git
cd symphony-dt
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The NumPyro/JAX inference path is optional. Install it only if you work on the
`nuts` commands:

```bash
pip install -e ".[dev,inference]"
```

Install the pre-commit hooks once per clone. The `dev` extra already includes
`pre-commit`:

```bash
pre-commit install
```

## Checks

Run these before you open a pull request. CI runs the same ones.

```bash
ruff check .
pytest
```

Tests that need NumPyro are skipped when the `inference` extra is not installed.

## Data

- Use synthetic data only. `symphony-dt synth` writes PPMI-shaped tables in
  which every value is invented, and `symphony-dt quickstart` runs the whole
  pipeline on them.
- Never commit PPMI data, or any file derived from it, in any form. This covers
  tables, extracts, summaries, figures, notebooks with outputs, and logs.
- Do not add `*.csv` or `*.parquet` files outside `tests/`. A pre-commit hook
  refuses them.
- If you find participant data in the repository, follow `SECURITY.md`.

## Commit messages

Write an imperative subject line of about 50 characters, a blank line, then a
short body that explains why the change is needed.

```text
Fail the gate when accept_prob is missing

NumPyro records only divergences by default, so the mean acceptance
probability was None and the diagnostics gate raised TypeError.
```

## Changelog

Add a line for every user-visible change to `CHANGELOG.md` under the
"Unreleased" heading, in the matching Added, Changed, Fixed or Removed group.
Create the heading if it is not there yet.
