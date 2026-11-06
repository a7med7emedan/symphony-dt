# Contributing

Contributions are welcome. Please read `CONTRIBUTING.md` in the repository.

- Open an issue before a large change.
- Never add participant data, real or derived. The privacy gate will refuse it.
- Never edit `models/pd_v2/model.stan`. New models live beside it.
- Every public function needs a validation label in its docstring and decorator.
- Run `pre-commit run --all-files` and `pytest` before you open a pull request.
- Write plain English: short sentences, British spelling, no dashes as punctuation.
