# Governance

## Maintainer

symphony-dt is developed and maintained by Ahmed Hemedan, who decides on releases and on changes to the model of
record.

## Data

- No participant data may enter the repository, its issues, its pull requests or its CI logs.
- The PPMI-derived study results stay out of the repository and the documentation until the validation manuscript is
  published, as the PPMI publication policy requires.
- Users obtain PPMI data themselves under the PPMI data use agreement.

## The model of record

`pd_v2/model.stan` is frozen. Its sha256 is checked at import, in the tests and in CI. A change to the model is a
new model with a new name.

## Validation claims

Only the pd_v2 model and the evaluation code of record, as used in the SYMPHONY v2 study, are labelled VALIDATED.
Everything else is TESTED or EXPERIMENTAL and must not be presented as validated.

## Releases

Releases are published from GitHub Actions with PyPI trusted publishing and attestations. A release requires CI to
pass, including the privacy gate and the model-sha gate. Each release attaches a CycloneDX SBOM.
