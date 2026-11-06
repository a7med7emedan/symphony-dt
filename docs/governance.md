# Governance and data access

## No data in this package

symphony-dt contains no participant data. Tests, examples and tutorials use the package's own simulator. A
privacy gate runs before every commit and in continuous integration. It blocks data files outside the synthetic
test fixtures, JSON files over 1 MB and identifier patterns such as PATNO.

## Obtaining PPMI data

The pd_v2 model was developed on data from the Parkinson's Progression Markers Initiative (PPMI). To use PPMI
data, apply for access at [ppmi-info.org](https://www.ppmi-info.org/access-data-specimens/download-data), accept
the data use agreement and download the study data yourself. Follow the PPMI publication policy when you report
results. Then read your download with `symphony_dt.io.from_ppmi_export`.

## Results on PPMI

The study's results will appear in the validation manuscript in preparation. Until then these pages report no
PPMI-derived number.

## Licence and authorship

symphony-dt 2.x is released under the BSD-3-Clause licence. Version 1 remains available on the `v1` branch and the
`v1.0-preprint` tag under its original Apache-2.0 licence. Developed by Ahmed Hemedan.

## Reporting

Use `symphony_dt.governance.tripod_ai_report` to check your own study against TRIPOD+AI (Collins et al., 2024).
