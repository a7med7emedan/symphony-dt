# The modeller's workflow

## One study file, one command

```bash
symphony-dt init my-study            # writes my-study/study.yaml with comments
symphony-dt run my-study/study.yaml --dry-run
symphony-dt run my-study/study.yaml
```

The study file names the data and its mapping, the eligibility rules and the model arm. It also sets the held-out
design, the sampler, any gate override, the forecast, scoring overrides and the outputs. `run` executes ten stages
in order: data, eligibility, profile, build, split, fit, gates, forecast, evaluate and report. Each stage writes
its outputs and a manifest into a numbered folder. A stage that already ran with the same settings is skipped, so
a long run resumes where it stopped. `--dry-run` stops before fitting, so a study and its data can be checked in
seconds. `--stages fit,gates` reruns named stages, and `--force` reruns complete ones.

`symphony-dt schema study` prints the JSON Schema of the study file, so other tools can write and check one.

## Eligibility

Rules apply in order. Each one decides, per participant, whether the participant meets it.

| rule | meaning |
|---|---|
| `cohort_in` | cohort label in a list |
| `age_range` | baseline age within a range |
| `required_at_baseline` | listed endpoints scored at day 0 |
| `min_visits` | at least n visits, optionally scoring one endpoint |
| `min_follow_up` | last visit at least so many years after baseline |
| `max_gap` | no gap between visits longer than so many years |
| `min_endpoint_observations` | at least n observations of an endpoint |

`CustomRule` takes any function of the dataset. The result holds the eligible dataset and an attrition table in
the OHDSI style. The table gives the count after each rule and the number each rule removed. It also gives each
rule's gain: the number it alone removes from the final cohort. Counts below the small-cell limit are suppressed.

```python
from symphony_dt.data import apply_eligibility, MinVisits, MinFollowUp, CohortIn
result = apply_eligibility(ds, [CohortIn(("PD",)), MinVisits(3), MinFollowUp(2.0)])
print(result.to_markdown(min_cell=10))
```

## Data formats

`read_table` reads CSV, TSV, Parquet, Excel, Arrow, Stata, SAS, SPSS, JSON and JSON Lines. `read_query` runs SQL
against a DuckDB or SQLite file or any open database connection. `from_meds` and `to_meds` read and write MEDS, the
Medical Event Data Standard. A `redcap` mapping template sits beside the other mappings.

## Data profile

`profile(ds)` reports, per endpoint, the share of visits scored and the shares at the floor and ceiling. It also
gives visits per participant, intervals, follow-up and cohorts, with small cells suppressed.
`symphony-dt profile` prints it, and `profile(ds).plot()` draws it.

## Outputs

`posterior_summary`, `forecast_table` and `scores_table` return tidy tables. `write_tables` writes them as CSV and
Parquet and as one Excel workbook with a sheet per table.

## Plots

Trace, rank and energy plots; predictive intervals; coverage against nominal level; and event reliability. Cohort
trajectories, the attrition flow, the data profile and a one-page participant report. `save_figure` writes PNG,
PDF and SVG in one call. Every plot uses the Okabe-Ito palette, which stays readable with colour blindness.
