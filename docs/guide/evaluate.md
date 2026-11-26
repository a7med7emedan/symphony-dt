# Evaluation

`score_fold` scores one held-out fold exactly as the study's code of record does, and `pool_folds` pools folds.
Golden tests require identical output.

```python
from symphony_dt.evaluate import ScoringSpec, freeze_spec, score_fold
spec = ScoringSpec.default()
frozen = freeze_spec(spec, "frozen_spec.json")   # before any target is scored
scores = score_fold(split.fit.data, split.targets.data, split.meta, forecast.draws,
                    split.test_index, spec=spec, frozen=frozen)
scores.write("scores")
```

Three methods are scored on the same targets. They are the twin, LOCF with horizon-binned training residuals and
the linear mixed model, fitted by EM on training participants only. Metrics are 50% and 95% interval coverage, CRPS and
the absolute error of the median. Windows are the next visit, one year, two years and all horizons. Intervals come
from a participant-cluster bootstrap with 1,000 replicates.

`pool_folds` adds event metrics: Brier, Brier skill and AUC. It adds strata by sex, cohort, age band, visits before
the anchor and follow-up. It suppresses any cell with fewer than 10 participants.

## Calibration by subgroup (TESTED)

A twin can be calibrated overall and miscalibrated in a subgroup, such as older participants or one site.
`subgroup_calibration` scores the per-target table within each level of any grouping you name. It reports 50% and
95% interval coverage and CRPS for the twin, LOCF and the mixed model, each with a participant-cluster bootstrap
interval. For the twin it adds the mean and spread of the PIT values and a Kolmogorov-Smirnov test of uniformity.

```python
from symphony_dt.evaluate import age_band, subgroup_calibration
ts = scores.target_scores
site = {i: site_of[i] for i in ts["i"].unique()}   # your own participant-to-site lookup
result = subgroup_calibration(
    ts,
    {"sex": "sex", "cohort": "cohort", "age band": age_band(ts["age"]), "site": site},
    twin_draws=forecast.draws,
)
print(result.to_markdown(metric="c95"))
result.plot(metric="c95", endpoint="moca")
```

A grouping is a column of the table, a mapping from participant index to label, or one label per row. Genotype
groups and sites come as mappings, because the scoring table does not hold them. An overall row is always included,
so each level can be read against the whole fold.

Three cautions apply. A cell with fewer than `min_cell` participants, 10 by default, is suppressed in the table,
the Markdown and the plot. The PIT values of one participant are correlated, so the uniformity p-values are
approximate. Many cells mean many tests, so the p-values are adjusted by Benjamini-Hochberg by default; pass
`multiplicity="bonferroni"` or `"none"` to change this.

`pool_folds` already reports twin coverage by sex, cohort and age band for the study. `subgroup_calibration`
extends that check to any grouping, to CRPS and PIT, and to both baselines.

## Decision curve analysis (TESTED)

Brier scores and AUC summarise the event forecasts. They do not say whether flagging a participant at a given risk
would be worth it. `decision_curve` reports net benefit across threshold probabilities, as proposed by Vickers
and Elkin (2006). A target is flagged when its forecast event probability is at least the threshold. Net benefit is
the share of true flags minus the share of false flags, weighted by the odds of the threshold.

```python
from symphony_dt.evaluate import decision_curve
curve = decision_curve(scores.target_scores)
print(curve.to_markdown(thresholds=(0.05, 0.1, 0.2, 0.3)))
curve.plot(endpoint="updrs3")
```

The twin, LOCF and the mixed model are scored on the same targets. Two references are always included: flag every
target ("all") and flag none ("none"). The literature calls these "treat all" and "treat none". A forecast is of use
at a threshold only where its curve lies above both references. Each estimate carries a participant-cluster
bootstrap interval, and the same resamples serve every curve of an endpoint.

The events follow the scoring code of record. They are a UPDRS-III worsening of 5 points or more and a MoCA decline
of 2 points or more. An endpoint with fewer than `min_cell` participants, events or non-events is suppressed. Net
benefit describes forecasts against observed events. It does not show that acting on a flag changes any outcome.
`net_benefit` scores a single threshold for your own probabilities.

The building blocks are public: `crps_ensemble`, `interval_coverage`, `fit_lmm_em`, `event_probability`,
`auc_with_ties`, `net_benefit` and `bootstrap_mean`.
