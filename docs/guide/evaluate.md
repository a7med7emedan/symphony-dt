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

The building blocks are public: `crps_ensemble`, `interval_coverage`, `fit_lmm_em`, `event_probability`,
`auc_with_ties` and `bootstrap_mean`.
