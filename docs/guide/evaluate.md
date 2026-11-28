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

## Leave-future-out predictive density (TESTED)

Coverage and CRPS judge the forecast draws one target at a time. The log predictive density scores the whole
forecast distribution at the observed value, and it is the natural criterion for comparing models. A forecast
should only be scored on data later than the history it saw. `leave_future_out` therefore scores exact
leave-future-out density, as set out by Bürkner, Gabry and Vehtari (2020). At each anchor the model is refitted on
the data up to that anchor, so no importance sampling is needed.

```python
from symphony_dt.evaluate import anchor_draws, leave_future_out
from symphony_dt.fit import fit
from symphony_dt.forecast import forecast_exact, split_at_anchor

per_anchor = {}
for anchor in (1.0, 2.0, 3.0):
    split = split_at_anchor(stan_data, test_index, anchor_years=anchor)
    result = fit(split.fit, chains=4, output_dir=f"lfo/fit_{anchor:g}")
    forecast = forecast_exact(result, split, output_dir=f"lfo/forecast_{anchor:g}")
    per_anchor[anchor] = anchor_draws(forecast, split)
scores = leave_future_out(per_anchor)
print(scores.to_markdown())
```

Each target scores the log of the mean, over posterior draws, of its observation density. The density is the one
in the model. A truncated endpoint gets a normal renormalised to the scale bounds. A censored endpoint gets a tobit
density with point masses at the bounds. `anchor_draws` rebuilds the observation mean and noise of every draw from
the thinned fit. The rebuilt mean matches the `mu_c` output of the generated-quantities program, and a slow test
checks this.

`scores.pointwise` holds one row per anchor and target, with a Monte Carlo standard error. `scores.joint` holds the
joint density of each participant's future block. Given a latent path the targets are independent, so the joint
density keeps the dependence that the shared path induces. `scores.summary` sums by anchor and endpoint, with a
standard error clustered on participant. Rows with anchor "all" pool the anchors. `scores.compare(other)` gives the
paired difference between two models or arms on their shared targets.

The densities are on the scale of the data, so they compare models fitted to the same targets. They describe the
fit of forecasts to later observations and do not support a causal or treatment claim. For your own forecasts, build
`AnchorDraws` directly from draws of the mean and noise, or call `predictive_log_density`.

## PSIS-LOO and stacking (TESTED)

Leave-one-out cross-validation asks how well a model predicts each observation when it has not seen it. Exact
leave-one-out needs a refit per observation. `psis_loo` instead reweights the full-data posterior draws by Pareto
smoothed importance sampling. The method is set out by Vehtari, Gelman and Gabry (2017) and Vehtari and colleagues
(2024).
The implementation follows the loo R package and ArviZ, and the tests reproduce ArviZ on its eight schools example.

```python
import pandas as pd
from symphony_dt.evaluate import fit_log_likelihood, psis_loo

draws = pd.concat([pd.read_csv(f, comment="#") for f in result.csv_files], ignore_index=True)
log_lik, obs = fit_log_likelihood(draws, split.fit)
by_visit = psis_loo(log_lik)
by_participant = psis_loo(log_lik, groups=obs["i"])
print(by_participant.to_markdown())
```

`fit_log_likelihood` scores each observation under each draw with the model's own truncated or censored density.
`psis_loo` takes any (draws, observations) log likelihood, so other models can be scored in the same way. With
`groups`, the log likelihoods of a group are summed first, so the whole group is left out. Grouping by participant
asks how well the population part of the model predicts a new person. Pass `r_eff`, for example from
`relative_efficiency`, when chains are autocorrelated. It sets the length of the smoothed tail.

Each unit carries a Pareto k. Above the threshold min(1 - 1 / log10(S), 0.7) for S draws, the estimate for that unit
is unreliable, and `reliable` is false. Participant-level parameters make many observations influential. High k
values are therefore to be expected for pd_v2, more so when whole participants are left out. Exact refits or
leave-future-out scores are then the safer route. `k_table` counts units by band, and `n_eff` gives the effective
sample size of the weights.

Stacking combines models rather than choosing one. `stacking_weights` finds the weights of a mixture of the
predictive distributions that maximise its summed log score, following Yao, Vehtari, Simpson and Gelman (2018).
`pseudo_bma_weights` gives the pseudo-BMA+ weights of the same paper, stabilised by the Bayesian bootstrap.
`loo_compare` ranks models, gives paired differences from the best with their standard errors, and adds the stacking
weights.

```python
from symphony_dt.evaluate import loo_compare, stacking_weights

print(loo_compare({"primary": loo_primary, "a01": loo_a01}))
weights = stacking_weights({"primary": lfo_primary, "a01": lfo_a01})   # LFOScores also work
```

Each input is a `LOOResult`, an `LFOScores` or a plain array of pointwise log densities. Units are matched by name, so
leave-future-out scores of two arms are stacked on their shared targets only. Weights describe which mixture of
forecasts best matched the observations. They do not show that one model is the true one.

The building blocks are public: `crps_ensemble`, `interval_coverage`, `fit_lmm_em`, `event_probability`,
`auc_with_ties`, `net_benefit`, `predictive_log_density`, `bootstrap_mean`, `psis` and
`fit_generalized_pareto`.
