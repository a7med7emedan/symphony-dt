# Calibration checks

## Simulation-based calibration

```python
from symphony_dt.calibration import SBCDesign, run_sbc, pd_v2_prior_design
design = pd_v2_prior_design(60, chains=4, iter_warmup=500, iter_sampling=500)
result = run_sbc(design, n_replicates=200, method="bonferroni")
print(result.verdict.passed, result.verdict.fwer_uncorrected)
```

Any model can be checked: an `SBCDesign` takes a generator and a fitter. `judge` applies Bonferroni,
Benjamini-Hochberg or no correction and always reports the uncorrected family-wise risk. `rank_histogram_bands`
gives the binomial band for rank histograms.

## Posterior predictive checks

`ppc_summary(draws, stan_data)` reports, per endpoint, interval coverage, the share of values at each scale
boundary against its replicated range, and a PIT histogram. `symphony-dt ppc fit` runs it on a fit's own data.
