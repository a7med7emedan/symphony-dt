# Simulation-based calibration

Simulation-based calibration (SBC) checks that a fitting procedure recovers the parameters that generated the
data. Draw a parameter from the prior, simulate data, fit, and record the rank of the true value among the
posterior draws. Over many replicates the ranks are uniform when the procedure is calibrated (Talts et al., 2018).

## Judging many parameters

pd_v2 has many global parameters. Testing each at the 1% level without correction would reject a correct model
often: with 55 parameters the family-wise false-rejection risk is about 42%. The `judge` function makes the
correction explicit.

- **Bonferroni** controls the family-wise error rate. Use it when any single rejection fails the campaign.
- **Benjamini-Hochberg** controls the false discovery rate. Use it when a few rejections are tolerable.
- **None** applies the level to each test, and the reported family-wise risk shows its cost.

## Regimes

The **prior regime** draws parameters from the fitting prior. The **posterior regime** draws them from a
posterior, which concentrates the check where real data put the model. It is valid only when the fitting prior
equals that generating distribution. The package refuses a posterior-regime design that does not assert this.

## Honest reporting

A replicate whose fit fails the gates is recorded as failed and excluded from the ranks. The share of failures is
part of the result.
