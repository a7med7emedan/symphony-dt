# Uncertainty and calibration

A forecast from symphony-dt is a set of draws from the posterior predictive distribution. Each draw combines
uncertainty about the population parameters, about the person's latent path and about measurement noise.

## Calibration

A forecast is calibrated when its stated probabilities match observed frequencies. The package measures this in
three ways.

- **Interval coverage.** The share of held-out scores that fall inside the central 50% and 95% intervals. The
  intervals include their end points.
- **CRPS.** The continuous ranked probability score compares the whole forecast distribution with the observed
  value. Lower is better. The estimator is the empirical form used by `properscoring.crps_ensemble`.
- **PIT.** The probability integral transform of each observation; uniform values indicate calibration.

## Baselines

Two simple methods set the bar. **LOCF** carries the last observed score forward and adds the spread of training
residuals at the same horizon. The **linear mixed model** fits a random intercept and slope per person by EM. A twin
that cannot beat these on held-out data has not earned its complexity.

## Events

Clinically meaningful change is scored as a binary event. The defaults are an increase of 5 or more on
MDS-UPDRS Part III and a decrease of 2 or more on MoCA. Thresholds are arguments. The package reports the Brier score, the Brier skill score
against each baseline and the AUC with ties counted one half.
