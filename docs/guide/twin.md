# The living twin (EXPERIMENTAL)

A `ParticipantTwin` is a twin of one person that keeps learning. It absorbs each new visit at once, without a
population refit, and it checks its own calibration as it goes.

```python
from symphony_dt.twin import ParticipantTwin, PopulationDraws, CalibrationMonitor, AdaptiveConformal
population = PopulationDraws.from_fit(fit_result, n_draws=200)
twin = ParticipantTwin.create(population, history, standardisation=std,
                              monitor=CalibrationMonitor(), conformal=AdaptiveConformal())
twin.observe({"time_days": 1460, "scores": {"updrs3": 31, "moca": 25}})
twin.forecast([1, 2])                       # draws, quantiles and the uncertainty split
twin.milestone_curve("moca", 26, years=5)   # probability of crossing a threshold by each time
twin.next_visit([0.25, 0.5, 1, 1.5])        # which visit time teaches the twin most
twin.save("twin")                           # arrays, history and audit log
```

## How a visit is absorbed

In pd_v2's non-centred form, everything about a person is a vector of standard normals. It holds the starting
state, the rate deviation and every innovation of the latent path. Given a population draw, the latent path is a
deterministic function of that vector.

The twin keeps K population draws and, for each, M particles of that vector. A new visit adds innovations for the
new interval, drawn from the prior. The visit's likelihood is then absorbed by tempering. Its power rises in
stages, each as large as keeps the effective sample size of the cloud at half. After each stage, degenerate draws
are resampled, and every particle takes Metropolis steps with a preconditioned Crank-Nicolson proposal:

\[
z' = \sqrt{1-\beta^2}\, z + \beta\,\xi, \qquad \xi \sim N(0, I).
\]

This proposal leaves the standard-normal prior exactly invariant. The acceptance ratio is therefore the
likelihood ratio of the person's whole history. The step size adapts towards 25 percent acceptance. The scheme is
a sequential Monte Carlo sampler in the sense of Del Moral, Doucet and Jasra (2006). It targets the exact posterior
of the person's latent path. Its moves keep the static rate from collapsing when an instrument is very precise.

Each stage also reweights the population draws by the visit's marginal likelihood. The twin therefore learns
which population parameters suit this person.

## What the twin reports

- **Forecasts** with the variance split, by the law of total variance, into population, person and measurement
  parts.
- **Milestone curves.** The latent state only rises and the links are monotone, so the expected score is monotone in
  time. The probability of crossing a threshold by time t is the probability that the expected score is past it.
- **Next-visit suggestions** ranked by the preposterior expected reduction in forecast variance at a target horizon
  (Lindley, 1956). Choosing when to measure is not a treatment decision.
- **An audit log** of every update: stages, acceptance, effective sample sizes and marginal likelihood.

## Monitoring, recalibration and relearning

Before absorbing a visit, the twin computes the one-step-ahead PIT of each new score. Under a correct model these
PITs are independent and uniform (Dawid's prequential principle).

- `CalibrationMonitor` is an e-process: a mixture of test martingales against bias and dispersion. By Ville's
  inequality, a calibrated twin raises a false alarm with probability at most alpha, however often it is checked.
- `AdaptiveConformal` moves each endpoint's interval level after every outcome, keeping long-run coverage on
  target (Gibbs and Candes, 2021).
- `TwinRegistry` runs many twins with one pooled monitor. Its `RelearnPolicy` refits the population on an alarm or
  after enough new outcomes. It rebuilds every twin on the new draws and records the version, reason and data
  fingerprint in its lineage.

## Evidence so far

Tests check the monitor's false-alarm rate under calibration and its power under bias and dispersion. They check
the conformal layer's long-run coverage under drift, and the held-out calibration of the twin with known
parameters. The [benchmark](../benchmark.md) runs five scenarios. All of this is synthetic. The twin is
EXPERIMENTAL until validated on a clinical cohort, and its forecasts are associational.

## Scope of the novelty claim

Each component exists in the literature. To our knowledge, no tool joins them into one governed loop for a
disease-progression twin. A visit-triggered PD twin on the same three scores appears in a 2026 preprint; the
monitoring, recalibration and scheduling layers go beyond it. A full literature review must precede any
publication claim.
