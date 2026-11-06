# Convergence gates

Markov chain Monte Carlo returns draws, not answers. The draws are trustworthy only when the chains agree and
carry enough independent information. symphony-dt fixes the study's rules in one place.

| gate | rule |
|---|---|
| C2 | max rank-normalised split R-hat over the global parameters at most 1.01 |
| C3 | min bulk effective sample size over the global parameters at least 400 |
| chains | at least four retained chains |
| exclusion | a chain is excluded when its mean log density is more than 5.0 within-chain standard deviations below the best chain |

`converged` is true only when all three verdicts pass. Divergences, tree depth saturation, E-BFMI and tail ESS are
reported beside the verdict but do not change it.

R-hat and ESS are computed by CmdStan's own `stansummary` on the retained chains. The package does not
re-implement them, because a re-implementation can agree on easy cases and fail on hard ones.

## Why a chain may be excluded

pd_v2 can trap a chain in a region of negligible posterior mass. The exclusion rule removes such chains by a
pre-declared criterion. A fit with fewer than four retained chains is a failed fit. It is never rerun until it passes,
because that would select on the outcome.

## Changing a threshold

The thresholds are constants. Changing one needs a `GateOverride` with a reason and an author. The override is
written into the gate report and the run manifest, and it raises a warning.
