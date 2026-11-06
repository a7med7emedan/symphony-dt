# Model card: pd_v2

**Name.** pd_v2, the SYMPHONY v2 population model of Parkinson's disease progression.
**Status.** Frozen model of record, VALIDATED as used in the SYMPHONY v2 study. Validation manuscript in preparation.
**File.** `model.stan`, sha256 `061c1a770c14c13ba9e2e0728938142279949bfd0091c061e0f5b16b75b5c42c`.
**Author.** Ahmed Hemedan.

## Purpose and scope

pd_v2 describes how three clinical scores change over time in people with Parkinson's disease or prodromal
Parkinson's disease. The scores are MDS-UPDRS Part III (motor), MoCA (cognition) and SCOPA-AUT (autonomic).
Its outputs are associational forecasts with uncertainty. It makes no causal claim, supports no treatment
decision and is not a medical device.

## Structure in words

Each participant has a latent severity in three domains: motor, cognitive and autonomic. Severity can only
increase. Its speed depends on a participant-level rate, on demographic covariates and on optional pathway
scores. It also depends on the current severity of the domains, through a sparse coupling matrix. Random, positive
multiplicative noise perturbs every step. Each score is a logistic function of its domain's severity,
scaled to the instrument range, observed with Gaussian noise.

## Structure in equations

For participant i and step length h in years, with x = logit^-1(s):

    s_i(t + h) = s_i(t) + h * softplus(r_i + C x_i(t)) * xi
    xi_j = exp(-0.5 h sigma_xi_j^2 + sqrt(h) sigma_xi_j eps_j),  eps ~ N(0, I_3)
    r_i = mu_r + Gamma d_i + A z_i + sigma_r * b_i,  b_i ~ N(0, I_3)
    s_i(0) ~ N(0, I_3)
    y_ik = L_k + (U_k - L_k) * logit^-1(a_k + b_k s_ik)       (UPDRS-III, SCOPA-AUT)
    y_ik = L_k + (U_k - L_k) * (1 - logit^-1(a_k + b_k s_ik)) (MoCA)

The coupling matrix C has six free entries and three structural zeros. Motor is driven only by itself.
Autonomic is driven by motor and itself. Time is in years. Each observed interval is split into equal
sub-steps of at most 0.5 years, counted in integer days.

## Switches of the primary arm

| switch | value | meaning |
|---|---|---|
| A01 `state_dependent_sigma` | 0 | observation noise does not depend on latent severity |
| A02 `ledd_in_mu_U` | 1 | levodopa-equivalent daily dose enters the UPDRS-III observation mean only |
| A03 `censored` | updrs3 0, moca 1, scopa_aut 0 | MoCA uses a censored (tobit) density; the others a truncated density |

The pre-declared sensitivity arm switches A01 on, with the centring constant (0.53, 0.54, 0.55).

## Priors

| parameter | prior |
|---|---|
| mu_r | normal(-2.6, 0.6) per domain (corrected prior) |
| Gamma | normal(0, 0.5) |
| A | normal(0, 0.25) |
| coupling entries | normal(0, 0.25) |
| sigma_r | half-normal(0, 0.5) |
| sigma_xi | half-normal(0, 0.15) |
| a | normal((-2.0, -2.2, -1.8), 0.75) |
| b | lognormal(0, 0.5) |
| sigma_obs | half-normal(0, 0.1 x scale range) |
| kappa, lambda_ledd | normal(0, 0.5) |
| s0, b_raw, eps | standard normal |

## Inputs

Per participant: visit times in whole days since baseline, the three scores where measured and LEDD per visit.
Also centred age in decades, sex and a genotype-assay availability flag. Optional standardised pathway scores.
All centring and scaling constants must come from training participants only.

## Convergence gates

At least four retained chains; max rank-normalised split R-hat at most 1.01; min bulk ESS at least 400.
A chain is excluded when its mean log density falls more than 5.0 within-chain standard deviations below
the best chain. See `symphony_dt.diagnostics`.

## Known limits

- **UPDRS-III floor and heteroscedasticity.** Many motor scores sit at zero, which the truncated density gives
  no point mass. Noise also grows with severity, which the primary arm does not model.
- **One link location for two cohorts.** The PD and prodromal cohorts share one link offset per endpoint. The
  model therefore does not let the score-to-severity map differ by cohort.
- **Process-noise prior conflict.** The half-normal prior on sigma_xi and the data pull in different directions,
  so its posterior is sensitive to the prior.
- **Slow mixing of the motor link slope.** b for UPDRS-III mixes slowly and often sets the effective sample size.
- **Multimodality.** Chains can be trapped in a mode of negligible posterior mass. The chain-exclusion rule exists
  for this. A fit with fewer than four retained chains is a failed fit.
- **Informative dropout** is not modelled.

## Data and governance

The model was developed on Parkinson's Progression Markers Initiative (PPMI) data under the PPMI data use
agreement. No PPMI data ship with this package. Results on PPMI are reported in the validation manuscript in
preparation, not here.

## Verified digests

- `conditional`: `466fde1a32c81dce28cb1706825cd4c53f10562b8d0316fc7b49ccc6365d5d6f`
- `forecast_gq`: `f06b1b0ba0d08a1069881b7aa375d8681b96ce04547071221f8529d1f58d0a41`
- `model`: `061c1a770c14c13ba9e2e0728938142279949bfd0091c061e0f5b16b75b5c42c`
