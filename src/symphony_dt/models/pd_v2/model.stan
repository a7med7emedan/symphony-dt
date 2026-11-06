// symphony_population.stan
// SYMPHONY v2: monotone latent state-space digital twin for PD progression.
// Population-level posterior. Phase P4 of PLAN.md. Implements M25 to M34, M37.
//
// Model, in words:
//   s_i(t) in R^3 is an unconstrained latent severity process, x_i(t) = inv_logit(s_i(t)).
//   Over a step of length h (YEARS):
//       s(t+h) = s(t) + h * softplus(r_i + C x(t)) .* xi,
//       xi_j   = exp(-0.5 h sigma_xi_j^2 + sqrt(h) sigma_xi_j eps_j),   eps ~ N(0, I_3)
//   so E[xi_j] = 1 and every increment is non-negative by construction.
//
// Key implementation identity used throughout:
//   the paper's link is h(p; a, b) = inv_logit(a + b * logit(p)) evaluated at p = inv_logit(s),
//   and logit(inv_logit(s)) = s exactly, so the link collapses to inv_logit(a + b * s).
//   Never compute logit(inv_logit(s)) numerically. See MODEL_MATH.md section 4.
//
// Time is in YEARS. See MODEL_MATH.md section 3 for why days makes the prior degenerate.
//
// The step grid is supplied by the caller. Set one step per observed inter-visit interval to
// reproduce the preprint's single-jump kernel, or subdivide to a fixed maximum sub-step to make
// the process consistent under subdivision. The Stan code is identical either way.

functions {
  // Log density for one endpoint value, under either observation convention.
  //
  // TRUNCATED, the published form. A continuous density renormalised to [L, U]. It assigns
  // probability zero to every single point, the boundary included. That is the defect A03
  // exists to fix: 2,480 of 17,653 MoCA observations, 14.05 percent, sit exactly at 30.
  //
  // CENSORED, A03. The tobit form. Interior values keep the untruncated density and each
  // boundary carries the mass of the latent variable beyond it, so the total integrates to
  // one over the mixed measure. `MODEL_MATH.md` section 4.2 and M81 need both, which is why
  // this is a switch rather than a replacement.
  real endpoint_log_density(real y, real mu, real sigma, real L, real U, int censored) {
    if (censored == 0) {
      return normal_lpdf(y | mu, sigma)
             - log_diff_exp(normal_lcdf(U | mu, sigma), normal_lcdf(L | mu, sigma));
    }
    if (y >= U) return normal_lccdf(U | mu, sigma);
    if (y <= L) return normal_lcdf(L | mu, sigma);
    return normal_lpdf(y | mu, sigma);
  }

  // A01. Observation noise as a function of the latent state rather than of the enrolment
  // label. F10.4 verified the reason: within the Prodromal label, the 242 converters have
  // UPDRS-III mean 14.89 and sd 12.19 against 4.11 and 5.25 for non-converters, with PD at
  // 24.82. The label pools two populations whose variances differ by a factor of five, so
  // stratifying on it conditions on the wrong variable. `inv_logit(s_k)` is the bounded
  // latent severity, so kappa > 0 means noise grows with severity and sigma stays positive.
  // `xbar_k` centres the exponent. Uncentred, sigma_k is the noise at x = 0, which is far
  // from where the latent state lives, so kappa and sigma_obs trade off along the whole
  // range and become multiplicatively confounded. Centred, sigma_k is the noise at the
  // typical state and is identified directly. xbar_k = 0 reproduces the uncentred form
  // exactly, so this is a switch rather than a replacement.
  //
  // The constant comes from the PRIOR PREDICTIVE distribution of x_k, not from a training
  // fold. A prior predictive constant is fixed before any data is seen and is identical
  // across folds, so it raises no M19 or M36 question and needs no fold_guard. It only has
  // to be close: being off by 0.02 is a small effect, being off by 0.4 is not.
  real sigma_at(real sigma_k, real kappa_k, real s_k, int state_dependent, real xbar_k) {
    if (state_dependent == 0) return sigma_k;
    return sigma_k * exp(kappa_k * (inv_logit(s_k) - xbar_k));
  }

  // per-participant log density, sliced over participants for reduce_sum
  real partial_sum(array[] int slice_i, int start, int end,
                        array[] int K, array[] int step_start, array[] int obs_start,
                        array[] int obs_count, vector h,
                        array[] int obs_node, array[] int obs_k, vector y,
                        vector obs_ledd,
                        matrix s0, matrix bmat, matrix eps,
                        vector mu_r, matrix Gamma, matrix Amat,
                        matrix dmat, matrix zmat,
                        matrix C, vector sigma_xi,
                        vector a, vector b, vector sigma_obs,
                        vector kappa, real lambda_ledd, vector sigma_xbar,
                        vector Lk, vector Uk,
                        array[] int censored, int state_dependent_sigma, int ledd_in_mu_U) {
    real lp = 0;
    for (ii in 1:size(slice_i)) {
      int i = slice_i[ii];
      int Ki = K[i];
      vector[3] r = mu_r + Gamma * dmat[, i] + Amat * zmat[, i] + bmat[, i];
      matrix[3, Ki + 1] s;

      s[, 1] = s0[, i];
      for (kk in 1:Ki) {
        real dt = h[step_start[i] + kk];
        real rdt = sqrt(dt);
        vector[3] x = inv_logit(s[, kk]);
        vector[3] inc;
        for (j in 1:3) {
          real drive = log1p_exp(r[j] + dot_product(C[j], x));
          real xi = exp(-0.5 * dt * square(sigma_xi[j])
                        + rdt * sigma_xi[j] * eps[j, step_start[i] + kk]);
          inc[j] = dt * drive * xi;
        }
        s[, kk + 1] = s[, kk] + inc;
      }

      for (oo in 1:obs_count[i]) {
        int idx = obs_start[i] + oo;
        int k = obs_k[idx];
        real s_k = s[k, obs_node[idx]];
        // A02. LEDD enters the UPDRS-III observation mean and nothing else. It is absent from
        // the transition above, which is what keeps M16's non-causal scope claim intact:
        // levodopa changes what the examiner records, not the latent disease process.
        real linpred = a[k] + b[k] * s_k;
        real mu;
        real hlink;
        if (ledd_in_mu_U == 1 && k == 1) linpred += lambda_ledd * obs_ledd[idx];
        hlink = inv_logit(linpred);
        if (k == 2) {
          mu = Lk[k] + (Uk[k] - Lk[k]) * (1 - hlink);   // MoCA decreases with severity
        } else {
          mu = Lk[k] + (Uk[k] - Lk[k]) * hlink;         // UPDRS-III and SCOPA-AUT increase
        }
        lp += endpoint_log_density(
          y[idx], mu,
          sigma_at(sigma_obs[k], kappa[k], s_k, state_dependent_sigma, sigma_xbar[k]),
          Lk[k], Uk[k], censored[k]
        );
      }
    }
    return lp;
  }
}

data {
  int<lower=1> N;                       // participants
  int<lower=0> P;                       // omics pathway scores, may be 0
  int<lower=1> n_steps;                 // total transition steps across all participants
  int<lower=1> n_obs;                   // total observed endpoint values

  // Steps for participant i. Zero is legal and common: 1,343 of the 5,065 have exactly one
  // canonical visit, so they have no transition and their latent state is s_i(0) alone. This
  // was <lower=1>, which rejected them. Dropping them instead would move a population
  // decision into the data block, which BLOCK_05 section 5.5 item 5 and F15 both forbid.
  array[N] int<lower=0> K;
  array[N] int<lower=0> step_start;     // step kk of participant i is h[step_start[i] + kk]
  array[N] int<lower=0> obs_start;      // observation oo of participant i is at obs_start[i] + oo
  array[N] int<lower=0> obs_count;

  vector<lower=0>[n_steps] h;           // step lengths in YEARS, strictly positive

  array[n_obs] int<lower=1> obs_node;   // node index within participant, 1 = baseline, max K[i]+1
  array[n_obs] int<lower=1, upper=3> obs_k;  // 1 = UPDRS-III, 2 = MoCA, 3 = SCOPA-AUT
  vector[n_obs] y;

  // A02. Per-observation LEDD, standardised on the training fold, zero where the endpoint is
  // not UPDRS-III. Carried per observation rather than per visit so the term cannot be
  // applied to an endpoint it does not belong to by an indexing slip.
  vector[n_obs] obs_ledd;

  // Demographic block, M32 and D29. Row 1 centred age in decades, row 2 male indicator,
  // row 3 genotype_assay_available. The third row is D29's named nuisance term: sequencing
  // availability is not random with respect to progression, at 98.0 percent sequenced before
  // 2021 against 40.4 percent after, so it is adjusted for rather than split on. It is on
  // SPLIT_KEY_BAN for the second reason and permitted here for the first.
  matrix[3, N] dmat;
  matrix[P, N] zmat;                    // pathway scores, standardised inside the training fold

  vector[3] Lk;                         // (0, 0, 0)
  vector[3] Uk;                         // (132, 30, 69)

  real mu_r_prior_mean;                 // -2.6, prior predictive calibrated, MODEL_MATH.md 3
  real<lower=0> mu_r_prior_sd;          // 0.6
  vector[3] a_prior_mean;               // (-2.0, -2.2, -1.8), MODEL_MATH.md 3.3
  real<lower=0> a_prior_sd;             // 0.75
  int<lower=0, upper=1> prior_only;     // 1 runs the prior predictive check for gate G3
  int<lower=1> grainsize;

  // The three observation-model additions, as switches on one model rather than three forks.
  // A03: censored[k] = 1 gives endpoint k the tobit density. The BLOCK 5 primary sets
  //      censored = (0, 1, 0), MoCA only, which is what section 5.4 and M81 specify.
  //      The switch is per endpoint because the same defect is present at the UPDRS-III
  //      floor, 2,554 of 24,544 = 10.41 percent, and at the SCOPA-AUT floor, 224 = 1.19
  //      percent. Making those available is not the same as turning them on.
  // A01: state_dependent_sigma = 1 makes sigma_k grow with latent severity, per F10.4.
  // A02: ledd_in_mu_U = 1 adds the LEDD term to the UPDRS-III observation mean only.
  array[3] int<lower=0, upper=1> censored;
  int<lower=0, upper=1> state_dependent_sigma;
  // A01 centring constant, from the prior predictive. Zero gives the uncentred form.
  vector<lower=0, upper=1>[3] sigma_xbar;
  int<lower=0, upper=1> ledd_in_mu_U;
}

transformed data {
  array[N] int pid;
  for (i in 1:N) pid[i] = i;
}

parameters {
  vector[3] mu_r;
  matrix[3, 3] Gamma;                   // 3 domains by 3 demographic columns, D29 widened it
  matrix[3, P] Amat;

  real c_MM;
  real c_CM;
  real c_CC;
  real c_CN;
  real c_NM;
  real c_NN;

  vector<lower=0>[3] sigma_r;
  vector<lower=0>[3] sigma_xi;

  vector[3] a;
  vector<lower=0>[3] b;
  vector<lower=0>[3] sigma_obs;

  // A01 and A02. Both are declared unconditionally and both are given proper priors, so a
  // run with the switch off samples them from the prior and they simply do not enter the
  // likelihood. That keeps one parameter vector across every switch combination, which is
  // what makes the M81 comparison a like-for-like one.
  vector[3] kappa;                      // A01, latent-state slope of log sigma_k
  real lambda_ledd;                     // A02, LEDD coefficient in mu_U

  matrix[3, N] s0;                      // s_i(0), the identifying anchor, prior N(0, I_3)
  matrix[3, N] b_raw;                   // non-centred participant deviations
  matrix[3, n_steps] eps;               // innovations, explicit latent variables
}

transformed parameters {
  matrix[3, 3] C = rep_matrix(0, 3, 3);
  matrix[3, N] bmat;

  // sparse topology: rows are M, C, N. Fixed zeros are c_MC, c_MN, c_NC.
  C[1, 1] = c_MM;
  C[2, 1] = c_CM;
  C[2, 2] = c_CC;
  C[2, 3] = c_CN;
  C[3, 1] = c_NM;
  C[3, 3] = c_NN;

  for (j in 1:3) bmat[j] = sigma_r[j] * b_raw[j];
}

model {
  // priors, Table S6 of the supplement, with the mu_r location corrected, see MODEL_MATH.md 3
  mu_r ~ normal(mu_r_prior_mean, mu_r_prior_sd);
  to_vector(Gamma) ~ normal(0, 0.5);
  if (P > 0) to_vector(Amat) ~ normal(0, 0.25);

  c_MM ~ normal(0, 0.25);
  c_CM ~ normal(0, 0.25);
  c_CC ~ normal(0, 0.25);
  c_CN ~ normal(0, 0.25);
  c_NM ~ normal(0, 0.25);
  c_NN ~ normal(0, 0.25);

  sigma_r ~ normal(0, 0.5);             // half-normal via the lower=0 constraint
  sigma_xi ~ normal(0, 0.15);

  a ~ normal(a_prior_mean, a_prior_sd);
  b ~ lognormal(0, 0.5);
  for (k in 1:3) sigma_obs[k] ~ normal(0, 0.1 * (Uk[k] - Lk[k]));

  // A01 and A02, both new at BLOCK 5 and both recorded in prior_spec.json as additions
  // rather than as changes to Table S6. kappa at N(0, 0.5) allows sigma to move by roughly a
  // factor of e over the full latent range while keeping the no-effect value well inside the
  // bulk. lambda_ledd at N(0, 0.5) is on the same scale as Gamma, since obs_ledd is
  // standardised on the training fold.
  kappa ~ normal(0, 0.5);
  lambda_ledd ~ normal(0, 0.5);

  to_vector(s0) ~ std_normal();
  to_vector(b_raw) ~ std_normal();
  to_vector(eps) ~ std_normal();

  if (prior_only == 0) {
    // reduce_sum_static, not reduce_sum. REPRODUCIBILITY.md R3 requires byte-identical output
    // across two runs, and `reduce_sum` schedules DYNAMICALLY: it treats grainsize as a hint
    // and the realised partition varies run to run. Floating-point addition is not
    // associative, so a different partition sums the log density in a different order and the
    // result differs in the last bits. R3 has held through BLOCK 4 only because no Stan was
    // involved. `reduce_sum_static` partitions statically and reproduces bitwise at a fixed
    // grainsize, independent of thread count, which is what makes R3 survive contact with
    // threading. The cost is that grainsize must now be chosen rather than hinted.
    target += reduce_sum_static(partial_sum, pid, grainsize,
                         K, step_start, obs_start, obs_count, h,
                         obs_node, obs_k, y, obs_ledd,
                         s0, bmat, eps,
                         mu_r, Gamma, Amat, dmat, zmat,
                         C, sigma_xi, a, b, sigma_obs,
                         kappa, lambda_ledd, sigma_xbar, Lk, Uk,
                         censored, state_dependent_sigma, ledd_in_mu_U);
  }
}
