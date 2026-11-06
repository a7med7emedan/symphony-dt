// symphony_conditional.stan
// SYMPHONY v2: clinic-updated personalisation for ONE participant at ONE conditioning visit.
// Phase P5 and P6 of PLAN.md. Implements M40, M41, M43, M55.
//
// The global parameters Phi are passed as DATA, fixed at one posterior draw index ell.
// Only participant-level quantities are sampled: the initial state, the participant deviation,
// and the innovations on the observed intervals up to the conditioning visit.
//
// This is a small target: 3 + 3 + 3*K parameters, where K is the number of steps up to the
// conditioning visit. For a typical participant K is 3 to 10, so the target has 15 to 39
// dimensions and NUTS converges in a few hundred iterations.
//
// Generated quantities do three jobs in one pass:
//   1. export the latent trajectory at every node, so Delta_{i,n} = max_j IQR(x_j) is computable
//   2. propagate forward to each requested forecast horizon with fresh innovations
//   3. draw posterior predictive outcomes from the truncated Gaussian at each horizon
//
// Suppression rules are NOT applied here. They are applied downstream, on the collected draws,
// so that the gate remains a reporting layer rather than an inference layer.

functions {
  // Identical to symphony_population.stan. Both files must carry the same conventions, or a
  // participant is conditioned under a different observation model than the one the global
  // parameters were fitted with. See MODEL_MATH.md section 4.2 and BLOCK 5 section 5.4.
  real endpoint_log_density(real y, real mu, real sigma, real L, real U, int censored) {
    if (censored == 0) {
      return normal_lpdf(y | mu, sigma)
             - log_diff_exp(normal_lcdf(U | mu, sigma), normal_lcdf(L | mu, sigma));
    }
    if (y >= U) return normal_lccdf(U | mu, sigma);
    if (y <= L) return normal_lcdf(L | mu, sigma);
    return normal_lpdf(y | mu, sigma);
  }

  // A01, state-dependent observation noise. F10.4.
  real sigma_at(real sigma_k, real kappa_k, real s_k, int state_dependent, real xbar_k) {
    if (state_dependent == 0) return sigma_k;
    return sigma_k * exp(kappa_k * (inv_logit(s_k) - xbar_k));
  }

  // observation mean for endpoint k given the unconstrained latent component s_k.
  // A02's LEDD term enters here for k == 1 only, and never in step_forward below.
  real obs_mean(real s_k, int k, vector a, vector b, vector Lk, vector Uk,
                real lambda_ledd, real ledd, int ledd_in_mu_U) {
    real linpred = a[k] + b[k] * s_k;
    if (ledd_in_mu_U == 1 && k == 1) linpred += lambda_ledd * ledd;
    if (k == 2) return Lk[k] + (Uk[k] - Lk[k]) * (1 - inv_logit(linpred));
    return Lk[k] + (Uk[k] - Lk[k]) * inv_logit(linpred);
  }

  // one transition step of length dt given innovation standard normal e
  vector step_forward(vector s, vector r, matrix C, vector sigma_xi, real dt, vector e) {
    vector[3] x = inv_logit(s);
    vector[3] out;
    real rdt = sqrt(dt);
    for (j in 1:3) {
      real drive = log1p_exp(r[j] + dot_product(C[j], x));
      real xi = exp(-0.5 * dt * square(sigma_xi[j]) + rdt * sigma_xi[j] * e[j]);
      out[j] = s[j] + dt * drive * xi;
    }
    return out;
  }

  // truncated normal rng by inverse CDF, exact on [L, U]
  real tnorm_rng(real mu, real sigma, real L, real U) {
    real pl = normal_cdf(L | mu, sigma);
    real pu = normal_cdf(U | mu, sigma);
    real u = uniform_rng(pl, pu);
    return inv_Phi(u) * sigma + mu;
  }

  // A03. Predictive draw under the censored convention: draw the latent value, then clamp.
  // The predictive must use the same convention as the likelihood, or the PIT and coverage
  // statistics that M81 compares are computed against a distribution the model never fitted.
  real endpoint_rng(real mu, real sigma, real L, real U, int censored) {
    if (censored == 0) return tnorm_rng(mu, sigma, L, U);
    return fmin(fmax(normal_rng(mu, sigma), L), U);
  }
}

data {
  int<lower=0> P;
  // Steps from baseline up to the conditioning visit. Zero when the conditioning visit IS
  // the baseline, which is the first anchor of every participant, so it is legal.
  int<lower=0> K;
  int<lower=0> n_obs;                    // observed endpoint values at nodes 1..K+1
  vector<lower=0>[K] h;                  // step lengths in YEARS

  array[n_obs] int<lower=1> obs_node;    // 1 = baseline, up to K+1 = conditioning visit
  array[n_obs] int<lower=1, upper=3> obs_k;
  vector[n_obs] y;
  vector[n_obs] obs_ledd;                // A02, standardised, zero where k is not UPDRS-III

  // Centred age in decades, male indicator, genotype_assay_available. The third entry is
  // D29's named nuisance term and must match the population model's dmat row order exactly,
  // or the fixed Gamma drawn from the population posterior is applied to the wrong column.
  vector[3] d_i;
  vector[P] z_i;                         // pathway scores, standardised on the training fold

  // global parameters Phi, held fixed at draw ell
  vector[3] mu_r;
  matrix[3, 3] Gamma;                    // 3 domains by 3 demographic columns, D29 widened it
  matrix[3, P] Amat;
  matrix[3, 3] C;                        // sparse topology already imposed by the caller
  vector<lower=0>[3] sigma_r;
  vector<lower=0>[3] sigma_xi;
  vector[3] a;
  vector<lower=0>[3] b;
  vector<lower=0>[3] sigma_obs;

  vector[3] Lk;
  vector[3] Uk;

  int<lower=0> n_fc;                     // forecast horizons requested
  vector<lower=0>[n_fc > 0 ? n_fc : 1] fc_dt;   // horizon lengths in YEARS from the conditioning visit
  int<lower=1> fc_substeps;              // subdivide each horizon into this many equal steps

  // A02 at each forecast horizon. LEDD is not forecast by this model, so the caller decides
  // what it carries forward and states it: the anchor's value is the natural default and it
  // makes the forecast a "same treatment" one. Section 5.4 puts LEDD in mu_U only, so this
  // shifts the predicted score without touching the latent trajectory.
  vector[n_fc > 0 ? n_fc : 1] fc_ledd;

  // The three switches, matching symphony_population.stan exactly.
  array[3] int<lower=0, upper=1> censored;
  int<lower=0, upper=1> state_dependent_sigma;
  int<lower=0, upper=1> ledd_in_mu_U;

  // A01 and A02 coefficients, held fixed at the same posterior draw ell as the rest of Phi.
  vector[3] kappa;
  real lambda_ledd;
  vector<lower=0, upper=1>[3] sigma_xbar;   // A01 centring, prior predictive
}

parameters {
  vector[3] s0;
  vector[3] b_raw;
  matrix[3, K] eps;
}

transformed parameters {
  vector[3] r = mu_r + Gamma * d_i + Amat * z_i + sigma_r .* b_raw;
  matrix[3, K + 1] s;
  {
    s[, 1] = s0;
    for (kk in 1:K) s[, kk + 1] = step_forward(s[, kk], r, C, sigma_xi, h[kk], eps[, kk]);
  }
}

model {
  s0 ~ std_normal();
  b_raw ~ std_normal();
  to_vector(eps) ~ std_normal();

  for (o in 1:n_obs) {
    int k = obs_k[o];
    real s_k = s[k, obs_node[o]];
    target += endpoint_log_density(
      y[o],
      obs_mean(s_k, k, a, b, Lk, Uk, lambda_ledd, obs_ledd[o], ledd_in_mu_U),
      sigma_at(sigma_obs[k], kappa[k], s_k, state_dependent_sigma, sigma_xbar[k]),
      Lk[k], Uk[k], censored[k]
    );
  }
}

generated quantities {
  // latent state at every node, for trajectory export and for Delta_{i,n}
  matrix[3, K + 1] x_latent;

  // forecasts: latent state and predictive draw at each requested horizon
  matrix[3, n_fc > 0 ? n_fc : 1] x_fc = rep_matrix(0, 3, n_fc > 0 ? n_fc : 1);
  matrix[3, n_fc > 0 ? n_fc : 1] mu_fc = rep_matrix(0, 3, n_fc > 0 ? n_fc : 1);
  matrix[3, n_fc > 0 ? n_fc : 1] y_fc = rep_matrix(0, 3, n_fc > 0 ? n_fc : 1);

  for (kk in 1:(K + 1)) x_latent[, kk] = inv_logit(s[, kk]);

  if (n_fc > 0) {
    for (f in 1:n_fc) {
      vector[3] sf = s[, K + 1];
      real hsub = fc_dt[f] / fc_substeps;
      for (m in 1:fc_substeps) {
        vector[3] e;
        for (j in 1:3) e[j] = std_normal_rng();
        sf = step_forward(sf, r, C, sigma_xi, hsub, e);
      }
      x_fc[, f] = inv_logit(sf);
      for (k in 1:3) {
        mu_fc[k, f] = obs_mean(sf[k], k, a, b, Lk, Uk,
                               lambda_ledd, fc_ledd[f], ledd_in_mu_U);
        y_fc[k, f] = endpoint_rng(mu_fc[k, f],
                                  sigma_at(sigma_obs[k], kappa[k], sf[k],
                                           state_dependent_sigma, sigma_xbar[k]),
                                  Lk[k], Uk[k], censored[k]);
      }
    }
  }
}
