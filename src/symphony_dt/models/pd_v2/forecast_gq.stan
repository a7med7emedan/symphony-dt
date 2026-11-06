// forecast_gq.stan, symphony-dt 2.0.
// Standalone generated quantities for pd_v2. It is NOT the model of record and it does not
// change it. The data, transformed data, parameters and transformed parameters blocks are copied
// from model.stan unchanged, so CmdStan's generate_quantities can read a pd_v2 fit's CSV files.
//
// For every observation in the data block (in forecasting, the masked targets) it writes:
//   y_rep_c: the observation model applied to the posterior latent path. Steps after a
//            participant's last fitted observation carry no data, so there the path is a draw
//            from the dynamics: a forecast conditional on the history.
//   y_rep_p: the same observation model on a path re-simulated with fresh innovations from the
//            participant's posterior s0 and b_raw.
//   mu_c:    the observation mean under the posterior path.
// Replicates use the likelihood's convention per endpoint: a truncated normal draw on [L, U] when
// censored[k] = 0, a normal draw clamped to [L, U] when censored[k] = 1.
// The study ran a GQ program of the same design on its cluster; this file is a rewrite of that
// design for the package and is labelled TESTED, not VALIDATED.

functions {
  real sigma_at(real sigma_k, real kappa_k, real s_k, int state_dependent, real xbar_k) {
    if (state_dependent == 0) return sigma_k;
    return sigma_k * exp(kappa_k * (inv_logit(s_k) - xbar_k));
  }

  real tnorm_rng(real mu, real sigma, real L, real U) {
    real pl = normal_cdf(L | mu, sigma);
    real pu = normal_cdf(U | mu, sigma);
    if (pu - pl < 1e-12) {
      // All mass sits beyond one boundary; the truncated draw is that boundary.
      return mu > U ? U : L;
    }
    real u = uniform_rng(pl, pu);
    return fmin(fmax(inv_Phi(u) * sigma + mu, L), U);
  }

  real endpoint_rng(real mu, real sigma, real L, real U, int censored) {
    if (censored == 0) return tnorm_rng(mu, sigma, L, U);
    return fmin(fmax(normal_rng(mu, sigma), L), U);
  }

  matrix latent_path(vector s0_i, vector r, matrix C, vector sigma_xi, vector h,
                     int start, int Ki, matrix eps) {
    matrix[3, Ki + 1] s;
    s[, 1] = s0_i;
    for (kk in 1:Ki) {
      real dt = h[start + kk];
      real rdt = sqrt(dt);
      vector[3] x = inv_logit(s[, kk]);
      for (j in 1:3) {
        real drive = log1p_exp(r[j] + dot_product(C[j], x));
        real xi = exp(-0.5 * dt * square(sigma_xi[j]) + rdt * sigma_xi[j] * eps[j, start + kk]);
        s[j, kk + 1] = s[j, kk] + dt * drive * xi;
      }
    }
    return s;
  }

  real obs_mean(real s_k, int k, vector a, vector b, vector Lk, vector Uk,
                real lambda_ledd, real ledd, int ledd_in_mu_U) {
    real linpred = a[k] + b[k] * s_k;
    if (ledd_in_mu_U == 1 && k == 1) linpred += lambda_ledd * ledd;
    if (k == 2) return Lk[k] + (Uk[k] - Lk[k]) * (1 - inv_logit(linpred));
    return Lk[k] + (Uk[k] - Lk[k]) * inv_logit(linpred);
  }
}

data {
  int<lower=1> N;
  int<lower=0> P;
  int<lower=1> n_steps;
  int<lower=1> n_obs;

  array[N] int<lower=0> K;
  array[N] int<lower=0> step_start;
  array[N] int<lower=0> obs_start;
  array[N] int<lower=0> obs_count;

  vector<lower=0>[n_steps] h;

  array[n_obs] int<lower=1> obs_node;
  array[n_obs] int<lower=1, upper=3> obs_k;
  vector[n_obs] y;

  vector[n_obs] obs_ledd;

  matrix[3, N] dmat;
  matrix[P, N] zmat;

  vector[3] Lk;
  vector[3] Uk;

  real mu_r_prior_mean;
  real<lower=0> mu_r_prior_sd;
  vector[3] a_prior_mean;
  real<lower=0> a_prior_sd;
  int<lower=0, upper=1> prior_only;
  int<lower=1> grainsize;

  array[3] int<lower=0, upper=1> censored;
  int<lower=0, upper=1> state_dependent_sigma;
  vector<lower=0, upper=1>[3] sigma_xbar;
  int<lower=0, upper=1> ledd_in_mu_U;
}

transformed data {
  array[N] int pid;
  for (i in 1:N) pid[i] = i;
}

parameters {
  vector[3] mu_r;
  matrix[3, 3] Gamma;
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

  vector[3] kappa;
  real lambda_ledd;

  matrix[3, N] s0;
  matrix[3, N] b_raw;
  matrix[3, n_steps] eps;
}

transformed parameters {
  matrix[3, 3] C = rep_matrix(0, 3, 3);
  matrix[3, N] bmat;

  C[1, 1] = c_MM;
  C[2, 1] = c_CM;
  C[2, 2] = c_CC;
  C[2, 3] = c_CN;
  C[3, 1] = c_NM;
  C[3, 3] = c_NN;

  for (j in 1:3) bmat[j] = sigma_r[j] * b_raw[j];
}

generated quantities {
  vector[n_obs] y_rep_c;
  vector[n_obs] y_rep_p;
  vector[n_obs] mu_c;
  {
    matrix[3, n_steps] eps_new;
    for (j in 1:3) for (t in 1:n_steps) eps_new[j, t] = std_normal_rng();
    for (i in 1:N) {
      vector[3] r = mu_r + Gamma * dmat[, i] + Amat * zmat[, i] + bmat[, i];
      matrix[3, K[i] + 1] s_c = latent_path(s0[, i], r, C, sigma_xi, h, step_start[i], K[i], eps);
      matrix[3, K[i] + 1] s_p = latent_path(s0[, i], r, C, sigma_xi, h, step_start[i], K[i], eps_new);
      for (oo in 1:obs_count[i]) {
        int idx = obs_start[i] + oo;
        int k = obs_k[idx];
        real sc = s_c[k, obs_node[idx]];
        real sp = s_p[k, obs_node[idx]];
        real m_c = obs_mean(sc, k, a, b, Lk, Uk, lambda_ledd, obs_ledd[idx], ledd_in_mu_U);
        real m_p = obs_mean(sp, k, a, b, Lk, Uk, lambda_ledd, obs_ledd[idx], ledd_in_mu_U);
        mu_c[idx] = m_c;
        y_rep_c[idx] = endpoint_rng(m_c, sigma_at(sigma_obs[k], kappa[k], sc, state_dependent_sigma,
                                                  sigma_xbar[k]), Lk[k], Uk[k], censored[k]);
        y_rep_p[idx] = endpoint_rng(m_p, sigma_at(sigma_obs[k], kappa[k], sp, state_dependent_sigma,
                                                  sigma_xbar[k]), Lk[k], Uk[k], censored[k]);
      }
    }
  }
}
