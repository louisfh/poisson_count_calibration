// Monotone nonparametric calibration curve over the distinct values of g.
//
//   f_i ~ NegBinomial2(mu_l, phi),  l = level(g_i)
//   log mu_l = log mu_{l-1} + (x_l - x_{l-1}) * b_l,   x_l = log(g_l + 1),  b_l > 0
//   log b_l ~ Normal(log b_bar, sigma_b)
//
// So E[f | g] is nondecreasing in g with a locally varying slope on the log-log scale;
// sigma_b -> 0 recovers the NB calibration GLM with slope b_bar. Levels with no labels
// are interpolated / extrapolated through the slopes. Outside the range of labeled g values the slopes are the common slope b_bar (no
// level-specific deviation), so extrapolation follows the average curve instead of
// unconstrained prior draws.

data {
  int<lower=1> N_lab;
  array[N_lab] int<lower=0> f_lab;
  int<lower=1> L;
  vector[L] x_lev;                              // increasing
  array[N_lab] int<lower=1, upper=L> lev_lab;
  int<lower=0> C;
  array[C] int<lower=1, upper=L> lev_unl;
  array[C] int<lower=1> mult_unl;
}

transformed data {
  int N_total = N_lab + sum(mult_unl);
  vector[L - 1] dx = x_lev[2:L] - x_lev[1:(L - 1)];
  // slope j joins levels j and j+1; it gets its own deviation only inside the labeled range
  int lo = min(lev_lab);
  int hi = max(lev_lab);
  vector[L - 1] inside;
  for (j in 1:(L - 1)) inside[j] = (j >= lo && j + 1 <= hi) ? 1 : 0;
}

parameters {
  real eta_lo;                                  // log mean at the lowest labeled level
  real log_bbar;
  real<lower=0> sigma_b;
  vector[L - 1] z_b;
  real log_inv_phi;
}

transformed parameters {
  vector[L] eta;
  real phi = exp(-log_inv_phi);
  {
    vector[L] cum = rep_vector(0, L);
    if (L > 1) cum[2:L] = cumulative_sum(dx .* exp(log_bbar + sigma_b * (inside .* z_b)));
    eta = eta_lo + cum - cum[lo];
  }
}

model {
  eta_lo ~ normal(0, 2);
  log_bbar ~ normal(0, 1);
  sigma_b ~ normal(0, 0.5);
  z_b ~ std_normal();
  log_inv_phi ~ normal(-1, 1.5);
  f_lab ~ neg_binomial_2_log(eta[lev_lab], phi);
}

generated quantities {
  real mean_f;
  int max_f = max(f_lab);
  {
    real total = sum(f_lab);
    for (c in 1:C) {
      real mu = fmin(exp(eta[lev_unl[c]]), 1e6);
      for (k in 1:mult_unl[c]) {
        int y = neg_binomial_2_rng(mu, phi);
        total += y;
        if (y > max_f) max_f = y;
      }
    }
    mean_f = total / N_total;
  }
}
