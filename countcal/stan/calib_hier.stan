// Calibrated-Bayes count calibration (Little 1993; Zheng & Little 2003):
// model the outcome flexibly in the design variable g, so that selection on g is
// ignorable without relying on the parametric form of E[f | g].
//
//   f_i ~ NegBinomial2(mu_i, phi)
//   log mu_i = alpha + beta * log(g_i + 1)        parametric calibration curve
//              + u[level(g_i)]                     deviation for each distinct value of g
//              + v[site_i]                         site random effect
//   u_l ~ Normal(0, tau_u),  v_s ~ Normal(0, tau_v)   (partial pooling, non-centred)
//
// With many labels at a value of g, that level's mean is estimated from its own labels
// (Bayesian post-stratification on g); with few, it shrinks to the curve. Levels and
// sites with no labels are predicted from the curve and the random-effect distributions.
// use_level / use_site switch the two random effects on and off.

data {
  int<lower=1> N_lab;
  array[N_lab] int<lower=0> f_lab;
  int<lower=1> L;                                  // distinct g values in the population
  vector[L] x_lev;                                 // log(g + 1) for each level
  array[N_lab] int<lower=1, upper=L> lev_lab;
  int<lower=1> S;
  array[N_lab] int<lower=1, upper=S> site_lab;
  int<lower=0> C;                                  // unlabeled (level, site) cells
  array[C] int<lower=1, upper=L> lev_unl;
  array[C] int<lower=1, upper=S> site_unl;
  array[C] int<lower=1> mult_unl;
  int<lower=0, upper=1> use_level;
  int<lower=0, upper=1> use_site;
}

transformed data {
  int N_total = N_lab + sum(mult_unl);
}

parameters {
  real alpha;
  real<lower=0> beta;
  real<lower=0> inv_phi;
  vector[use_level ? L : 0] z_lev;
  array[use_level] real<lower=0> tau_lev;
  vector[use_site ? S : 0] z_site;
  array[use_site] real<lower=0> tau_site;
}

transformed parameters {
  real phi = 1 / inv_phi;
  vector[L] eta_lev = alpha + beta * x_lev;
  vector[S] v_site = rep_vector(0, S);
  if (use_level) eta_lev += tau_lev[1] * z_lev;
  if (use_site) v_site = tau_site[1] * z_site;
}

model {
  alpha ~ normal(0, 2);
  beta ~ normal(1, 1);
  inv_phi ~ exponential(1);
  z_lev ~ std_normal();
  tau_lev ~ normal(0, 0.5);
  z_site ~ std_normal();
  tau_site ~ normal(0, 0.5);
  f_lab ~ neg_binomial_2_log(eta_lev[lev_lab] + v_site[site_lab], phi);
}

generated quantities {
  real mean_f;
  int max_f = max(f_lab);
  {
    real total = sum(f_lab);
    for (c in 1:C) {
      real mu = fmin(exp(eta_lev[lev_unl[c]] + v_site[site_unl[c]]), 1e6);
      for (k in 1:mult_unl[c]) {
        int y = neg_binomial_2_rng(mu, phi);
        total += y;
        if (y > max_f) max_f = y;
      }
    }
    mean_f = total / N_total;
  }
}
