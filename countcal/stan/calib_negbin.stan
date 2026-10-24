// Negative binomial count calibration: as calib_poisson.stan, but
//   f_i ~ NegBinomial2(mu_i, phi),  log(mu_i) = alpha + beta * log(g_i + g_offset)
// so Var(f | g) = mu + mu^2 / phi allows overdispersion around the calibration curve.

data {
  int<lower=1> N_lab;
  array[N_lab] int<lower=0> f_lab;
  array[N_lab] int<lower=0> g_lab;
  int<lower=0> U;
  array[U] int<lower=0> g_unique;
  array[U] int<lower=1> g_mult;
  real<lower=0> g_offset;
}

transformed data {
  int N_total = N_lab + sum(g_mult);
  vector[N_lab] x_lab;
  vector[U] x_unl;
  for (i in 1:N_lab) x_lab[i] = log(g_lab[i] + g_offset);
  for (u in 1:U) x_unl[u] = log(g_unique[u] + g_offset);
}

parameters {
  real alpha;
  real<lower=0> beta;
  real<lower=0> inv_phi;  // 1 / phi; inv_phi -> 0 recovers the Poisson model
}

transformed parameters {
  real phi = 1 / inv_phi;
}

model {
  alpha ~ normal(0, 2);
  beta ~ normal(1, 1);
  inv_phi ~ exponential(1);
  f_lab ~ neg_binomial_2_log(alpha + beta * x_lab, phi);
}

generated quantities {
  real mean_f;
  int max_f = max(f_lab);
  {
    real total = sum(f_lab);
    for (u in 1:U) {
      real mu = exp(alpha + beta * x_unl[u]);
      for (k in 1:g_mult[u]) {
        int y = neg_binomial_2_rng(mu, phi);
        total += y;
        if (y > max_f) max_f = y;
      }
    }
    mean_f = total / N_total;
  }
}
