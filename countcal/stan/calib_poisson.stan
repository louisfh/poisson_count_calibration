// Poisson count calibration: regress true counts on detector counts.
//   f_i ~ Poisson(lambda_i),  log(lambda_i) = alpha + beta * log(g_i + g_offset)
// Fit on labeled images only; unlabeled images enter through the posterior
// predictive in generated quantities. Unlabeled detector counts are collapsed
// to unique values (g_unique, with multiplicities g_mult) for speed.
//
// Note: with the canonical log link and a free intercept, the MLE satisfies
// sum(f_lab) = sum(lambda_lab), so the plug-in estimate of the population mean
// is a GREG-style difference estimator (see README).

data {
  int<lower=1> N_lab;
  array[N_lab] int<lower=0> f_lab;
  array[N_lab] int<lower=0> g_lab;
  int<lower=0> U;                       // number of distinct unlabeled detector counts
  array[U] int<lower=0> g_unique;
  array[U] int<lower=1> g_mult;         // how many unlabeled images have each g_unique
  real<lower=0> g_offset;                // log(g + g_offset); g_offset = 1 keeps g = 0 finite
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
}

model {
  alpha ~ normal(0, 2);
  beta ~ normal(1, 1);
  f_lab ~ poisson_log(alpha + beta * x_lab);
}

generated quantities {
  // Finite-population estimands over all N_total images:
  // labeled true counts are known; unlabeled ones are posterior predictive draws.
  real mean_f;
  int max_f = max(f_lab);
  {
    real total = sum(f_lab);
    for (u in 1:U) {
      real lam = exp(alpha + beta * x_unl[u]);
      for (k in 1:g_mult[u]) {
        int y = poisson_rng(lam);
        total += y;
        if (y > max_f) max_f = y;
      }
    }
    mean_f = total / N_total;
  }
}
