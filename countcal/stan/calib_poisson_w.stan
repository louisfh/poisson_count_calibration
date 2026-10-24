// Survey-weighted pseudo-posterior for the Poisson calibration GLM (Savitsky & Toth 2016):
// each labeled image's log-likelihood is multiplied by its design weight (normalised to sum
// to the number of labeled images). The weighted score equations make the plug-in mean
// design-consistent under PPS even when the GLM is wrong; the posterior spread is not
// design-calibrated (that would need a sandwich adjustment), which the benchmark measures.

data {
  int<lower=1> N_lab;
  array[N_lab] int<lower=0> f_lab;
  array[N_lab] int<lower=0> g_lab;
  vector<lower=0>[N_lab] w;
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
}

model {
  alpha ~ normal(0, 2);
  beta ~ normal(1, 1);
  for (i in 1:N_lab) {
    target += w[i] * poisson_log_lpmf(f_lab[i] | alpha + beta * x_lab[i]);
  }
}

generated quantities {
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
