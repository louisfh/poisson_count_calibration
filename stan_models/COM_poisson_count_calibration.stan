// COM-Poisson (Conway-Maxwell-Poisson) count calibration
// Same as poisson_count_calibration.stan but with dispersion parameter nu:
// nu < 1 => overdispersion, nu > 1 => underdispersion, nu = 1 => Poisson

functions {
  // log approximate normalizing constant (equations 4 and 31 of doi:10.1007/s10463-017-0629-6)
  real log_Z_com_poisson_approx(real log_mu, real nu) {
    real nu2 = nu^2;
    real log_common = log(nu) + log_mu / nu;
    vector[4] resids;
    real ans;
    real lcte = (nu * exp(log_mu / nu)) -
      ((nu - 1) / (2 * nu) * log_mu +
       (nu - 1) / 2 * log(2 * pi()) + 0.5 * log(nu));
    real c_1 = (nu2 - 1) / 24;
    real c_2 = (nu2 - 1) / 1152 * (nu2 + 23);
    real c_3 = (nu2 - 1) / 414720 * (5 * square(nu2) - 298 * nu2 + 11237);
    resids[1] = 1;
    resids[2] = c_1 * exp(-1 * log_common);
    resids[3] = c_2 * exp(-2 * log_common);
    resids[4] = c_3 * exp(-3 * log_common);
    ans = lcte + log(sum(resids));
    return ans;
  }

  real log_k_term(real log_mu, real nu, int k) {
    return (k - 1) * log_mu - nu * lgamma(k);
  }

  real bound_remainder(real k_current_term, real k_previous_term) {
    return k_current_term - log(-expm1(k_current_term - k_previous_term));
  }

  int stopping_criterio_bucket(real k_current_term, real k_previous_term, int k, real leps) {
    if (k % 50 == 0) {
      return (bound_remainder(k_current_term, k_previous_term) >= leps);
    }
    return (1e300 >= leps);
  }

  real log_Z_com_poisson(real log_mu, real nu) {
    real log_Z;
    int k = 2;
    int M = 10000;
    real leps = -23 * log(2);
    vector[M] log_Z_terms;

    if (nu == 1) {
      return exp(log_mu);
    }
    if (nu <= 0) {
      reject("nu must be positive");
    }
    if (nu == positive_infinity()) {
      reject("nu must be finite");
    }
    if (log_mu * nu >= log(1.5) && log_mu >= log(1.5)) {
      return log_Z_com_poisson_approx(log_mu, nu);
    }
    if (bound_remainder(log_k_term(log_mu, nu, M),
                       log_k_term(log_mu, nu, M - 1)) >= leps) {
      reject("nu is too close to zero.");
    }

    log_Z_terms[1] = log_k_term(log_mu, nu, 1);
    log_Z_terms[2] = log_k_term(log_mu, nu, 2);

    while (((log_Z_terms[k] >= log_Z_terms[k - 1]) ||
            (stopping_criterio_bucket(log_Z_terms[k], log_Z_terms[k - 1], k, leps))) &&
           k < M) {
      k += 1;
      log_Z_terms[k] = log_k_term(log_mu, nu, k);
    }
    log_Z = log_sum_exp(log_Z_terms[1 : k]);
    return log_Z;
  }

  real com_poisson_log_lpmf(int y, real log_mu, real nu) {
    if (nu == 1) return poisson_log_lpmf(y | log_mu);
    return y * log_mu - nu * lgamma(y + 1) - log_Z_com_poisson(log_mu, nu);
  }

  real com_poisson_lpmf(int y, real mu, real nu) {
    if (nu == 1) return poisson_lpmf(y | mu);
    return com_poisson_log_lpmf(y | log(mu), nu);
  }

  real com_poisson_lcdf(int y, real mu, real nu) {
    real log_mu;
    real log_Z;
    int yp1;
    if (nu == 1) {
      return poisson_lcdf(y | mu);
    }
    if (nu <= 0) reject("nu must be positive");
    if (nu == positive_infinity()) reject("nu must be finite");
    if (y > 10000) reject("cannot handle y > 10000");
    log_mu = log(mu);
    if (y * log_mu - nu * lgamma(y + 1) <= -36.0) {
      return 0;
    }
    log_Z = log_Z_com_poisson(log_mu, nu);
    if (y == 0) {
      return -log_Z;
    }
    yp1 = y + 1;
    {
      vector[10001] log_num_terms;
      log_num_terms[1] = 0;  // j = 0: mu^0 / (0!)^nu = 1
      for (kk in 2 : yp1) {
        log_num_terms[kk] = (kk - 1) * log_mu - nu * lgamma(kk);
      }
      return log_sum_exp(log_num_terms[1 : yp1]) - log_Z;
    }
  }

  int com_poisson_rng(real mu, real nu) {
    real u = uniform_rng(0, 1);
    int y = 0;
    real cdf;
    if (nu == 1) {
      return poisson_rng(mu);
    }
    cdf = exp(com_poisson_lcdf(y | mu, nu));
    while (cdf < u && y < 10000) {
      y += 1;
      cdf = exp(com_poisson_lcdf(y | mu, nu));
    }
    return y;
  }
}

data {
  int<lower=0> N_labeled;
  int<lower=0> N_unlabeled;
  vector<lower=0>[N_labeled] predicted_counts_labeled;
  array[N_labeled] int<lower=0> true_counts_labeled;
  vector<lower=0>[N_unlabeled] predicted_counts_unlabeled;
  real<lower=0> epsilon; // small constant to avoid log(0)
}

parameters {
  real alpha;
  real<lower=0> beta; // constrained to be positive because otherwise your model is worse than chance lol
  real<lower=0> nu;   // dispersion: < 1 overdispersion, > 1 underdispersion, = 1 Poisson
}

model {
  // PRIORS
  alpha ~ normal(0, 2);
  beta ~ normal(1, 1);
  nu ~ lognormal(0, 1); // nu = 1 is central (standard Poisson)

  // LIKELIHOOD (COM-Poisson)
  vector[N_labeled] lambda_labeled;
  for (i in 1 : N_labeled) {
    lambda_labeled[i] = exp(alpha + beta * log(predicted_counts_labeled[i] + epsilon));
  }
  for (i in 1 : N_labeled) {
    target += com_poisson_lpmf(true_counts_labeled[i] | lambda_labeled[i], nu);
  }
}

generated quantities {
  vector[N_labeled] lambda_labeled;
  vector[N_unlabeled] lambda_unlabeled;
  for (i in 1 : N_labeled) {
    lambda_labeled[i] = exp(alpha + beta * log(predicted_counts_labeled[i] + epsilon));
  }
  for (i in 1 : N_unlabeled) {
    lambda_unlabeled[i] = exp(alpha + beta * log(predicted_counts_unlabeled[i] + epsilon));
  }
  // Posterior predictive: COM-Poisson samples
  array[N_unlabeled] int<lower=0> true_counts_unlabeled_rep;
  for (i in 1 : N_unlabeled) {
    true_counts_unlabeled_rep[i] = com_poisson_rng(lambda_unlabeled[i], nu);
  }
  real mean_rays_per_image = (sum(lambda_labeled) + sum(lambda_unlabeled)) / (N_labeled + N_unlabeled);

  int<lower=0> max_count;
  max_count = 0;
  for (i in 1 : N_labeled) {
    max_count = max(max_count, true_counts_labeled[i]);
  }
  for (i in 1 : N_unlabeled) {
    max_count = max(max_count, true_counts_unlabeled_rep[i]);
  }
}
