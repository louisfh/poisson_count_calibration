// Multi-site Poisson count calibration.
// N_sites sites; each site has its own (alpha_s, beta_s).
// Observations are in flat vectors with site_id_labeled and site_id_unlabeled
// assigning each observation to a site (1-indexed). Ragged: different numbers
// of labeled/unlabeled per site are allowed.
data {
  int<lower=1> N_sites;
  int<lower=0> N_labeled;
  int<lower=0> N_unlabeled;
  vector<lower=0>[N_labeled] predicted_counts_labeled;
  array[N_labeled] int<lower=0> true_counts_labeled;
  array[N_labeled] int<lower=1, upper=N_sites> site_id_labeled;
  vector<lower=0>[N_unlabeled] predicted_counts_unlabeled;
  array[N_unlabeled] int<lower=1, upper=N_sites> site_id_unlabeled;
  real<lower=0> epsilon;
}

parameters {
  vector[N_sites] alpha;
  vector<lower=0>[N_sites] beta;
}

model {
  // Priors (per site)
  alpha ~ normal(0, 5);
  beta ~ normal(1, 1);

  // Likelihood: each labeled observation uses its site's (alpha, beta)
  for (i in 1:N_labeled) {
    int s = site_id_labeled[i];
    real lambda_i = exp(alpha[s] + beta[s] * log(predicted_counts_labeled[i] + epsilon));
    true_counts_labeled[i] ~ poisson(lambda_i);
  }
}

generated quantities {
  vector[N_labeled] lambda_labeled;
  vector[N_unlabeled] lambda_unlabeled;
  for (i in 1:N_labeled) {
    int s = site_id_labeled[i];
    lambda_labeled[i] = exp(alpha[s] + beta[s] * log(predicted_counts_labeled[i] + epsilon));
  }
  for (i in 1:N_unlabeled) {
    int s = site_id_unlabeled[i];
    lambda_unlabeled[i] = exp(alpha[s] + beta[s] * log(predicted_counts_unlabeled[i] + epsilon));
  }

  // Posterior predictive: predicted true counts for unlabeled (with uncertainty)
  array[N_unlabeled] int<lower=0> true_counts_unlabeled_rep;
  for (i in 1:N_unlabeled) {
    true_counts_unlabeled_rep[i] = poisson_rng(lambda_unlabeled[i]);
  }

  // Overall average rate (population mean across all images)
  real mean_rays_per_image = (sum(lambda_labeled) + sum(lambda_unlabeled)) / (N_labeled + N_unlabeled);

  // Per-site average rate: sum of lambda at that site / number of images at that site
  vector[N_sites] mean_rays_per_image_site;
  for (s in 1:N_sites) {
    real sum_lambda_s = 0;
    int n_s = 0;
    for (i in 1:N_labeled) {
      if (site_id_labeled[i] == s) {
        sum_lambda_s += lambda_labeled[i];
        n_s += 1;
      }
    }
    for (i in 1:N_unlabeled) {
      if (site_id_unlabeled[i] == s) {
        sum_lambda_s += lambda_unlabeled[i];
        n_s += 1;
      }
    }
    mean_rays_per_image_site[s] = n_s > 0 ? sum_lambda_s / n_s : 0;
  }

  // Per-site maximum count in any image (labeled observed max vs unlabeled posterior predictive max)
  array[N_sites] int max_count_site;
  for (s in 1:N_sites) {
    int max_l = 0;
    for (i in 1:N_labeled) {
      if (site_id_labeled[i] == s && true_counts_labeled[i] > max_l) {
        max_l = true_counts_labeled[i];
      }
    }
    int max_u = 0;
    for (i in 1:N_unlabeled) {
      if (site_id_unlabeled[i] == s && true_counts_unlabeled_rep[i] > max_u) {
        max_u = true_counts_unlabeled_rep[i];
      }
    }
    max_count_site[s] = max_l > max_u ? max_l : max_u;
  }
}
