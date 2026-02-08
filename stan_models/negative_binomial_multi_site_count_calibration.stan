// Multi-site negative binomial count calibration.
// N_sites sites; each site has its own (alpha_s, beta_s) and overdispersion phi_s.
// Observations are in flat vectors with site_id_labeled and site_id_unlabeled
// assigning each observation to a site (1-indexed). Ragged: different numbers
// of labeled/unlabeled per site are allowed.
// Negative binomial: variance = mu + mu^2/phi_s (larger phi_s => closer to Poisson).
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
  vector<lower=0>[N_sites] phi;  // site-specific inverse overdispersion (phi_s)
}

model {
  // Priors (per site)
  alpha ~ normal(0, 5);
  beta ~ normal(1, 1);
  phi ~ gamma(2, 0.1);  // positive, allows moderate overdispersion; tune if needed

  // Likelihood: negative binomial with site-specific (alpha, beta, phi)
  for (i in 1:N_labeled) {
    int s = site_id_labeled[i];
    real mu_i = exp(alpha[s] + beta[s] * log(predicted_counts_labeled[i] + epsilon));
    true_counts_labeled[i] ~ neg_binomial_2(mu_i, phi[s]);
  }
}

generated quantities {
  vector[N_labeled] mu_labeled;
  vector[N_unlabeled] mu_unlabeled;
  for (i in 1:N_labeled) {
    int s = site_id_labeled[i];
    mu_labeled[i] = exp(alpha[s] + beta[s] * log(predicted_counts_labeled[i] + epsilon));
  }
  for (i in 1:N_unlabeled) {
    int s = site_id_unlabeled[i];
    mu_unlabeled[i] = exp(alpha[s] + beta[s] * log(predicted_counts_unlabeled[i] + epsilon));
  }

  // Posterior predictive: predicted true counts for unlabeled (with uncertainty)
  array[N_unlabeled] int<lower=0> true_counts_unlabeled_rep;
  for (i in 1:N_unlabeled) {
    int s = site_id_unlabeled[i];
    true_counts_unlabeled_rep[i] = neg_binomial_2_rng(mu_unlabeled[i], phi[s]);
  }

  // Overall average rate (population mean across all images)
  real mean_rays_per_image = (sum(mu_labeled) + sum(mu_unlabeled)) / (N_labeled + N_unlabeled);

  // Per-site average rate: sum of mu at that site / number of images at that site
  vector[N_sites] mean_rays_per_image_site;
  for (s in 1:N_sites) {
    real sum_mu_s = 0;
    int n_s = 0;
    for (i in 1:N_labeled) {
      if (site_id_labeled[i] == s) {
        sum_mu_s += mu_labeled[i];
        n_s += 1;
      }
    }
    for (i in 1:N_unlabeled) {
      if (site_id_unlabeled[i] == s) {
        sum_mu_s += mu_unlabeled[i];
        n_s += 1;
      }
    }
    mean_rays_per_image_site[s] = n_s > 0 ? sum_mu_s / n_s : 0;
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
