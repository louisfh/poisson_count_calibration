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
}

model {
  // PRIORS
  alpha ~ normal(0, 5); // should probably be tighter. Reflects baseline rate of events (that aren't predicted)
  beta ~ normal(1, 1); // maybe beta should be constrained to be positive

  // LIKELIHOOD
  vector[N_labeled] lambda_labeled;
  for (i in 1:N_labeled) {
    lambda_labeled[i] = exp(alpha + beta * log(predicted_counts_labeled[i] + epsilon));
  }
  true_counts_labeled ~ poisson(lambda_labeled);


}
generated quantities {
  vector[N_labeled] lambda_labeled;
  vector[N_unlabeled] lambda_unlabeled;
  for (i in 1:N_labeled) {
    lambda_labeled[i] = exp(alpha + beta * log(predicted_counts_labeled[i] + epsilon));
  }
  for (i in 1:N_unlabeled) {
    lambda_unlabeled[i] = exp(alpha + beta * log(predicted_counts_unlabeled[i] + epsilon));
  }
  // Posterior predictive: predicted true counts (with uncertainty)
  array[N_unlabeled] int<lower=0> true_counts_unlabeled_rep;
  for (i in 1:N_unlabeled) {
    true_counts_unlabeled_rep[i] = poisson_rng(lambda_unlabeled[i]);
  }
  // Average number of rays per image (population mean rate) — for 90% CI
  real mean_rays_per_image = (sum(lambda_labeled) + sum(lambda_unlabeled)) / (N_labeled + N_unlabeled);
}
