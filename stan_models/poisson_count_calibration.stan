data {
  int<lower=0> N_labeled;
  int<lower=0> N_unlabeled;
  vector<lower=0>[N_labeled] predicted_counts_labeled;
  array[N_labeled] int<lower=0> true_counts_labeled;
  vector<lower=0>[N_unlabeled] predicted_counts_unlabeled;
  real<lower=0> epsilon; // small constant to avoid log(0)
}

transformed data {
  /// INCLUDE IN ALL MODELS
  int total_count_labeled = sum(true_counts_labeled);
  int total_N = N_labeled + N_unlabeled;
}

parameters {
  real alpha;
  real<lower=0> beta; // constrained to be positive because otherwise your model is worse than chance lol
}

model {
  // PRIORS
  alpha ~ normal(0, 2); // should probably be tighter. Reflects baseline rate of events (that aren't predicted)
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
  // Posterior predictive: predicted true counts for unlabeled (with uncertainty)
  array[N_unlabeled] int<lower=0> true_counts_unlabeled_rep;
  for (i in 1:N_unlabeled) {
    true_counts_unlabeled_rep[i] = poisson_rng(lambda_unlabeled[i]);
  }
  // Population mean (matches generative_model naming)
  real mean_positives_dataset = (total_count_labeled + sum(true_counts_unlabeled_rep)) * 1.0 / total_N;
  real mean_positives_expected = (total_count_labeled + sum(lambda_unlabeled)) * 1.0 / total_N;

  // also get the maximum count in any image (both labeled and unlabeled)
  real max_count = max(max(true_counts_labeled), max(true_counts_unlabeled_rep));
}
