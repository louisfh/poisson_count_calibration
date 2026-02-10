data {
  int<lower=0> N_labeled;
  int<lower=0> N_unlabeled;
  vector<lower=0>[N_labeled] predicted_counts_labeled;
  array[N_labeled] int<lower=0> true_counts_labeled;
  vector<lower=0>[N_unlabeled] predicted_counts_unlabeled;
}

parameters {
  real alpha;
}

model {
  // PRIOR
  alpha ~ normal(0, 5);

  // LIKELIHOOD (beta fixed to 1: lambda = exp(alpha) * (g + epsilon))
  vector[N_labeled] lambda_labeled;
  for (i in 1:N_labeled) {
    lambda_labeled[i] = exp(alpha) + predicted_counts_labeled[i];
  }
  true_counts_labeled ~ poisson(lambda_labeled);
}

generated quantities {
  vector[N_labeled] lambda_labeled;
  vector[N_unlabeled] lambda_unlabeled;
  for (i in 1:N_labeled) {
    lambda_labeled[i] = exp(alpha) + predicted_counts_labeled[i];
  }
  for (i in 1:N_unlabeled) {
    lambda_unlabeled[i] = exp(alpha) + predicted_counts_unlabeled[i];
  }
  vector[N_unlabeled] pred_counts_unlabeled_rep;
  for (i in 1:N_unlabeled) {

  }
  // Average number of rays per image (population mean rate) — for 90% CI
  real mean_rays_per_image = (sum(lambda_labeled) + sum(lambda_unlabeled)) / (N_labeled + N_unlabeled);
  // also get the maximum count in any image (both labeled and unlabeled)
  // int<lower=0> max_count;
  // max_count = 0;
  // for (i in 1:N_labeled) {
  //   max_count = max(max_count, true_counts_labeled[i]);
  // }
  // for (i in 1:N_unlabeled) {
  //   max_count = max(max_count, true_counts_unlabeled_rep[i]);
  // }
}
