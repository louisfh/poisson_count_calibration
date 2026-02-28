// Efficient generative model: same as generative_model.stan but unlabeled likelihood
// TODO: THIS IS NOW OUT OF SYNC WITH GENERATIVE MODEL.STAN. FIX.
// is computed once per unique predicted_count and weighted by how many had that value.
// Unique values and counts are derived in transformed data from predicted_counts_unlabeled.

data {
  int<lower=0> N_labeled;
  int<lower=0> N_unlabeled;
  array[N_labeled] int<lower=0> predicted_counts_labeled;
  array[N_labeled] int<lower=0> true_counts_labeled;
  array[N_unlabeled] int<lower=0> predicted_counts_unlabeled;
  real<lower=0> epsilon; // small constant (kept for same data interface; unused in this model)
  // Unlabeled: 1 = use in likelihood, 0 = labeled-only fit (unlabeled still in generated quantities)
  int<lower=0, upper=1> use_unlabeled_likelihood;
  // If > 0, use only this many unlabeled in likelihood (subsample for speed); 0 = use all
  int<lower=0> max_unlabeled_in_likelihood;
}

transformed data {
  /// INCLUDE IN ALL MODELS
  int total_count_labeled = sum(true_counts_labeled);
  int total_N_unlabeled = N_labeled + N_unlabeled;

  // THE REST
  int n_unlabeled_used = max_unlabeled_in_likelihood > 0
    ? min(N_unlabeled, max_unlabeled_in_likelihood) : N_unlabeled;
  array[n_unlabeled_used] int sorted_pred;
  if (n_unlabeled_used > 0) {
    sorted_pred = sort_asc(segment(predicted_counts_unlabeled, 1, n_unlabeled_used));
  }
  // Count unique predicted_count values and their frequencies (runs in sorted array)
  int N_unique_unlabeled = 0;
  array[n_unlabeled_used] int unique_predicted_counts;
  array[n_unlabeled_used] int counts_per_unique;
  if (n_unlabeled_used > 0) {
    N_unique_unlabeled = 1;
    unique_predicted_counts[1] = sorted_pred[1];
    counts_per_unique[1] = 1;
    for (i in 2:n_unlabeled_used) {
      if (sorted_pred[i] == sorted_pred[i - 1]) {
        counts_per_unique[N_unique_unlabeled] += 1;
      } else {
        N_unique_unlabeled += 1;
        unique_predicted_counts[N_unique_unlabeled] = sorted_pred[i];
        counts_per_unique[N_unique_unlabeled] = 1;
      }
    }
  }
}

parameters {
  real<lower=1e-6> mu;
  real<lower=1e-6, upper=1e6> phi;
  real<lower=0, upper=1> detection_p;
  real<lower=0> avg_false_pos;
}

model {
  // PRIORS
  mu ~ normal(0, 10);
  phi ~ gamma(2, 0.1);
  detection_p ~ beta(1, 1);
  avg_false_pos ~ gamma(2, 1);

  // LIKELIHOOD — labeled data (unchanged)
  for (i in 1:N_labeled) {
    int pred_i = predicted_counts_labeled[i];
    int tc = true_counts_labeled[i];
    target += neg_binomial_2_lpmf(tc | mu, phi);
    int tp_max = min(tc, pred_i);
    real log_prob_pred = negative_infinity();
    for (tp in 0:tp_max) {
      int fp = pred_i - tp;
      log_prob_pred = log_sum_exp(
        log_prob_pred,
        binomial_lpmf(tp | tc, detection_p) + poisson_lpmf(fp | avg_false_pos)
      );
    }
    target += log_prob_pred;
  }

  // LIKELIHOOD — unlabeled: one marginal log_prob per unique predicted_count, weighted by count
  if (use_unlabeled_likelihood) {
    for (u in 1:N_unique_unlabeled) {
      int pred_i = unique_predicted_counts[u];
      real lp = negative_infinity();
      for (tp in 0:pred_i) {
        int fp = pred_i - tp;
        lp = log_sum_exp(lp,
          neg_binomial_2_lpmf(tp | mu * detection_p, phi)
          + poisson_lpmf(fp | avg_false_pos)
        );
      }
      target += counts_per_unique[u] * lp;
    }
  }
}

generated quantities {
  array[N_unlabeled] int<lower=0> true_counts_unlabeled_rep;
  array[N_unlabeled] int<lower=0> predicted_counts_unlabeled_rep;

  for (i in 1:N_unlabeled) {
    int tc_rep = neg_binomial_2_rng(mu, phi);
    int tp_rep = binomial_rng(tc_rep, detection_p);
    int fp_rep = poisson_rng(avg_false_pos);
    true_counts_unlabeled_rep[i] = tc_rep;
    predicted_counts_unlabeled_rep[i] = tp_rep + fp_rep;
  }
  real total_count = total_count_labeled + sum(true_counts_unlabeled_rep);
  real mean_true_count = total_count / (N_labeled + N_unlabeled);
  real mean_predicted_count = mu * detection_p + avg_false_pos;
  real max_count = max(max(true_counts_labeled), max(true_counts_unlabeled_rep));
}
