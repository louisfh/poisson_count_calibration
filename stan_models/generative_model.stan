// Fully generative model for predicted vs true counts:
//   true_count ~ NegBinomial2(mu, phi)    [ecological process; mean=mu, var=mu+mu^2/phi]
//   true_positives ~ Binomial(true_count, detection_p)
//   false_positives ~ Poisson(avg_false_pos)
//   pred_count = true_positives + false_positives

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
  int total_N = N_labeled + N_unlabeled;
}

parameters {
  real<lower=1e-6> mu;                // mean true count (ecological)
  real<lower=1e-6, upper=1e6> phi; // neg_binomial_2 dispersion (larger = closer to Poisson); lower avoids gamma(0)
  real<lower=0, upper=1> detection_p;  // P(detecting a true ray/fish)
  real<lower=0> avg_false_pos;      // mean number of false positive detections
}

model {
  // PRIORS
  mu ~ normal(0, 10);           // vague; tighten if you have scale info
  phi ~ gamma(2, 0.1);          // dispersion (e.g. mean=20; small phi = overdispersion)
  detection_p ~ beta(1, 1);     // uniform prior on detection probability
  avg_false_pos ~ gamma(2, 1);  // vague prior on false positive rate (mean=2, shape=1)

  // LIKELIHOOD — labeled data (observe both true_count and pred_count)
  for (i in 1:N_labeled) {
    int pred_i = predicted_counts_labeled[i];
    int tc = true_counts_labeled[i];
    // true_count (ecological): negative binomial
    target += neg_binomial_2_lpmf(tc | mu, phi);
    // P(pred_i | tc) = sum_{tp=0}^{min(tc,pred_i)} Binomial(tp|tc,detection_p) * Poisson(pred_i-tp|avg_false_pos).
    // We don't observe tp, so we marginalize over it (convolution of Binomial and Poisson).
    // Truncating to tp_max/2 or any fixed offset would drop mass and bias the likelihood; if you need
    // speed for large counts, truncate instead around the mode (e.g. window [tp*-K, tp*+K]) and use
    // a large enough K that the omitted tail is negligible.
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

  // LIKELIHOOD — unlabeled data (observe only pred_count; marginalize over true_count)
  // use_unlabeled_likelihood=0: fit on labeled only. max_unlabeled_in_likelihood>0: subsample (e.g. 5000 of 100k).
// Unlabeled: pred = tp + fp, tp ~ NB2(mu*detection_p, phi), fp ~ Poisson(avg_false_pos)
  if (use_unlabeled_likelihood) {
    int n_unlabeled_used = max_unlabeled_in_likelihood > 0
      ? min(N_unlabeled, max_unlabeled_in_likelihood) : N_unlabeled;

    for (i in 1:n_unlabeled_used) {
      int pred_i = predicted_counts_unlabeled[i];
      real lp = negative_infinity();
      for (tp in 0:pred_i) {
        int fp = pred_i - tp;
        lp = log_sum_exp(lp,
          neg_binomial_2_lpmf(tp | mu * detection_p, phi)
          + poisson_lpmf(fp | avg_false_pos)
        );
      }
      target += lp;
    }
  }
}

generated quantities {
  // Posterior predictive: true counts and predicted counts for unlabeled
  array[N_unlabeled] int<lower=0> true_counts_unlabeled_rep;
  array[N_unlabeled] int<lower=0> predicted_counts_unlabeled_rep;

  for (i in 1:N_unlabeled) {
    int tc_rep = neg_binomial_2_rng(mu, phi);
    int tp_rep = binomial_rng(tc_rep, detection_p);
    int fp_rep = poisson_rng(avg_false_pos);
    true_counts_unlabeled_rep[i] = tc_rep;
    predicted_counts_unlabeled_rep[i] = tp_rep + fp_rep;
  }

  // Population mean: labeled data (known) + realized unlabeled predictions / total N
  real mean_true_count = (total_count_labeled + sum(true_counts_unlabeled_rep)) * 1.0 / total_N;
  real mean_predicted_count = mu * detection_p + avg_false_pos;
  real max_count = max(max(true_counts_labeled), max(true_counts_unlabeled_rep));
}
