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
  array[N_unlabeled] int<lower=0> positives_detected_unlabeled_latent;

  real total_positives_dataset = total_count_labeled;
  real mean_positives_dataset;

  real total_positives_expected = total_count_labeled;
  real mean_positives_expected;

  // Max positives over the whole dataset (labeled observed + unlabeled latent draw)
  int<lower=0> max_count = max(true_counts_labeled);

  for (i in 1:N_unlabeled) {
    int pred_i = predicted_counts_unlabeled[i];

    vector[pred_i + 1] log_w;
    for (tp in 0:pred_i) {
      int fp = pred_i - tp;
      log_w[tp + 1] =
        neg_binomial_2_lpmf(tp | mu * detection_p, phi) +
        poisson_lpmf(fp | avg_false_pos);
    }
    vector[pred_i + 1] w = softmax(log_w);

    int tp_draw = categorical_rng(w) - 1;
    positives_detected_unlabeled_latent[i] = tp_draw;

    int missed_draw = neg_binomial_2_rng(mu * (1 - detection_p), phi);
    int total_pos_draw = tp_draw + missed_draw;

    total_positives_dataset += total_pos_draw;

    // update max
    if (total_pos_draw > max_count)
      max_count = total_pos_draw;

    // expectation accumulation
    real E_tp = 0;
    for (tp in 0:pred_i) E_tp += tp * w[tp + 1];
    total_positives_expected += E_tp + mu * (1 - detection_p);
  }

  mean_positives_dataset  = total_positives_dataset  / total_N;
  mean_positives_expected = total_positives_expected / total_N;
}