// Generative detection model for detector counts:
//   lambda_i ~ Gamma(phi, phi / mu)            (so f_i ~ NegBinomial2(mu, phi) marginally)
//   f_i | lambda_i ~ Poisson(lambda_i)         true count
//   tp_i | f_i ~ Binomial(f_i, p)              true positives
//   fp_i ~ Poisson(lambda_fp)                  false positives
//   g_i = tp_i + fp_i                          detector count
//
// Likelihood, with w = w_marginal in [0, 1]:
//   sum_{labeled}  log p(f_i | g_i)  +  w * sum_{all images} log p(g_i)
// w = 1 is the full joint model (labeled joint + unlabeled marginal).
// w = 0 is the conditional ("cut") model: the detector-count marginal does not
// inform the parameters. Both are valid when labels are chosen based on g,
// because selection on g is ignorable given g. (Dropping the unlabeled images
// but keeping the labeled joint p(f, g) is NOT valid under selection on g.)
//
// Unlabeled detector counts are collapsed to unique values for speed, and the
// NB / Poisson log-pmfs over 0..G_max are computed once per log-density evaluation.

functions {
  // log NB2(k | m, phi) for k = 0..K-1 (kk = k, lgk1 = lgamma(k + 1))
  vector nb2_lpmf_vec(vector kk, vector lgk1, real m, real phi) {
    return lgamma(kk + phi) - lgk1 - lgamma(phi)
           - phi * log1p(m / phi) + kk * (log(m) - log(m + phi));
  }

  // log p(g) = log sum_tp NB2(tp | mu p, phi) Poisson(g - tp | lambda_fp)
  // (binomial thinning of NB2(mu, phi) is NB2(mu p, phi)).
  real log_p_g(int g, vector lnb_tp, vector lpois) {
    return log_sum_exp(lnb_tp[1:(g + 1)] + reverse(lpois[1:(g + 1)]));
  }

  // log p(g | f) = log sum_tp Binomial(tp | f, p) Poisson(g - tp | lambda_fp)
  real log_p_g_given_f(int f, int g, real p, vector lpois) {
    int tp_max = min(f, g);
    vector[tp_max + 1] lp;
    for (tp in 0:tp_max) {
      lp[tp + 1] = lchoose(f, tp) + tp * log(p) + (f - tp) * log1m(p) + lpois[g - tp + 1];
    }
    return log_sum_exp(lp);
  }
}

data {
  int<lower=1> N_lab;
  array[N_lab] int<lower=0> f_lab;
  array[N_lab] int<lower=0> g_lab;
  int<lower=0> U;
  array[U] int<lower=0> g_unique;
  array[U] int<lower=1> g_mult;
  real<lower=0, upper=1> w_marginal;
}

transformed data {
  int N_total = N_lab + sum(g_mult);
  int G_max = max(g_lab);
  if (U > 0) G_max = max(G_max, max(g_unique));
  vector[G_max + 1] kk = linspaced_vector(G_max + 1, 0, G_max);
  vector[G_max + 1] lgk1 = lgamma(kk + 1);
}

parameters {
  real<lower=1e-6, upper=1e4> mu;      // mean true count per image
  real<lower=1e-8> inv_phi;            // 1 / NB2 dispersion
  real<lower=0, upper=1> p;            // detection probability
  real<lower=1e-8> lambda_fp;          // mean false positives per image
}

transformed parameters {
  real phi = 1 / inv_phi;
}

model {
  mu ~ normal(0, 10);
  inv_phi ~ exponential(1);
  p ~ beta(1, 1);
  lambda_fp ~ exponential(0.5);

  vector[G_max + 1] lnb_tp = nb2_lpmf_vec(kk, lgk1, mu * p, phi);
  vector[G_max + 1] lpois = kk * log(lambda_fp) - lambda_fp - lgk1;

  // labeled: log p(f, g) = log p(f) + log p(g | f)
  target += neg_binomial_2_lpmf(f_lab | mu, phi);
  for (i in 1:N_lab) {
    target += log_p_g_given_f(f_lab[i], g_lab[i], p, lpois);
    if (w_marginal < 1) {
      target += -(1 - w_marginal) * log_p_g(g_lab[i], lnb_tp, lpois);
    }
  }
  // unlabeled: log p(g), weighted
  if (w_marginal > 0) {
    for (u in 1:U) {
      target += w_marginal * g_mult[u] * log_p_g(g_unique[u], lnb_tp, lpois);
    }
  }
}

generated quantities {
  // Posterior predictive for each unlabeled image's true count, given its g:
  //   tp | g ~ categorical with weights NB2(tp | mu p, phi) Poisson(g - tp | lambda_fp)
  //   lambda | tp ~ Gamma(phi + tp, phi / mu + p)        (conjugate update)
  //   missed | lambda ~ Poisson(lambda (1 - p))
  //   f = tp + missed
  // Missed animals are NOT independent of tp under NB2, which is why lambda is
  // updated before drawing them.
  real mean_f;
  int max_f = max(f_lab);
  {
    vector[G_max + 1] lnb_tp = nb2_lpmf_vec(kk, lgk1, mu * p, phi);
    vector[G_max + 1] lpois = kk * log(lambda_fp) - lambda_fp - lgk1;
    real total = sum(f_lab);
    for (u in 1:U) {
      int g = g_unique[u];
      vector[g + 1] w = softmax(lnb_tp[1:(g + 1)] + reverse(lpois[1:(g + 1)]));
      for (k in 1:g_mult[u]) {
        int tp = categorical_rng(w) - 1;
        real lam = gamma_rng(phi + tp, phi / mu + p);
        int y = tp + poisson_rng(lam * (1 - p));
        total += y;
        if (y > max_f) max_f = y;
      }
    }
    mean_f = total / N_total;
  }
}
