# Combining model-based and design-based inference for detector counts

## The problem, stated as an objective

A detector gives a count g for every image; hand-counting gives the true count f for n
images chosen by a labeling design. We want the population mean (equivalently total) of f
over all N images. A *procedure* is a labeling design plus an estimator. The objective
implemented in `scripts/leaderboard.py`:

> Make 90% intervals as narrow as possible — measured relative to random labeling + the PPI
> difference estimator at the same n — while keeping coverage: mean coverage >= 0.88 and no
> dataset/n cell below 0.80, over cells with n >= 25 (n = 10 reported separately).

Why this form:

- **Coverage is the constraint, width the objective.** RMSE alone rewards confident wrong
  answers; coverage alone rewards infinitely wide intervals. With 100 repeats per cell a
  coverage estimate has a standard error of about 0.03, so the thresholds tolerate Monte
  Carlo noise around 0.90 but not systematic undercoverage.
- **Design and estimator are judged together**, because the best estimator depends on how
  the labels were chosen (results 1 and 2 below).
- **Worst case over datasets**, because in practice you do not know which regime you are in:
  a procedure that fails on one realistic detector is not safe to recommend.
- The population **max** was dropped from the objective: no method estimates it reliably
  across real datasets (it depends entirely on the model's tails).

Each dataset is a fixed finite population and coverage is over repeated labeling draws —
exactly the frequentist calibration that Little's calibrated Bayes asks Bayesian procedures
to have.

## What the benchmark showed (before the new approaches)

1. **Pure design-based** (PPI, DISCount): unbiased under any design, but intervals are
   normal-theory and undercover at small n and with heavy-tailed counts.
2. **Pure model-based** (calibration GLMs, generative detection model): efficient when right,
   but (a) labeling by detector count turns misspecification into bias that does not shrink
   with n, and (b) letting the unlabeled images' detector counts inform the parameters
   (the joint generative model) makes the model confidently wrong.
3. The Poisson GLM sits in between: its point estimate is a GREG estimator under SRS, so it
   is robust there, but its intervals are model-based.

## Three ways to combine them

### 1. Calibrated Bayes (Little 2006, 2011): Bayesian inference, design-robust models

Keep inference Bayesian, but choose the model so that it has good repeated-sampling
properties, and check them (this benchmark is that check). The central rule is to **put the
design variables in the model, flexibly**: here selection depends on g, so the model for
f | g must not lean on a parametric form in g (Little 1993, post-stratification;
Zheng & Little 2003, penalised spline of the selection size under PPS). Implemented:

- `cb_ps`: NB calibration curve + a partially pooled deviation for every distinct g value
  (Bayesian post-stratification on g). Dense g values are estimated from their own labels,
  sparse ones shrink to the curve.
- `cb_ps_site`: the same + site random effects — the multi-site model, and the natural route
  to per-site estimates (small-area estimation is where Bayes clearly beats design-based).
- Less parametric versions (forked approach): stratified Pólya posterior / Bayesian bootstrap
  within g-strata, monotone calibration curves — see `countcal/approaches/bnp.py`.

### 2. Model-assisted inference: design-based inference, model-based efficiency

Keep inference design-based and use the model only as a working predictor (Särndal et al.
1992; Wu & Sitter 2001; PPI is the special case m(g) = g). The estimator
mean(m(g), all images) + design-weighted mean(f - m(g), labeled) is design-unbiased
whatever m is; a better m only narrows the interval. Flexible m needs cross-fitting so the
residuals are honest. Implemented: `ma_poisson`, `ma_flex` (+ small-sample intervals,
`ma_poisson_jack`).

### 3. Weighting the likelihood (pseudo-posterior)

Multiply each labeled image's log-likelihood by its design weight (Savitsky & Toth 2016):
`poisson_w`. Gives design-consistent point estimates but posterior spread that is not
design-calibrated — included as the comparison Little argues against.

### Design as part of the procedure

Because every design here selects on g only, it is ignorable for any model that conditions
on g flexibly; that frees the design to be chosen for efficiency. Forked approach:
stratification on g and two-phase Neyman allocation (`countcal/approaches/designs_learned.py`).

### Interval calibration

Forked approach (`countcal/approaches/intervals.py`): t quantiles and jackknife variances fix
most small-n undercoverage for the design-based / model-assisted estimators. Heavy tails that
the detector misses (Serengeti herds at threshold 0.5) cannot be fixed by any interval rule:
the herds are rarely in the sample.

## Results

See `results/leaderboard.md` (filled in by the benchmark).

## Recommended next steps

(Written after the full benchmark; see the README.)
