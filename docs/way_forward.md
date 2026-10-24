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

### 4. Design-based point, Bayesian variance (what worked best)

The surprise of the full benchmark was that the dominant failure of design-based intervals
on real data is not bias but the *variance estimate*: with an accurate detector most
residuals f − g are exactly 0, so a small labeled sample often has zero residual variance
and the interval collapses (Channel Islands: coverage 0.31 at n = 10, 0.54 at n = 25).
Model-based intervals never collapse because their variance has a prior. The hybrid keeps
the design-based point estimate and puts a conjugate prior on the residual variance — two
pseudo-observations at the variance a Poisson count would have at the detector's count
(`countcal/approaches/hybrid.py`). It is negligible when the sample variance is
informative and decisive when it is not. Combined with stratifying the labels on g this is
the best procedure found.

## Results (13 datasets, 100 repeats per cell; `results/leaderboard.md`)

| procedure | keeps coverage everywhere? | width vs SRS + PPI | RMSE vs SRS + PPI |
|---|---|---|---|
| stratified on g (√ alloc.) + regression + variance prior | yes (min 0.84) | 0.96 | 0.74 |
| SRS + PPI++ jackknife + variance prior | yes (min 0.82) | 1.06 | 0.89 |
| stratified on g + calibrated Bayes (`cb_ps`) | yes (min 0.83) | 1.51 | 0.87 |
| PPS + calibrated Bayes with sites | yes, 9 site datasets (min 0.89) | 1.67 | 1.01 |
| SRS + PPI (reference) | no (min 0.54) | 1.00 | 1.00 |
| PPS + DISCount-CV | no (min 0.74) | 0.89 | 0.90 |
| SRS + generative, cut | no (min 0.59) | 0.82 | 0.87 |
| PPS + generative, joint | no (min 0.00) | 0.76 | 1.59 |

Reading across the families:

- **Pure model-based** methods are efficient when right and badly wrong when not, and the
  two ways they go wrong are specific and reproducible: labeling driven by g (bias that does
  not shrink with n) and learning from the unlabeled images' g (confidently wrong).
- **Calibrated Bayes** (flexible in the design variable, sites as random effects) largely
  fixes both: it keeps coverage on every real dataset under every design, and on all 13
  under stratified labeling (under SRS and PPS it is still biased upward on the simulated
  zero-inflated population, +7% to +34% for n ≥ 25, where the curve's shape near g = 0 is
  badly wrong). Its point
  estimates are as good as the best design-based ones (RMSE 0.87× PPI under stratified
  labeling), but its intervals are 1.5–2× too wide (coverage ~0.98).
- **Less-parametric Bayes** did not beat calibrated Bayes: the stratified Pólya posterior is
  efficient but undercovers (it can only replay residuals it has seen), and the monotone NB
  curve, which passed on the first nine datasets, fails on WCS (coverage 0.76 at n = 100; it
  has no site effects and the detector's bias differs by site) and is wildly biased at small
  n on Channel Islands (+145% at n = 25, hidden by very wide intervals).
- **Model-assisted** estimators are design-unbiased by construction; with small-sample
  (jackknife / t) intervals they nearly pass, and the variance prior takes them over the line.
- **The design matters as much as the estimator**: stratifying on g with √ allocation
  improves every estimator; PPS ∝ g + 0.5 often hurts.

## Recommended next steps

1. **Use now**: stratify the labeling on g (√ allocation, ≤ 8 strata of contiguous g),
   estimate with the stratified regression (PPI++/GREG) estimator, and use the variance-prior
   interval. It needs no MCMC, keeps coverage on all 13 datasets, and has ~26% lower RMSE than
   random labeling + PPI at the same labeling cost.
2. **Make calibrated Bayes as narrow as it is accurate.** The Bayesian models' estimates are
   competitive; their posteriors are over-dispersed. Candidates: tighter / data-calibrated
   priors on the per-g deviation scale and the NB dispersion, a smoothness (random-walk /
   P-spline) prior across g instead of exchangeable deviations, and checking calibration with
   held-out datasets rather than tuning on the benchmark. This is the step that would make
   the fully Bayesian route — which also gives per-site estimates and posterior predictive
   checks for free — the recommended one.
3. **Use the detector's whole output, not one thresholded count.** On Serengeti at threshold
   0.5, the herds that break every method are exactly the images where the count at a low
   confidence threshold far exceeds the count at the high one (residual SD 3.9 vs 0.4).
   Stratifying on that gap cut the design SE by 20–30% in a pilot and lifted coverage. Add
   auxiliary detector outputs to `CountData` (counts at several thresholds for the camera-trap
   datasets, the second tracker for CFC) and use them as design variables and as covariates.
4. **Per-site (small-area) estimates.** For multi-site surveys the per-site means are often
   the real target; evaluate per-site coverage, where partial pooling across sites should
   beat any design-based method.
5. **Paper framing.** Calibrated Bayes for ML-assisted counting: (i) two reproducible
   failure modes of model-based inference (detector-driven labeling; learning from unlabeled
   predictions), (ii) a failure mode of design-based inference that gets *worse* as detectors
   improve (zero residual variance), and (iii) a design + estimator + variance-prior procedure
   that keeps coverage on every dataset at lower RMSE, with the fully Bayesian version as the
   open problem.
