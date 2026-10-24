# Count calibration: from detector counts to population counts

An ML detector gives a predicted count `g` for every image in a survey; hand-counting
gives the true count `f`, but only for a few images. This repo compares ways to combine
the two to estimate the **population mean (equivalently total) count** and the
**population maximum count**, with honest uncertainty:

- **design-based** estimators that use the randomness of which images were hand-labeled
  (sample mean, PPI difference estimator, PPI++ / regression estimator, DISCount's
  Hansen–Hurwitz estimator and its control-variate version), and
- **model-based** Bayesian estimators that model `f` given `g` (Poisson and negative
  binomial calibration GLMs, and a generative detection model with misses and false
  positives) and predict the unlabeled images.

The benchmark runs every estimator on every dataset under two labeling designs and
several labeling budgets, and reports bias, RMSE, interval coverage and width.

## Layout

```
countcal/                 the package
  data.py                 dataset loaders + synthetic populations
  designs.py              labeling designs (SRS, PPS-with-replacement)
  estimators.py           all estimators
  benchmark.py            one benchmark task
  stan/                   Stan models used by the estimators
scripts/
  prepare_existing.py     data/raw -> data/processed for fish and rays
  prepare_serengeti.py    Snapshot Serengeti + MegaDetector dataset (download + detect)
  run_benchmark.py        run the benchmark (parallel, resumable) -> results/benchmark.csv
  summarize_benchmark.py  results/benchmark.csv -> results/summary.{md,csv}, results/figures/
data/
  raw/                    source data as received / downloaded
  processed/              one CSV per dataset: image_id, site, true_count, pred_count
results/                  benchmark output (summary.md is the place to start)
notebooks/                exploratory notebooks for the two original datasets
tests/                    pytest (fast tests; `-m slow` also fits the Stan models)
archive/                  the pre-refactor scripts, Stan models, notebooks and plots
```

## Setup

```bash
pip install -e ".[dev]"
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"   # once; compiles CmdStan
python scripts/prepare_existing.py                          # builds data/processed/{fish,rays}.csv
python -m pytest -m "not slow"
```

Stan models compile on first use. The Serengeti dataset is already in
`data/processed/`; rebuilding it from images needs `pip install megadetector` and ~1 GB of
downloads (see the script's docstring).

## Running the benchmark

```bash
python scripts/run_benchmark.py                     # all datasets, n = 10 25 50 100, 100 repeats
python scripts/run_benchmark.py --datasets fish --repeats 20 --out results/quick.csv
python scripts/summarize_benchmark.py               # tables + figures
```

A task is one (dataset, design, n, repeat); all estimators see the same labeled sample.
Results are appended as tasks finish, so an interrupted run resumes where it stopped.

## Datasets

| name | source | N | mean f | mean g | notes |
|---|---|---|---|---|---|
| `fish` | `data/raw/fish_discount_f_g_2025-11-19.json` | 658 | 9.14 | 7.44 | detector undercounts; 15 days of clips |
| `rays` | `data/raw/rays_maria_frame_counts.csv` (Maria) | 1763 | 1.14 | 0.72 | video frames, 6 sites / 16 videos; detector misses most rays in 3 videos |
| `serengeti_md_t10/t20/t50` | Snapshot Serengeti boxes (LILA) + MegaDetector v5a | 1845 | 2.62 | see summary | 9 camera sites, season 1; predicted count = animal detections above confidence 0.1 / 0.2 / 0.5 |
| `synth_well_specified` | simulated | 1000 | | | the generative detection model holds exactly |
| `synth_crowding` | simulated | 1000 | | | detection probability falls with crowding |
| `synth_double_count` | simulated | 1000 | | | duplicate detections: detector overcounts |
| `synth_zero_inflated` | simulated | 1000 | | | 60% empty images that attract false positives |

Each dataset is treated as a fixed finite population; the estimand is its mean (or max)
true count over all N images, labeled and unlabeled.

Caveats: MegaDetector's training data includes Snapshot Serengeti, so the detector is
better there than on a new camera network. The Serengeti survey contains only images that
have at least one animal box. Rays site names are lower-cased in `data/processed/rays.csv`
because the raw file mixes `Pescador`/`pescador` and `Rincon`/`rincon`.

## Methods

Labeling designs (`countcal/designs.py`):

- `srs` — simple random sample of n images without replacement.
- `pps` — n draws with replacement with probability ∝ g + 0.5 (DISCount-style importance
  sampling; the 0.5 floor lets images with g = 0 be drawn, which the Hansen–Hurwitz
  estimator needs to stay unbiased when the detector misses everything in an image).

Estimators (`countcal/estimators.py`):

| design | method | what it is |
|---|---|---|
| SRS | `sample_mean` | mean of labeled f (ignores the detector) |
| SRS | `ppi` | mean(g, all images) + mean(f − g, labeled) — PPI / difference estimator |
| SRS | `ppi_pp` | same with g scaled by the fitted slope — PPI++ / GREG regression estimator |
| PPS | `hh` | Hansen–Hurwitz, mean of f / (N q) — DISCount |
| PPS | `hh_cv` | mean(g) + Hansen–Hurwitz on f − g — DISCount with control variate, = PPI under PPS |
| both | `poisson` | f ~ Poisson(exp(α + β log(g + 1))), fit on labeled images |
| both | `negbin` | same with a negative binomial (overdispersion φ) |
| both | `gen_joint` | generative model: f ~ NB(μ, φ), detections ~ Binomial(f, p), false positives ~ Poisson(λ); likelihood = labeled p(f, g) + unlabeled p(g) |
| both | `gen_cut` | same model, likelihood = labeled p(f \| g) only (unlabeled g's do not inform the parameters) |

Design-based intervals are 90% normal-theory intervals (with the finite-population
correction under SRS). Model-based intervals are central 90% posterior intervals of the
finite-population quantity (labeled f known, unlabeled f drawn from the posterior
predictive). Only model-based methods estimate the max.

Two properties worth knowing when reading the results:

- **The Poisson GLM is a GREG estimator in disguise.** With a log link and a free
  intercept, the maximum-likelihood fit satisfies Σ_labeled f = Σ_labeled λ̂, so the
  plug-in estimate of the population mean is mean(λ̂, all images) +
  mean(f − λ̂, labeled): a difference estimator with the GLM as the working model. Under
  SRS it is therefore approximately design-unbiased even when the Poisson model is wrong.
- **Selection on g is ignorable only for the right likelihood.** Under PPS the labeled
  images are chosen using g, so p(f | g) is unaffected but the labeled p(f) is not. The
  joint model is valid only if the unlabeled images' p(g) is included (`gen_joint`), and
  the conditional model (`gen_cut`) drops p(g) everywhere. The old `generative_model.stan`
  option `use_unlabeled_likelihood = 0` kept the labeled p(f, g) and dropped the unlabeled
  p(g), which is biased under importance sampling.

## Changes from the February 2026 code (now in `archive/`)

- **PPI baseline fixed.** The old simulation passed only the *unlabeled* images' detector
  counts as PPI's prediction term. Under importance sampling the unlabeled set is
  depleted of high-g images, so PPI was biased low, increasingly so with n (−0.54 at
  n = 100, coverage 0.30). PPI now uses g for all images.
- **Importance sampling is now with replacement**, matching the Hansen–Hurwitz/DISCount
  estimator. `rng.choice(replace=False, p=q)` does not give inclusion probabilities ∝ q.
- **Generative model posterior predictive fixed.** Missed animals were drawn independently
  of the detected count, which is only right for a Poisson. Under the negative binomial
  they are correlated through the latent rate, so the model now updates λ given the
  detected count before drawing the misses (matters most for the max).
- `efficient_generative_model.stan` (out of sync) and `generative_model.stan` are merged
  into `countcal/stan/generative_detection.stan`, which collapses unlabeled images to
  unique g values and vectorises the likelihood, and has the `w_marginal` switch above.
- Calibration GLMs use log(g + 1) instead of log(g + 1e-6), so images with g = 0 are not
  forced to have a true count near zero.
- False-positive prior changed from Gamma(2, 1) (zero density at 0) to Exponential(0.5).
- The PPI "max via the 0.99999 quantile" is dropped: it is not an estimator of the max.
- Stan executables are no longer committed (they were macOS arm64 binaries).
- `run_simulation_ppi_discount_poisson_calibration.py` and `visualize_simulation.py` are
  replaced by `scripts/run_benchmark.py` and `scripts/summarize_benchmark.py`.

## Results

Full tables: [`results/summary.md`](results/summary.md); per-dataset figures (relative RMSE,
coverage, interval width vs n, both designs): `results/figures/<dataset>.png`; coverage
heatmap at n = 25: `results/figures/coverage.png`. 100 repeats per cell, so a coverage
estimate has a standard error of about 0.03.

1. **Labeling chosen by detector count (PPS) makes model-based estimators biased when the
   model is wrong, and more labels do not fix it.** On `rays`, every model-based estimator
   is unbiased under SRS (bias ≤ 0.4% at n = 100), but under PPS the GLMs are biased by
   −5% to −9% and the generative models by −9% to −17% at every n from 10 to 100;
   generative coverage falls from 0.92–0.98 at n = 10 to 0.22–0.24 at n = 100. The detector
   misses rays in some videos much more than others, and PPS under-samples the frames
   where it saw nothing (46% of frames, 19% of draws), so the fitted p(f | g) is wrong
   exactly where it is extrapolated. `synth_zero_inflated` shows
   the same for the GLMs (+9% under PPS at n = 100 vs +2% under SRS). The design-based
   estimators (`hh`, `hh_cv`) are unbiased under PPS on every dataset. Selection on g is
   ignorable in theory; in practice it turns misspecification into bias.
2. **The unlabeled detector counts make the joint generative model confidently wrong
   when it is misspecified; the cut model is robust.** With duplicate detections
   (`synth_double_count`, SRS) the joint model is biased +14% at n = 25 with coverage 0.07;
   the cut model, which uses the same model but only p(f | g), is unbiased with coverage
   0.88. Same direction on `synth_zero_inflated` (+7.7% vs +1.1%) and under PPS on
   `synth_crowding` and `fish`. When the model is right the joint model is the most
   efficient estimator (smallest RMSE on `synth_well_specified`, and the narrowest
   intervals under PPS).
3. **The Poisson GLM behaves like a GREG estimator under SRS; the NegBin GLM does not.**
   On `synth_zero_inflated` (SRS) the Poisson GLM's bias shrinks from +28% at n = 10 to
   +2% at n = 100 although the model is wrong; the NegBin GLM (non-canonical link) stays
   at +16%. The Poisson GLM's *intervals* are still model-based and undercover when the
   model is wrong (0.66–0.86 on `synth_crowding` and `synth_zero_inflated`); on `fish` and
   `rays` they are conservative (0.94–1.00).
4. **Design-based intervals undercover at small n.** At n = 10, sample-mean and PPI
   intervals cover 0.79–0.91 and PPI++ 0.69–0.85 (nominal 0.90); by n = 25–50 they are
   close to nominal. PPI is *worse* than labels-only on `rays` because the detector is
   badly calibrated there (f − g has more variance than f); PPI++ fixes that.
5. **The population max is only as good as the model's tails.** The generative models
   cover the max when they are well specified (1.00 on `synth_well_specified` and
   `synth_zero_inflated`), but under SRS coverage is near 0 on `rays` (true max 4, predicted 6–8) and
   0.43–0.62 on `fish` for the generative models. No method here is a reliable max estimator on real data.

MCMC: fewer than 4% of fits have R-hat > 1.05. The cut generative model on
`synth_zero_inflated` has a few divergent transitions in most fits (median 3 of 1000
draws), a sign that its posterior is weakly identified without the marginal likelihood.

## Datasets we would like to add

These are public but could not be fetched from the environment the benchmark was built in:

- **MegaDetector results for LILA camera-trap datasets** (lila.science) — would add the
  LILA datasets with bounding boxes (Caltech Camera Traps, Channel Islands, ENA24,
  Missouri, NACTI, WCS, all Snapshot Serengeti seasons) without re-running MegaDetector.
- **Caltech Fish Counting** (Kay et al., ECCV 2022; data.caltech.edu) — sonar fish counts
  per clip with the published tracker's counts, across five rivers.
- **DISCount's bird-roost radar and damaged-building data** (Perez et al., AAAI 2024).
