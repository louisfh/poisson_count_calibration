# Count calibration: from detector counts to population counts

An ML detector gives a predicted count `g` for every image in a survey; hand-counting
gives the true count `f`, but only for the images someone labels. This repo benchmarks
ways to choose which images to label and to combine the two into an estimate of the
**population mean (equivalently total) count** with an honest interval — design-based
(PPI, DISCount), model-based (Bayesian calibration and detection models), and hybrids in
the spirit of Little's *calibrated Bayes*.

**Start with** [`results/leaderboard.md`](results/leaderboard.md) (every procedure ranked
against the objective), [`docs/way_forward.md`](docs/way_forward.md) (the framing and the
recommended next steps) and `results/figures/`.

## Objective

A *procedure* is a labeling design plus an estimator. On each dataset — a fixed finite
population — and each labeling budget n, the procedure is run on 100 independent labeled
samples. `scripts/leaderboard.py` then asks:

- **Does it keep coverage?** Mean coverage of its 90% intervals ≥ 0.88 and no
  (dataset, n) cell below 0.80, over n ∈ {25, 50, 100}. (A coverage estimate from 100
  repeats has a standard error of ~0.03, so this tolerates noise but not systematic
  undercoverage.) n = 10 is reported separately.
- **Among those that do, how narrow are its intervals?** Geometric-mean interval width
  relative to simple random sampling + the PPI difference estimator at the same n (RMSE
  reported alongside).

Requiring coverage on *every* dataset is deliberate: a user does not know in advance
whether their detector over-counts, under-counts, misses herds or is nearly perfect.

## Headline results

Thirteen datasets (9 real, from 6 sources, + 4 simulated stress tests), 49 design +
estimator procedures run at full scale, 248k labeled-sample replicates. Full table: `results/leaderboard.md`.

| procedure | keeps coverage, all 13 datasets | interval width vs SRS + PPI | RMSE vs SRS + PPI | coverage at n = 10 |
|---|---|---|---|---|
| **stratified on g (√ allocation) + regression estimator + variance prior** (`strat_sqrt` + `st_reg_vp`) | **yes** (mean 0.94, min 0.84) | 0.96 | **0.74** | 0.92 |
| SRS + PPI++ with jackknife + variance prior (`ppi_pp_jack_vp`) | yes (0.91, 0.82) | 1.06 | 0.89 | 0.93 |
| stratified on g + calibrated Bayes (`strat_sqrt` + `cb_ps`) | yes (0.98, 0.83) | 1.51 | 0.87 | 0.99 |
| PPS + calibrated Bayes with sites (`cb_ps_site`) | yes on the 9 datasets with sites (0.98, 0.89) | 1.67 | 1.01 | 0.99 |
| SRS + PPI difference estimator (reference) | no (0.85, 0.54) | 1.00 | 1.00 | 0.74 |
| PPS + DISCount with control variate (`hh_cv`) | no (0.86, 0.74) | 0.89 | 0.90 | 0.80 |
| SRS + generative detection model, cut (`gen_cut`) | no (0.82, 0.59) | 0.82 | 0.87 | 0.87 |
| PPS + generative detection model, joint (`gen_joint`) | no (0.52, 0.00) | 0.76 | 1.59 | 0.60 |

What the benchmark established, in order of how surprising it is:

1. **Accurate detectors break design-based intervals.** On Channel Islands, 96% of images
   have f = g exactly, so at n = 25 the labeled residuals are often all zero and PPI's
   interval has zero width (coverage 0.54 at n = 25, 0.31 at n = 10). A weak conjugate prior
   on the residual variance (two pseudo-observations at a Poisson variance g + 0.5;
   `countcal/approaches/hybrid.py`) fixes it (0.92) without widening intervals elsewhere.
2. **Letting the unlabeled detector counts inform a Bayesian model makes it confidently
   wrong when it is misspecified.** The joint generative model covers 0.00–0.33 on Serengeti
   and is biased up to −26%; the same model fit only to p(f | g) ("cut") is unbiased.
3. **Labeling chosen by detector count makes misspecified models biased, and more labels do
   not help.** On rays, model-based bias under PPS stays at −5% to −17% from n = 10 to 100;
   under random labeling it vanishes. Selection on g is ignorable in theory; in practice it
   turns misspecification into bias. DISCount/PPI remain unbiased under PPS.
4. **Stratifying the labels on g beats importance sampling on g.** √-allocation strata on g
   (Neyman allocation if Var(f | g) ∝ g + 1) with the regression estimator have 0.60–0.91×
   the RMSE of random labeling + PPI on every one of the 13 datasets; PPS ∝ g + 0.5 with
   DISCount-CV is worse than random labeling on 5 of 13 (up to 1.45× on rays). A
   pilot-based Neyman allocation adds nothing over the √ heuristic.
5. **Fully Bayesian calibrated models keep coverage everywhere but are over-conservative**
   (coverage ~0.98, intervals 1.5–2× wider than PPI) even though their point estimates
   are competitive (RMSE 0.87× under stratified labeling) — their intervals, not their
   estimates, are the gap to close.
6. **Heavy tails the detector cannot see** (Serengeti herds at threshold 0.5, where 1% of
   images carry most of the residual variance) limit every method; stratification and the
   variance prior bring coverage to 0.88, but no interval rule can see herds that are not in
   the sample. The detector's own confidence profile (counts at several thresholds) *does*
   locate them — the most promising next step (see `docs/way_forward.md`).
7. Earlier findings that still hold: the Poisson GLM is a GREG estimator in disguise
   (design-unbiased under SRS even when wrong; the NegBin GLM is not), and the population
   *max* is only as good as the model's tails, so it was dropped from the objective.

## Datasets

| name | source | N | mean f | mean g | notes |
|---|---|---|---|---|---|
| `fish` | DISCount-style fish clips (`data/raw/fish_discount_f_g_2025-11-19.json`) | 658 | 9.14 | 7.44 | detector undercounts; 15 days |
| `rays` | Maria's video frames (`data/raw/rays_maria_frame_counts.csv`) | 1763 | 1.14 | 0.72 | 6 sites, 16 videos; detector misses most rays in 3 videos |
| `serengeti_md_t10/t20/t50` | Snapshot Serengeti boxes (LILA) + MegaDetector v5a run here | 1845 | 2.66 | 2.89 / 2.57 / 2.16 | 9 sites; detections ≥ 0.1 / 0.2 / 0.5 confidence; herds up to 29 |
| `cfc_baseline`, `cfc_baselinepp` | Caltech Fish Counting (Kay et al. 2022), per-clip sonar counts | 1085 | 5.05 | 4.70 / 4.75 | 5 rivers; two trackers; counts up to 80 (Nushagak) |
| `wcs_md_t20` | WCS Camera Traps boxes + published MegaDetector v5a results | 2657 | 3.86 | 4.58 | 5 high-count sites (peccary herds); detector over-counts at some sites, under-counts at others |
| `cict_md_t20` | Channel Islands Camera Traps boxes + MegaDetector v5a | 2000 | 0.59 | 0.62 | 60 sites, 47% empty; detector nearly exact |
| `synth_well_specified` | simulated | 1000 | 5.95 | 5.27 | generative detection model holds |
| `synth_crowding` | simulated | 1000 | 6.02 | 3.04 | detection falls with crowding |
| `synth_double_count` | simulated | 1000 | 3.87 | 4.61 | duplicate detections |
| `synth_zero_inflated` | simulated | 1000 | 3.40 | 3.39 | 62% empty images with false positives |

Each is built by a script in `scripts/` (`prepare_existing.py`, `prepare_serengeti.py`,
`prepare_cfc.py`, `prepare_lila.py`) from `data/raw/` into `data/processed/<name>.csv`
(`image_id, site, true_count, pred_count`). Caveats: MegaDetector's training data includes
the LILA datasets, so it is better on them than on new cameras; Caltech Camera Traps was
checked and left out (one animal per image, counted correctly); rays site names are
lower-cased (the raw file mixes `Pescador`/`pescador`); NACTI and the full Serengeti
MegaDetector results are kept outside git (over GitHub's 100 MB limit).

## Methods

Every estimator is registered in `countcal/methods.py`; families live in separate modules:

| module | family | methods |
|---|---|---|
| `estimators.py` | design-based | `sample_mean`, `ppi`, `ppi_pp` (SRS); `hh`, `hh_cv` (PPS: DISCount) |
| `estimators.py` | model-based (Stan) | `poisson`, `negbin` (calibration GLMs), `gen_joint`, `gen_cut` (generative detection model with misses and false positives; joint vs conditional likelihood) |
| `approaches/calibrated.py` | calibrated Bayes; pseudo-likelihood; model-assisted | `cb_ps`, `cb_ps_site` (NB curve + partially pooled per-g deviations [+ site effects]); `poisson_w` (design-weighted likelihood); `ma_poisson`, `ma_flex` (GREG with Poisson / flexible cross-fitted working model) |
| `approaches/bnp.py` | less-parametric Bayes | `polya_ps`, `polya_ps_cal` (stratified Pólya posterior), `mono_nb` (monotone NB calibration curve) |
| `approaches/intervals.py` | small-sample intervals | t, jackknife, Hall skew-corrected and bootstrap-t intervals for the design-based and model-assisted estimators |
| `approaches/designs_learned.py` | learned designs | designs `strat_prop`, `strat_sqrt`, `strat_lin`, `neyman2`, `pps_sqrt` with stratified, two-phase and Hansen–Hurwitz estimators |
| `approaches/hybrid.py` | design-based + variance prior | `ppi_vp`, `ppi_pp_jack_vp`, `hh_cv_vp`, `ma_poisson_jack_vp`, `st_*_vp`, `tp_naive_diff_vp` |

Designs (`countcal/designs.py` + `approaches/designs_learned.py`): `srs` (random), `pps`
(with replacement, ∝ g + 0.5, DISCount-style), and the learned designs above. All select on
g only (the two-phase design also on its own phase-1 labels), so they are ignorable for any
model that conditions on g; model-based methods run under every design.

Design-based intervals are normal or t intervals (finite-population corrected under SRS);
model-based intervals are central 90% posterior intervals of the finite-population mean
(labeled counts known, unlabeled counts drawn from the posterior predictive).

## Layout

```
countcal/            package: data.py, designs.py, methods.py (registry), estimators.py,
                     benchmark.py, approaches/ (one module per family), stan/ (Stan models)
scripts/             prepare_*.py (datasets), run_benchmark.py, leaderboard.py,
                     summarize_benchmark.py, check_sampler.py
data/raw/, data/processed/
results/             benchmark.csv (every replicate), leaderboard.md, summary.{md,csv},
                     figures/, pilot_*_leaderboard.md (exploratory pilots)
docs/way_forward.md  the framing and next steps
tests/               pytest (`-m slow` also fits the Stan models)
notebooks/           exploration of the fish and rays data
archive/             the February 2026 code, models, notebooks and plots
```

## Setup and running

```bash
pip install -e ".[dev]"
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"   # once
python -m pytest -m "not slow"

python scripts/run_benchmark.py --datasets fish --methods ppi cb_ps --repeats 20 --out results/quick.csv
python scripts/leaderboard.py --results results/benchmark.csv
python scripts/summarize_benchmark.py
```

A benchmark task is one (dataset, design, n, repeat); every estimator applicable to the
design sees the same labeled sample. `run_benchmark.py` skips (dataset, design, n, repeat,
method) combinations already in `--out` or `--also-done`, so runs resume and new methods
can be added incrementally.

## Notes on the benchmark

- `gen_joint` was not run on `cfc_baseline`, `wcs_md_t20` and `cict_md_t20` (expensive, and
  it had already failed on the other ten datasets with coverage 0.00–0.60).
- Under PPS, n counts draws; a few draws repeat an image, so PPS uses slightly fewer
  distinct labels than n (up to 10% at n = 100), which disadvantages it slightly.
- The variance prior's settings (2 pseudo-observations, Poisson variance g + 0.5) were
  fixed before running the benchmark, not tuned on it. The intervals it gives are slightly
  conservative (mean coverage 0.94), so there is room to tighten them with a principled
  choice validated on held-out datasets.
- MCMC: fewer than 4% of fits have R-hat > 1.05; `scripts/check_sampler.py` re-ran the
  worst-diverging cell with `adapt_delta = 0.95` without changing its results.

## Changes from the February 2026 code (now in `archive/`)

- PPI baseline fixed (it used only the unlabeled images' g as the prediction term, biasing
  it low under importance sampling); importance sampling is with replacement; the
  generative model's posterior predictive draws missed animals conditional on detections;
  generative models merged and vectorised; calibration GLMs use log(g + 1); the PPI "max"
  hack is dropped; compiled Stan binaries are no longer committed.
