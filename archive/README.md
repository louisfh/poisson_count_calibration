# Archive

The code as it stood in February 2026, kept for reference. It is superseded by
`countcal/` and `scripts/` (see the top-level README, "Changes from the February 2026
code", for what changed and why). File paths were updated for the new layout (run the
scripts from the repo root), but nothing here was re-run, and some notebooks refer to Stan
variables that the February models had already renamed.

- `scripts/run_simulation_ppi_discount_poisson_calibration.py`, `scripts/visualize_simulation.py`
  — the old fish simulation; its PPI baseline under importance sampling was biased
  (see top-level README). Last output: `archive/results/27_02_26_simulation_ppi_discount_poisson_results.json`.
- `stan_models/multi_site_count_calibration.stan`,
  `stan_models/negative_binomial_multi_site_count_calibration.stan`,
  `scripts/run_negative_binomial_multi_site_demo.py` and the `demo_multi_site*` /
  `negative_binomial_multi_site*` notebooks — per-site calibration (no pooling across
  sites). A natural next step is a hierarchical version for the rays and Serengeti sites.
- `stan_models/COM_poisson_count_calibration.stan` — Conway–Maxwell–Poisson calibration.
