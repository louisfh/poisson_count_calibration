#!/usr/bin/env python3
"""
Run simulation over annotation effort levels (n_labeled = 10, 20, ...).
Target: estimate the MAX of all data (both labeled and unlabeled), i.e. max(f).
For each level, 20 repeats with two labeling strategies:
  (1) Importance sampling: labeled subset sampled with q ∝ g.
  (2) Random: labeled subset sampled uniformly at random.

Methods implemented:
  - Poisson calibration: Stan model posterior over max (labeled true counts + posterior-predictive unlabeled).
  - Labeled max: naive baseline max(f_labeled).

PPI, DIScount, and single-parameter Poisson for max are not implemented yet.

Usage:
  python run_simulation_max.py [--output model_outputs/simulation_max_results.json]
  python run_simulation_max.py --stan-output-dir model_outputs/stan_fits_max
"""

import argparse
import json
import os
import numpy as np
import arviz as az
from cmdstanpy import CmdStanModel


def _stan_diagnostics(fit):
    """Extract num_divergent and per-parameter Rhat from a CmdStanMCMC fit."""
    try:
        diag = fit.sampler_diagnostics()
        num_divergent = int(np.sum(diag["divergent__"]))
    except Exception:
        num_divergent = None

    try:
        summary_df = fit.summary()
        rhat_col = "R_hat" if "R_hat" in summary_df.columns else "rhat"
        rhat = summary_df[rhat_col].dropna().astype(float).to_dict()
        rhat = {str(k): float(v) for k, v in rhat.items()}
    except Exception:
        rhat = None

    return {"num_divergent": num_divergent, "rhat": rhat}


def load_data(path: str = "data/2025-11-19_discount_f_g.json"):
    with open(path) as f:
        data_raw = json.load(f)
    f_arr = np.array(data_raw["f"], dtype=np.int32)
    g_arr = np.array(data_raw["g"], dtype=np.float64)
    return f_arr, g_arr


def run_one_split(
    f: np.ndarray,
    g: np.ndarray,
    idx_labeled: np.ndarray,
    n_labeled: int,
    n_unlabeled: int,
    model: CmdStanModel,
    epsilon: float,
    n_boot: int,
    rng: np.random.Generator,
    labeling_strategy: str,
    stan_output_dir: str | None = None,
    repeat: int = 0,
):
    """Run Poisson calibration (max) and labeled_max for one labeled/unlabeled split."""
    idx_unlabeled = np.setdiff1d(np.arange(len(f)), idx_labeled)
    f_labeled = f[idx_labeled]
    g_labeled = g[idx_labeled]
    f_unlabeled = f[idx_unlabeled]
    g_unlabeled = g[idx_unlabeled]

    N = len(f)
    g_floor_for_q = 1e-6
    q = (g + g_floor_for_q) / (g + g_floor_for_q).sum()

    # --- Poisson calibration: posterior over max (labeled + posterior-predictive unlabeled) ---
    data_stan = {
        "N_labeled": n_labeled,
        "N_unlabeled": n_unlabeled,
        "predicted_counts_labeled": g_labeled,
        "true_counts_labeled": f_labeled,
        "predicted_counts_unlabeled": g_unlabeled,
        "epsilon": epsilon,
    }
    out_dir_poisson = None
    if stan_output_dir:
        run_tag = f"n{n_labeled}_rep{repeat}_{labeling_strategy}"
        out_dir_poisson = os.path.join(stan_output_dir, "poisson_count_calibration_max", run_tag)
        os.makedirs(out_dir_poisson, exist_ok=True)

    fit = model.sample(
        data=data_stan,
        seed=int(rng.integers(1, 2**31)),
        show_progress=False,
        output_dir=out_dir_poisson if out_dir_poisson else None,
    )
    max_count_samples = fit.stan_variable("max_count")
    point_poisson = float(np.median(max_count_samples))
    ci_poisson_lo, ci_poisson_hi = map(int, np.percentile(max_count_samples, [5, 95]))
    diag_poisson = _stan_diagnostics(fit)

    # --- Labeled max: naive baseline ---
    point_labeled_max = int(np.max(f_labeled))
    boot_maxs = []
    for _ in range(n_boot):
        idx_b = rng.integers(0, n_labeled, size=n_labeled)
        boot_maxs.append(np.max(f_labeled[idx_b]))
    boot_maxs = np.array(boot_maxs)
    ci_labeled_lo = int(np.percentile(boot_maxs, 5))
    ci_labeled_hi = int(np.percentile(boot_maxs, 95))

    result = {
        "poisson_calibration": {
            "point_estimate": point_poisson,
            "ci_90_lo": ci_poisson_lo,
            "ci_90_hi": ci_poisson_hi,
        },
        "labeled_max": {
            "point_estimate": point_labeled_max,
            "ci_90_lo": ci_labeled_lo,
            "ci_90_hi": ci_labeled_hi,
        },
    }
    result["stan_diagnostics"] = {"poisson_calibration": diag_poisson}
    if out_dir_poisson is not None:
        result["stan_output_dirs"] = {"poisson_calibration": out_dir_poisson}
    return result


def main():
    parser = argparse.ArgumentParser(description="Run max-estimation simulation (Poisson calibration + labeled max).")
    parser.add_argument(
        "--output",
        default="model_outputs/simulation_max_results.json",
        help="Output JSON path for results",
    )
    parser.add_argument(
        "--n-labeled-levels",
        type=int,
        default=10,
        help="Number of annotation effort levels (n=10,20,...,10*levels)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=20,
        help="Repeats per n_labeled level with different random seeds",
    )
    parser.add_argument("--data", default="data/2025-11-19_discount_f_g.json", help="Path to f/g JSON data")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap samples for labeled_max CI")
    parser.add_argument(
        "--stan-output-dir",
        default=None,
        help="Directory to save Stan MCMC CSV outputs per run. Default: do not save.",
    )
    args = parser.parse_args()

    f, g = load_data(args.data)
    N = len(f)
    true_max = int(np.max(f))
    epsilon = 1e-6
    g_floor_for_q = 1e-6
    q = (g + g_floor_for_q) / (g + g_floor_for_q).sum()

    n_labeled_levels = [10 * (i + 1) for i in range(args.n_labeled_levels)]
    total_runs = args.n_labeled_levels * args.n_repeats * 2

    if args.stan_output_dir:
        os.makedirs(args.stan_output_dir, exist_ok=True)
        print(f"Stan outputs will be saved to: {args.stan_output_dir}")

    print(f"Data: N={N}, true_max (max of all f) = {true_max}")
    print(f"n_labeled levels: {n_labeled_levels}")
    print(f"Repeats per level: {args.n_repeats} (total {total_runs} runs)")
    print("Labeling strategies: importance sampling (q ∝ g) and random (uniform)")
    print("Running simulation...")

    model = CmdStanModel(stan_file="stan_models/poisson_count_calibration.stan")
    by_labeling = {"importance_sampling": {}, "random": {}}

    for n_labeled in n_labeled_levels:
        if n_labeled > N:
            print(f"  Skipping n_labeled={n_labeled} (exceeds N={N})")
            continue
        n_unlabeled = N - n_labeled
        key = str(n_labeled)
        by_labeling["importance_sampling"][key] = []
        by_labeling["random"][key] = []

        for repeat in range(args.n_repeats):
            rng_is = np.random.default_rng(args.seed + n_labeled * 1000 + repeat)
            rng_rand = np.random.default_rng(args.seed + n_labeled * 2000 + repeat)

            seed_labeling_is = args.seed + n_labeled * 1000 + repeat
            idx_labeled_is = rng_is.choice(N, size=n_labeled, replace=False, p=q)
            rec_is = run_one_split(
                f, g, idx_labeled_is, n_labeled, n_unlabeled, model,
                epsilon, args.n_boot, rng_is, labeling_strategy="importance_sampling",
                stan_output_dir=args.stan_output_dir, repeat=repeat,
            )
            rec_is["seed_labeling"] = seed_labeling_is
            by_labeling["importance_sampling"][key].append(rec_is)

            seed_labeling_rand = args.seed + n_labeled * 2000 + repeat
            idx_labeled_rand = rng_rand.choice(N, size=n_labeled, replace=False)
            rec_rand = run_one_split(
                f, g, idx_labeled_rand, n_labeled, n_unlabeled, model,
                epsilon, args.n_boot, rng_rand, labeling_strategy="random",
                stan_output_dir=args.stan_output_dir, repeat=repeat,
            )
            rec_rand["seed_labeling"] = seed_labeling_rand
            by_labeling["random"][key].append(rec_rand)

        print(f"  n_labeled={n_labeled}: {args.n_repeats} repeats (IS + random) done.")

    out = {
        "N": N,
        "true_max": true_max,
        "n_labeled_levels": n_labeled_levels,
        "n_repeats_per_level": args.n_repeats,
        "labeling_strategies": ["importance_sampling", "random"],
        "seed_global": args.seed,
        "by_labeling": by_labeling,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fout:
        json.dump(out, fout, indent=2)

    print(f"Results saved to {args.output}")
    for strategy in ["importance_sampling", "random"]:
        print(f"  [{strategy}]")
        for method in ["poisson_calibration", "labeled_max"]:
            print(f"    {method}:")
            for n_labeled in n_labeled_levels:
                key = str(n_labeled)
                if key not in by_labeling[strategy]:
                    continue
                recs = by_labeling[strategy][key]
                pts = [r[method]["point_estimate"] for r in recs]
                print(f"      n={n_labeled}: mean(point)={np.mean(pts):.1f}, std={np.std(pts):.1f}")


if __name__ == "__main__":
    main()
