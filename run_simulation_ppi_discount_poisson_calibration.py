#!/usr/bin/env python3
"""
Run simulation over annotation effort levels (n_labeled = 10, 20, ...).
For each level, 20 repeats with two labeling strategies:
  (1) Importance sampling: labeled subset sampled with q ∝ g.
  (2) Random: labeled subset sampled uniformly at random.
Poisson calibration, PPI, and DISCount share the same labeled data per run.

Usage:
  python run_simulation_ppi_discount_poisson_calibration.py [--output RESULTS.json]
"""

import argparse
import json
import numpy as np
import arviz as az
from cmdstanpy import CmdStanModel
from ppi_py import ppi_mean_pointestimate, ppi_mean_ci


def load_data(path: str = "2025-11-19_discount_f_g.json"):
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
    model_single_parameter: CmdStanModel,
    epsilon: float,
    n_boot: int,
    rng: np.random.Generator,
    labeling_strategy: str,
):
    """Run Poisson calibration, PPI, and DISCount (or sample mean) for one labeled/unlabeled split.
    labeling_strategy: "importance_sampling" (q ∝ g) or "random" (uniform). DISCount uses
    importance weights only under importance_sampling; under random it uses the sample mean.
    """
    idx_unlabeled = np.setdiff1d(np.arange(len(f)), idx_labeled)
    f_labeled = f[idx_labeled]
    g_labeled = g[idx_labeled]
    f_unlabeled = f[idx_unlabeled]
    g_unlabeled = g[idx_unlabeled]

    N = len(f)
    # Small constant so q = (g + ...)/sum(...) is defined when g=0 (importance weights).
    g_floor_for_q = 1e-6
    q = (g + g_floor_for_q) / (g + g_floor_for_q).sum()
    q_labeled = q[idx_labeled]

    # --- Poisson calibration (epsilon used in Stan: log(predicted_counts + epsilon) to avoid log(0)) ---
    data_stan = {
        "N_labeled": n_labeled,
        "N_unlabeled": n_unlabeled,
        "predicted_counts_labeled": g_labeled,
        "true_counts_labeled": f_labeled,
        "predicted_counts_unlabeled": g_unlabeled,
        "epsilon": epsilon,
    }
    fit = model.sample(data=data_stan, seed=int(rng.integers(1, 2**31)), show_progress=False)
    mean_rays = fit.stan_variable("mean_rays_per_image")
    point_poisson = float(mean_rays.mean())
    ci_poisson_lo, ci_poisson_hi = map(float, az.hdi(mean_rays, hdi_prob=0.9))

    # --- Poisson single-parameter (alpha only) ---
    fit_single = model_single_parameter.sample(
        data=data_stan, seed=int(rng.integers(1, 2**31)), show_progress=False
    )
    mean_rays_single = fit_single.stan_variable("mean_rays_per_image")
    point_poisson_single = float(mean_rays_single.mean())
    ci_poisson_single_lo, ci_poisson_single_hi = map(
        float, az.hdi(mean_rays_single, hdi_prob=0.9)
    )

    # --- PPI ---
    mean_ppi = ppi_mean_pointestimate(
        Y=f_labeled.astype(np.float64),
        Yhat=g_labeled,
        Yhat_unlabeled=g_unlabeled,
    )
    ci_ppi = ppi_mean_ci(
        Y=f_labeled.astype(np.float64),
        Yhat=g_labeled,
        Yhat_unlabeled=g_unlabeled,
        alpha=0.1,
    )
    point_ppi = float(np.atleast_1d(mean_ppi)[0])
    ci_ppi_lo = float(np.atleast_1d(ci_ppi[0])[0])
    ci_ppi_hi = float(np.atleast_1d(ci_ppi[1])[0])

    # --- DISCount (importance sampling) or sample mean (random labeling) ---
    if labeling_strategy == "importance_sampling":
        F_hat_discount = (1 / n_labeled) * (f_labeled / q_labeled).sum()
        mean_discount = F_hat_discount / N
        boot_means = []
        for _ in range(n_boot):
            idx_b = rng.integers(0, n_labeled, size=n_labeled)
            F_b = (1 / n_labeled) * (f_labeled[idx_b] / q_labeled[idx_b]).sum()
            boot_means.append(F_b / N)
    else:
        # Random labeling: unbiased estimator is the sample mean (no importance weights)
        mean_discount = (1 / n_labeled) * f_labeled.sum()
        boot_means = []
        for _ in range(n_boot):
            idx_b = rng.integers(0, n_labeled, size=n_labeled)
            boot_means.append((1 / n_labeled) * f_labeled[idx_b].sum())
    point_discount = float(mean_discount)
    ci_discount_lo, ci_discount_hi = map(float, az.hdi(np.array(boot_means), hdi_prob=0.9))

    return {
        "poisson_calibration": {
            "point_estimate": point_poisson,
            "ci_90_lo": ci_poisson_lo,
            "ci_90_hi": ci_poisson_hi,
        },
        "poisson_single_parameter": {
            "point_estimate": point_poisson_single,
            "ci_90_lo": ci_poisson_single_lo,
            "ci_90_hi": ci_poisson_single_hi,
        },
        "ppi": {
            "point_estimate": point_ppi,
            "ci_90_lo": ci_ppi_lo,
            "ci_90_hi": ci_ppi_hi,
        },
        "discount": {
            "point_estimate": point_discount,
            "ci_90_lo": ci_discount_lo,
            "ci_90_hi": ci_discount_hi,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run PPI/DISCount/Poisson calibration simulation.")
    parser.add_argument(
        "--output",
        default="simulation_ppi_discount_poisson_results.json",
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
    parser.add_argument("--data", default="2025-11-19_discount_f_g.json", help="Path to f/g JSON data")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap samples for DISCount CI")
    args = parser.parse_args()

    f, g = load_data(args.data)
    N = len(f)
    # Passed to Stan: log(predicted_counts + epsilon) to avoid log(0).
    epsilon = 1e-6
    # For importance weights q ∝ g: small constant so q is defined when g=0.
    g_floor_for_q = 1e-6
    q = (g + g_floor_for_q) / (g + g_floor_for_q).sum()

    n_labeled_levels = [10 * (i + 1) for i in range(args.n_labeled_levels)]
    total_runs = args.n_labeled_levels * args.n_repeats * 2  # IS + random

    print(f"Data: N={N}")
    print(f"n_labeled levels: {n_labeled_levels}")
    print(f"Repeats per level: {args.n_repeats} (total {total_runs} runs = 2 strategies × levels × repeats)")
    print("Labeling strategies: importance sampling (q ∝ g) and random (uniform)")
    print("Running simulation...")

    model = CmdStanModel(stan_file="poisson_count_calibration.stan")
    model_single_parameter = CmdStanModel(stan_file="poisson_single_parameter.stan")
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
            # Importance sampling: q ∝ g
            seed_labeling_is = args.seed + n_labeled * 1000 + repeat
            idx_labeled_is = rng_is.choice(N, size=n_labeled, replace=False, p=q)
            rec_is = run_one_split(
                f, g, idx_labeled_is, n_labeled, n_unlabeled, model, model_single_parameter,
                epsilon, args.n_boot, rng_is, labeling_strategy="importance_sampling",
            )
            rec_is["seed_labeling"] = seed_labeling_is
            by_labeling["importance_sampling"][key].append(rec_is)
            # Random: uniform over indices
            seed_labeling_rand = args.seed + n_labeled * 2000 + repeat
            idx_labeled_rand = rng_rand.choice(N, size=n_labeled, replace=False)
            rec_rand = run_one_split(
                f, g, idx_labeled_rand, n_labeled, n_unlabeled, model, model_single_parameter,
                epsilon, args.n_boot, rng_rand, labeling_strategy="random",
            )
            rec_rand["seed_labeling"] = seed_labeling_rand
            by_labeling["random"][key].append(rec_rand)
        print(f"  n_labeled={n_labeled}: {args.n_repeats} repeats (IS + random) done.")

    out = {
        "N": N,
        "n_labeled_levels": n_labeled_levels,
        "n_repeats_per_level": args.n_repeats,
        "labeling_strategies": ["importance_sampling", "random"],
        "seed_global": args.seed,
        "by_labeling": by_labeling,
    }
    with open(args.output, "w") as fout:
        json.dump(out, fout, indent=2)

    print(f"Results saved to {args.output}")
    # Summary by labeling strategy, method, and n_labeled
    for strategy in ["importance_sampling", "random"]:
        print(f"  [{strategy}]")
        for method in ["poisson_calibration", "poisson_single_parameter", "ppi", "discount"]:
            print(f"    {method}:")
            for n_labeled in n_labeled_levels:
                key = str(n_labeled)
                if key not in by_labeling[strategy]:
                    continue
                recs = by_labeling[strategy][key]
                pts = [r[method]["point_estimate"] for r in recs]
                print(f"      n={n_labeled}: mean={np.mean(pts):.3f}, std={np.std(pts):.3f}")


if __name__ == "__main__":
    main()
