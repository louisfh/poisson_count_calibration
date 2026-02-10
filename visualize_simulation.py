#!/usr/bin/env python3
"""
Visualize simulation results: MSE, bias, 90% CI width, and coverage vs n_labeled for each method.
Produces two plots: one for importance sampling (IS) labeling, one for random labeling.

Usage:
  python visualize_simulation.py [--results model_outputs/simulation_ppi_discount_poisson_results.json] [--data data/...] [--output plots/figure.png]
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


def load_results(path: str):
    with open(path) as f:
        return json.load(f)


def load_true_mean(data_path: str):
    with open(data_path) as f:
        raw = json.load(f)
    f_arr = np.array(raw["f"], dtype=np.float64)
    return float(f_arr.mean())


def compute_metrics(runs: list, method: str, true_mean: float):
    pts = np.array([r[method]["point_estimate"] for r in runs])
    lo = np.array([r[method]["ci_90_lo"] for r in runs])
    hi = np.array([r[method]["ci_90_hi"] for r in runs])
    mse = np.mean((pts - true_mean) ** 2)
    bias = np.mean(pts) - true_mean
    widths = hi - lo
    mean_width = np.mean(widths)
    coverage = np.mean((lo <= true_mean) & (true_mean <= hi))
    return {"mse": mse, "bias": bias, "mean_ci_width": mean_width, "coverage": coverage}


def main():
    parser = argparse.ArgumentParser(description="Visualize simulation: MSE, bias, CI width, coverage vs n_labeled.")
    parser.add_argument(
        "--results",
        default="model_outputs/simulation_ppi_discount_poisson_results.json",
        help="Path to simulation results JSON",
    )
    parser.add_argument(
        "--data",
        default="data/2025-11-19_discount_f_g.json",
        help="Path to f/g data JSON (for true mean)",
    )
    parser.add_argument(
        "--output",
        default="plots/simulation_visualization.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    results = load_results(args.results)
    true_mean = load_true_mean(args.data)

    # Support both legacy (by_n_labeled) and new (by_labeling) result formats
    if "by_labeling" in results:
        strategies = results["labeling_strategies"]  # ["importance_sampling", "random"]
        by_labeling = results["by_labeling"]
        n_labeled_levels = sorted(int(k) for k in by_labeling[strategies[0]].keys())
    else:
        strategies = ["importance_sampling"]
        by_labeling = {"importance_sampling": results["by_n_labeled"]}
        n_labeled_levels = sorted(int(k) for k in results["by_n_labeled"].keys())

    all_methods = ["poisson_calibration", "ppi", "discount"]
    method_colors = {"poisson_calibration": "C0", "ppi": "C1", "discount": "C2"}
    method_labels_map = {
        "importance_sampling": ["Poisson calibration", "PPI", "DISCount"],
        "random": ["Poisson calibration", "PPI", "Sample mean"],
    }
    strategy_titles = {"importance_sampling": "Importance sampling (q ∝ g)", "random": "Random labeling"}
    strategy_label_col = {"importance_sampling": "Importance sampling", "random": "Random labeling"}

    # Detect which methods exist in results (support legacy results with fewer methods)
    sample_run = next(iter(by_labeling[strategies[0]][str(n_labeled_levels[0])]))
    methods = [m for m in all_methods if m in sample_run]
    colors = [method_colors[m] for m in methods]

    x = np.array(n_labeled_levels)
    n_rows = len(strategies)
    fig, axes = plt.subplots(n_rows, 5, figsize=(18, 4 * n_rows), sharex="col", sharey="col")
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, strategy in enumerate(strategies):
        by_n_labeled = by_labeling[strategy]
        labels = [method_labels_map[strategy][all_methods.index(m)] for m in methods]
        mse_by_n = {m: [] for m in methods}
        bias_by_n = {m: [] for m in methods}
        ci_width_by_n = {m: [] for m in methods}
        coverage_by_n = {m: [] for m in methods}

        for n in n_labeled_levels:
            runs = by_n_labeled[str(n)]
            for m in methods:
                met = compute_metrics(runs, m, true_mean)
                mse_by_n[m].append(met["mse"])
                bias_by_n[m].append(met["bias"])
                ci_width_by_n[m].append(met["mean_ci_width"])
                coverage_by_n[m].append(met["coverage"])

        # --- Left column: approach name only ---
        ax = axes[row, 0]
        ax.set_axis_off()
        ax.text(0.5, 0.5, strategy_label_col.get(strategy, strategy), transform=ax.transAxes,
                fontsize=12, fontweight="bold", ha="center", va="center")

        # --- MSE ---
        ax = axes[row, 1]
        for m, lab, c in zip(methods, labels, colors):
            ax.plot(x, mse_by_n[m], "o-", label=lab, color=c)
        ax.set_xlabel("n_labeled")
        ax.set_ylabel("MSE")
        ax.set_title("Mean squared error")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Bias ---
        ax = axes[row, 2]
        for m, lab, c in zip(methods, labels, colors):
            ax.plot(x, bias_by_n[m], "o-", label=lab, color=c)
        ax.axhline(0, color="black", ls="--", lw=1, label="Unbiased")
        ax.set_xlabel("n_labeled")
        ax.set_ylabel("Bias")
        ax.set_title("Bias")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- 90% CI width ---
        ax = axes[row, 3]
        for m, lab, c in zip(methods, labels, colors):
            ax.plot(x, ci_width_by_n[m], "o-", label=lab, color=c)
        ax.set_xlabel("n_labeled")
        ax.set_ylabel("90% CI width (mean)")
        ax.set_title("90% CI width")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Coverage ---
        ax = axes[row, 4]
        for m, lab, c in zip(methods, labels, colors):
            ax.plot(x, coverage_by_n[m], "o-", label=lab, color=c)
        ax.axhline(0.9, color="black", ls="--", lw=1.5, label="Nominal 90%")
        ax.set_xlabel("n_labeled")
        ax.set_ylabel("Coverage")
        ax.set_title("90% CI coverage")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"N={results['N']}, true mean = {true_mean:.3f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {args.output}")
    print(f"True mean: {true_mean:.3f}")
    for strategy in strategies:
        by_n_labeled = by_labeling[strategy]
        labels = [method_labels_map[strategy][all_methods.index(m)] for m in methods]
        print(f"  [{strategy_titles.get(strategy, strategy)}]")
        for m, lab in zip(methods, labels):
            print(f"    {lab}:")
            for n in n_labeled_levels:
                idx = n_labeled_levels.index(n)
                runs = by_n_labeled[str(n)]
                met = compute_metrics(runs, m, true_mean)
                print(f"      n={n}: MSE={met['mse']:.4f}, bias={met['bias']:.4f}, CI width={met['mean_ci_width']:.3f}, coverage={met['coverage']:.2%}")


if __name__ == "__main__":
    main()
