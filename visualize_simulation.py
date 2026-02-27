#!/usr/bin/env python3
"""
Visualize simulation results: MSE, bias, 90% CI width, and coverage vs n_labeled for each method.
Produces two figures:
  (1) Total fish (mean): MSE, CI width, coverage, bias — scaled by N for total count.
  (2) Max count: MSE, CI width, coverage, bias — for methods that estimate max (Poisson, PPI, generative).

Usage:
  python visualize_simulation.py [--results ...] [--data ...] [--output plots/simulation_visualization_mean.png] [--output-max plots/simulation_visualization_max.png]
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


def load_true_max(data_path: str):
    with open(data_path) as f:
        raw = json.load(f)
    f_arr = np.array(raw["f"], dtype=np.float64)
    return int(f_arr.max())


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


def compute_max_metrics(runs: list, method: str, true_max: float):
    """Compute max estimation metrics. Returns None if method has no max estimate (e.g. DISCount)."""
    if "max" not in runs[0].get(method, {}):
        return None
    max_data = [r[method]["max"] for r in runs]
    if any(m["point_estimate"] is None for m in max_data):
        return None
    pts = np.array([m["point_estimate"] for m in max_data])
    lo = np.array([m["ci_90_lo"] for m in max_data])
    hi = np.array([m["ci_90_hi"] for m in max_data])
    mse = np.mean((pts - true_max) ** 2)
    bias = np.mean(pts) - true_max
    mean_width = np.mean(hi - lo)
    coverage = np.mean((lo <= true_max) & (true_max <= hi))
    return {"mse": mse, "bias": bias, "mean_ci_width": mean_width, "coverage": coverage, "mean_point": np.mean(pts)}


def main():
    parser = argparse.ArgumentParser(description="Visualize simulation: MSE, bias, CI width, coverage vs n_labeled.")
    parser.add_argument(
        "--results",
        default="model_outputs/27_02_26_simulation_ppi_discount_poisson_results.json",
        help="Path to simulation results JSON",
    )
    parser.add_argument(
        "--data",
        default="data/2025-11-19_discount_f_g.json",
        help="Path to f/g data JSON (for true mean)",
    )
    parser.add_argument(
        "--output",
        default="plots/simulation_visualization_mean.png",
        help="Output path for total fish (mean) figure",
    )
    parser.add_argument(
        "--output-max",
        default="plots/simulation_visualization_max.png",
        help="Output path for max count figure",
    )
    args = parser.parse_args()

    results = load_results(args.results)
    true_mean = load_true_mean(args.data)
    true_max = load_true_max(args.data)
    N = results["N"]  # number of samples (e.g. 658) — scale to total fish count

    # Support both legacy (by_n_labeled) and new (by_labeling) result formats
    if "by_labeling" in results:
        strategies = results["labeling_strategies"]  # ["importance_sampling", "random"]
        by_labeling = results["by_labeling"]
        n_labeled_levels = sorted(int(k) for k in by_labeling[strategies[0]].keys())
    else:
        strategies = ["importance_sampling"]
        by_labeling = {"importance_sampling": results["by_n_labeled"]}
        n_labeled_levels = sorted(int(k) for k in results["by_n_labeled"].keys())

    all_methods = ["poisson_calibration", "ppi", "discount", "generative_model"]
    method_colors = {"poisson_calibration": "C0", "ppi": "C1", "discount": "C2", "generative_model": "C3"}
    method_labels_map = {
        "importance_sampling": ["Poisson calibration", "PPI", "DISCount", "Generative model"],
        "random": ["Poisson calibration", "PPI", "Sample mean", "Generative model"],
    }
    strategy_titles = {"importance_sampling": "Importance sampling (q ∝ g)", "random": "Random labeling"}
    strategy_label_col = {"importance_sampling": "Importance sampling", "random": "Random labeling"}

    # Detect which methods exist in results (support legacy results with fewer methods)
    sample_run = next(iter(by_labeling[strategies[0]][str(n_labeled_levels[0])]))
    methods = [m for m in all_methods if m in sample_run]
    colors = [method_colors[m] for m in methods]

    x = np.array(n_labeled_levels)
    n_rows = len(strategies)
    methods_with_max = [m for m in methods if compute_max_metrics(by_labeling[strategies[0]][str(n_labeled_levels[0])], m, true_max) is not None]

    # ========== Figure 1: Total fish (mean) ==========
    fig_mean, axes_mean = plt.subplots(n_rows, 5, figsize=(18, 4 * n_rows), sharex="col", sharey="col")
    if n_rows == 1:
        axes_mean = axes_mean[np.newaxis, :]

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
                mse_by_n[m].append(met["mse"] * (N ** 2))
                bias_by_n[m].append(met["bias"] * N)
                ci_width_by_n[m].append(met["mean_ci_width"] * N)
                coverage_by_n[m].append(met["coverage"])

        ax = axes_mean[row, 0]
        ax.set_axis_off()
        ax.text(0.5, 0.5, strategy_label_col.get(strategy, strategy), transform=ax.transAxes,
                fontsize=12, fontweight="bold", ha="center", va="center")

        ax = axes_mean[row, 1]
        for m, lab, c in zip(methods, labels, colors):
            ax.plot(x, mse_by_n[m], "o-", label=lab, color=c, alpha=0.7)
        ax.set_xlabel("n_labeled")
        ax.set_ylabel("MSE (total count)")
        ax.set_title("MSE of total fish estimate")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes_mean[row, 2]
        for m, lab, c in zip(methods, labels, colors):
            ax.plot(x, ci_width_by_n[m], "o-", label=lab, color=c, alpha=0.7)
        ax.set_xlabel("n_labeled")
        ax.set_ylabel("90% CI width (total count)")
        ax.set_title("90% CI width (total fish)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes_mean[row, 3]
        for m, lab, c in zip(methods, labels, colors):
            ax.plot(x, coverage_by_n[m], "o-", label=lab, color=c, alpha=0.7)
        ax.axhline(0.9, color="black", ls="--", lw=1.5, label="Nominal 90%")
        ax.set_xlabel("n_labeled")
        ax.set_ylabel("Coverage")
        ax.set_title("90% CI coverage")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes_mean[row, 4]
        for m, lab, c in zip(methods, labels, colors):
            ax.plot(x, bias_by_n[m], "o-", label=lab, color=c, alpha=0.7)
        ax.axhline(0, color="black", ls="--", lw=1, label="Unbiased")
        ax.set_xlabel("n_labeled")
        ax.set_ylabel("Bias (total count)")
        ax.set_title("Bias of total fish estimate")
        ax.legend()
        ax.grid(True, alpha=0.3)

    true_total = true_mean * N
    fig_mean.suptitle(f"Total fish (mean): N={N}, true total = {true_total:.0f} (mean = {true_mean:.3f})", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {args.output}")

    # ========== Figure 2: Max count ==========
    fig_max, axes_max = plt.subplots(n_rows, 5, figsize=(18, 4 * n_rows), sharex="col", sharey="col")
    if n_rows == 1:
        axes_max = axes_max[np.newaxis, :]

    for row, strategy in enumerate(strategies):
        by_n_labeled = by_labeling[strategy]
        max_labels = [method_labels_map[strategy][all_methods.index(m)] for m in methods_with_max]
        max_colors = [method_colors[m] for m in methods_with_max]
        mse_max_by_n = {m: [] for m in methods_with_max}
        bias_max_by_n = {m: [] for m in methods_with_max}
        ci_width_max_by_n = {m: [] for m in methods_with_max}
        coverage_max_by_n = {m: [] for m in methods_with_max}

        for n in n_labeled_levels:
            runs = by_n_labeled[str(n)]
            for m in methods_with_max:
                met = compute_max_metrics(runs, m, true_max)
                if met is not None:
                    mse_max_by_n[m].append(met["mse"])
                    bias_max_by_n[m].append(met["bias"])
                    ci_width_max_by_n[m].append(met["mean_ci_width"])
                    coverage_max_by_n[m].append(met["coverage"])

        ax = axes_max[row, 0]
        ax.set_axis_off()
        ax.text(0.5, 0.5, strategy_label_col.get(strategy, strategy), transform=ax.transAxes,
                fontsize=12, fontweight="bold", ha="center", va="center")

        ax = axes_max[row, 1]
        for m, lab, c in zip(methods_with_max, max_labels, max_colors):
            ax.plot(x, mse_max_by_n[m], "o-", label=lab, color=c, alpha=0.7)
        ax.set_xlabel("n_labeled")
        ax.set_ylabel("MSE (max count)")
        ax.set_title("MSE of max count estimate")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes_max[row, 2]
        for m, lab, c in zip(methods_with_max, max_labels, max_colors):
            ax.plot(x, ci_width_max_by_n[m], "o-", label=lab, color=c, alpha=0.7)
        ax.set_xlabel("n_labeled")
        ax.set_ylabel("90% CI width (max count)")
        ax.set_title("90% CI width (max count)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes_max[row, 3]
        for m, lab, c in zip(methods_with_max, max_labels, max_colors):
            ax.plot(x, coverage_max_by_n[m], "o-", label=lab, color=c, alpha=0.7)
        ax.axhline(0.9, color="black", ls="--", lw=1.5, label="Nominal 90%")
        ax.set_xlabel("n_labeled")
        ax.set_ylabel("Coverage")
        ax.set_title("90% CI coverage (max)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes_max[row, 4]
        for m, lab, c in zip(methods_with_max, max_labels, max_colors):
            ax.plot(x, bias_max_by_n[m], "o-", label=lab, color=c, alpha=0.7)
        ax.axhline(0, color="black", ls="--", lw=1, label="Unbiased")
        ax.set_xlabel("n_labeled")
        ax.set_ylabel("Bias (max count)")
        ax.set_title("Bias of max count estimate")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig_max.suptitle(f"Max count: N={N}, true max = {true_max}", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.output_max, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {args.output_max}")
    print(f"True mean: {true_mean:.3f}, true total fish: {true_total:.0f}, true max: {true_max}")
    for strategy in strategies:
        by_n_labeled = by_labeling[strategy]
        labels = [method_labels_map[strategy][all_methods.index(m)] for m in methods]
        print(f"  [{strategy_titles.get(strategy, strategy)}]")
        for m, lab in zip(methods, labels):
            print(f"    {lab}:")
            for n in n_labeled_levels:
                runs = by_n_labeled[str(n)]
                met = compute_metrics(runs, m, true_mean)
                line = f"      n={n}: MSE(total)={met['mse'] * (N**2):.1f}, bias(total)={met['bias'] * N:.1f}, CI width(total)={met['mean_ci_width'] * N:.1f}, coverage={met['coverage']:.2%}"
                max_met = compute_max_metrics(runs, m, true_max)
                if max_met is not None:
                    line += f" | max: pt={max_met['mean_point']:.1f}, cov={max_met['coverage']:.2%}"
                print(line)


if __name__ == "__main__":
    main()
