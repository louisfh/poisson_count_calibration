#!/usr/bin/env python3
"""
Demo script: simulate 20-site negative-binomial data, stratified labeling,
confusion matrix, fit model, and plot per-site maximum count posteriors.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from cmdstanpy import CmdStanModel


def simulate_data(n_sites=20, n_images_per_site=250, n_labeled_per_site=30, seed=43):
    """Simulate multi-site data with (predicted, true) counts; return labeled subset
    stratified by predicted count (30 per site). Data generated from the calibration
    model: mu = exp(alpha_s + beta_s*log(pred+eps)), true ~ NB(mu, phi_s).
    """
    rng = np.random.default_rng(seed)
    epsilon = 1e-6

    # True (alpha, beta, phi) per site
    alpha_true = rng.normal(0.2, 0.3, n_sites)
    beta_true = rng.uniform(0.4, 0.9, n_sites)
    phi_true = rng.uniform(2.0, 8.0, n_sites)  # inverse overdispersion (phi_s)

    site_ids_full = np.repeat(np.arange(n_sites), n_images_per_site) + 1
    n_total = len(site_ids_full)
    true_counts = np.zeros(n_total, dtype=np.int64)
    predicted_counts = np.zeros(n_total)

    for s in range(n_sites):
        start = s * n_images_per_site
        end = start + n_images_per_site
        # First generate predicted counts (e.g. from a count distribution)
        predicted_counts[start:end] = rng.poisson(3.0, n_images_per_site) + rng.poisson(2.0, n_images_per_site)
        predicted_counts[start:end] = np.clip(predicted_counts[start:end], 0, 100).astype(np.float64)
        for i in range(n_images_per_site):
            idx = start + i
            pred = predicted_counts[idx]
            mu = np.exp(alpha_true[s] + beta_true[s] * np.log(pred + epsilon))
            mu = np.clip(mu, 1e-6, 1e4)
            # NB2: nbinom(n=phi, p=phi/(phi+mu)) gives mean mu, var mu + mu^2/phi
            p_nb = phi_true[s] / (phi_true[s] + mu)
            true_counts[idx] = nbinom.rvs(phi_true[s], p_nb, random_state=rng)

    # Stratified labeling: 30 per site, stratified by predicted count
    labeled_indices = []
    for s in range(n_sites):
        site_inds = np.where(site_ids_full == s + 1)[0]
        pred_s = predicted_counts[site_inds]
        n_take = min(n_labeled_per_site, len(site_inds))
        # Stratify: sort by predicted count, split into n_take bins, sample one per bin
        order = np.argsort(pred_s)
        bins = np.array_split(order, n_take)
        chosen_local = []
        for b in bins:
            if len(b) > 0:
                chosen_local.append(rng.choice(b))
        chosen_global = site_inds[np.array(chosen_local)]
        labeled_indices.extend(chosen_global)
    labeled_indices = np.array(labeled_indices)
    unlabeled_mask = np.ones(n_total, dtype=bool)
    unlabeled_mask[labeled_indices] = False

    pred_labeled = predicted_counts[labeled_indices].astype(np.float64)
    truth_labeled = true_counts[labeled_indices]
    site_id_labeled = site_ids_full[labeled_indices].astype(np.int32)
    pred_unlabeled = predicted_counts[unlabeled_mask].astype(np.float64)
    site_id_unlabeled = site_ids_full[unlabeled_mask].astype(np.int32)
    truth_unlabeled = true_counts[unlabeled_mask]

    n_labeled = len(pred_labeled)
    n_unlabeled = len(pred_unlabeled)

    return {
        "n_sites": n_sites,
        "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled,
        "pred_labeled": pred_labeled,
        "truth_labeled": truth_labeled,
        "site_id_labeled": site_id_labeled,
        "pred_unlabeled": pred_unlabeled,
        "site_id_unlabeled": site_id_unlabeled,
        "truth_unlabeled": truth_unlabeled,
        "site_ids_full": site_ids_full,
        "true_counts_full": true_counts,
        "predicted_counts_full": predicted_counts,
        "epsilon": epsilon,
    }


def plot_confusion_matrix(truth, predicted, ax=None, max_count=None):
    """Plot count confusion matrix: rows = true count, cols = predicted count (binned)."""
    if ax is None:
        fig, ax = plt.subplots()
    if max_count is None:
        max_count = int(max(truth.max(), predicted.max())) + 1
    max_count = min(max_count, 50)  # cap for visibility
    # 2D histogram: (predicted, true) for convention "predicted vs true"
    H, xe, ye = np.histogram2d(
        np.clip(predicted, 0, max_count - 0.1),
        np.clip(truth, 0, max_count - 0.1),
        bins=max_count,
        range=[[0, max_count], [0, max_count]],
    )
    im = ax.pcolormesh(xe, ye, H.T, cmap="Blues", edgecolors="lightgray", linewidths=0.5)
    plt.colorbar(im, ax=ax, label="Count")
    ax.set_xlabel("Predicted count")
    ax.set_ylabel("True count")
    ax.set_title("Count confusion matrix (labeled data)")
    ax.set_aspect("equal")
    return ax


def main():
    print("1. Simulating data (20 sites, 30 labeled per site stratified by predicted count)...")
    data_dict = simulate_data(
        n_sites=20,
        n_images_per_site=250,
        n_labeled_per_site=30,
        seed=43,
    )
    n_sites = data_dict["n_sites"]
    N_labeled = data_dict["n_labeled"]
    N_unlabeled = data_dict["n_unlabeled"]
    print(f"   Sites: {n_sites}. Labeled: {N_labeled}. Unlabeled: {N_unlabeled}.")

    print("2. Plotting count confusion matrix...")
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_confusion_matrix(
        data_dict["truth_labeled"],
        data_dict["pred_labeled"],
        ax=ax,
        max_count=int(np.percentile(data_dict["truth_labeled"], 99)) + 5,
    )
    plt.tight_layout()
    plt.savefig("plots/simulated_data_count_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("3. Fitting negative binomial multi-site model...")
    stan_data = {
        "N_sites": n_sites,
        "N_labeled": N_labeled,
        "N_unlabeled": N_unlabeled,
        "predicted_counts_labeled": data_dict["pred_labeled"],
        "true_counts_labeled": data_dict["truth_labeled"].tolist(),
        "site_id_labeled": data_dict["site_id_labeled"].tolist(),
        "predicted_counts_unlabeled": data_dict["pred_unlabeled"],
        "site_id_unlabeled": data_dict["site_id_unlabeled"].tolist(),
        "epsilon": data_dict["epsilon"],
    }
    model = CmdStanModel(stan_file="stan_models/negative_binomial_multi_site_count_calibration.stan")
    fit = model.sample(data=stan_data, seed=456, show_progress=True)

    print("4. Estimating site-level maximums...")
    max_count_draws = fit.stan_variable("max_count_site")  # (n_draws, N_sites)
    true_max_per_site = np.array([
        data_dict["true_counts_full"][data_dict["site_ids_full"] == s + 1].max()
        for s in range(n_sites)
    ])

    print("   Per-site max (posterior median [90% CI]) vs true max:")
    for s in range(n_sites):
        mc = max_count_draws[:, s]
        lo, hi = np.percentile(mc, [5, 95])
        print(f"   Site {s+1:2d}: posterior max = {int(np.median(mc)):3d} [{int(lo):3d}, {int(hi):3d}]  |  true max = {int(true_max_per_site[s]):3d}")

    print("5. Plotting per-site distribution of predicted maximums with true max...")
    n_col = 5
    n_row = (n_sites + n_col - 1) // n_col
    fig, axes = plt.subplots(n_row, n_col, figsize=(3 * n_col, 2.5 * n_row))
    axes = np.atleast_2d(axes)
    for s in range(n_sites):
        ax = axes.flat[s]
        mc = max_count_draws[:, s]
        lo, hi = np.percentile(mc, [5, 95])
        x_min, x_max = int(lo) - 1, int(hi) + 1
        x_max = max(x_max, int(true_max_per_site[s]) + 1)  # ensure true max visible if outside CI
        x_min = max(0, x_min)
        bins = max(10, min(50, x_max - x_min + 1))
        ax.hist(mc, bins=bins, color="steelblue", alpha=0.8, edgecolor="white", density=True)
        ax.axvline(lo, color="gray", ls="--", lw=1.5, label="90% CI")
        ax.axvline(hi, color="gray", ls="--", lw=1.5)
        ax.axvline(true_max_per_site[s], color="red", lw=2, label=f"True max = {int(true_max_per_site[s])}")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Max count")
        ax.set_ylabel("Density")
        ax.set_title(f"Site {s+1}")
        ax.legend(fontsize=8)
    for j in range(s + 1, axes.size):
        axes.flat[j].set_visible(False)
    plt.suptitle("Posterior distribution of site-level maximum count (red = true max; dashed = 90% CI)")
    plt.tight_layout()
    plt.savefig("plots/site_max_posteriors.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Done. Saved plots/simulated_data_count_confusion_matrix.png and plots/site_max_posteriors.png.")


if __name__ == "__main__":
    main()
