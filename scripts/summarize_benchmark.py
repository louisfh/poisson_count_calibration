#!/usr/bin/env python3
"""Summarise results/benchmark.csv: bias, RMSE, 90% interval coverage and width.

The full ranking of every design + estimator procedure against the objective is
results/leaderboard.md (scripts/leaderboard.py). This script writes
  results/summary.csv               one row per dataset x design x n x method x estimand
  results/summary.md                headline procedures: coverage and RMSE by dataset
  results/figures/coverage.png      coverage heatmap, datasets x headline procedures, n = 25
  results/figures/<dataset>.png     relative RMSE / coverage / width vs n, headline procedures
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from countcal import data  # noqa: E402

# Headline procedures (design, method, label), one per family; at most 8 so each keeps a
# fixed, colour-blind-safe colour from the reference categorical palette.
HEADLINE = [
    ("srs", "ppi", "SRS + PPI (reference)"),
    ("pps", "hh_cv", "PPS + DISCount-CV"),
    ("srs", "poisson", "SRS + Poisson GLM"),
    ("srs", "gen_cut", "SRS + generative (cut)"),
    ("pps", "cb_ps_site", "PPS + calibrated Bayes (sites)"),
    ("srs", "mono_nb", "SRS + monotone NB"),
    ("srs", "ppi_pp_jack_vp", "SRS + PPI++ jackknife, var. prior"),
    ("strat_sqrt", "st_reg_vp", "stratified + regression, var. prior"),
]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NOMINAL = 0.90


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(err=df.point - df.truth, covered=(df.lo <= df.truth) & (df.truth <= df.hi), width=df.hi - df.lo)
    g = df.groupby(["dataset", "design", "n", "method", "estimand"])
    out = g.agg(truth=("truth", "first"), reps=("rep", "size"), bias=("err", "mean"),
                rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))), coverage=("covered", "mean"),
                width=("width", "mean"), max_rhat=("max_rhat", "max"),
                frac_divergent=("n_divergent", lambda x: float((x > 0).mean()) if x.notna().any() else np.nan))
    out = out.reset_index()
    out["rel_rmse"] = out.rmse / out.truth
    out["rel_bias"] = out.bias / out.truth
    out["rel_width"] = out.width / out.truth
    return out


def headline(s: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for i, (des, meth, lab) in enumerate(HEADLINE):
        p = s[(s.design == des) & (s.method == meth) & (s.estimand == "mean")].copy()
        p["label"], p["order"] = lab, i
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_dataset(h: pd.DataFrame, name: str, path: Path):
    h = h[h.dataset == name]
    metrics = [("rel_rmse", "relative RMSE"), ("coverage", "90% interval coverage"),
               ("rel_width", "relative interval width")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=SURFACE)
    ns = sorted(h.n.unique())
    for ax, (col, title) in zip(axes, metrics):
        _style(ax)
        for i, (_, _, lab) in enumerate(HEADLINE):
            sl = h[h.label == lab].sort_values("n")
            if sl.empty:
                continue
            ax.plot(sl.n, sl[col], color=PALETTE[i], marker=MARKERS[i], markersize=6, linewidth=2, label=lab,
                    markeredgecolor=SURFACE, markeredgewidth=1)
        if col == "coverage":
            ax.axhline(NOMINAL, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
            ax.set_ylim(max(0.0, min(0.5, h[col].min() - 0.05)), 1.01)
        else:
            ax.set_ylim(bottom=0, top=min(h[col].max() * 1.05, 3 * h[col].median()))
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks(ns)
        ax.set_xticklabels([str(v) for v in ns])
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("hand-labeled images (n)", color=INK2, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK)
    d = data.load(name)
    fig.suptitle(f"{name}  (N = {d.N}, mean true count {d.f.mean():.2f}, mean detector count {d.g.mean():.2f})",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_coverage(h: pd.DataFrame, n: int, path: Path):
    h = h[h.n == n]
    datasets = sorted(h.dataset.unique(), key=lambda x: (x.startswith("synth"), x))
    labels = [lab for _, _, lab in HEADLINE]
    M = np.full((len(datasets), len(labels)), np.nan)
    for a, ds in enumerate(datasets):
        for b, lab in enumerate(labels):
            v = h[(h.dataset == ds) & (h.label == lab)].coverage
            if len(v):
                M[a, b] = v.iloc[0]
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    norm = TwoSlopeNorm(vmin=0.5, vcenter=NOMINAL, vmax=1.0)
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(datasets) + 2.6), facecolor=SURFACE)
    ax.imshow(np.clip(M, 0.5, 1.0), cmap=cmap, norm=norm, aspect="auto")
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            if not np.isnan(M[a, b]):
                ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=8, color=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(datasets)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.set_title(f"Coverage of nominal 90% intervals for the population mean, n = {n} "
                 f"(red = undercovers, blue = conservative; blank = not applicable or not run)",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def table(h: pd.DataFrame, value: str, n: int, fmt: str) -> str:
    t = h[h.n == n].pivot_table(index="dataset", columns="label", values=value)
    cols = [lab for _, _, lab in HEADLINE if lab in t.columns]
    t = t[cols]
    lines = ["| dataset | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for ds in sorted(t.index, key=lambda x: (x.startswith("synth"), x)):
        lines.append(f"| {ds} | " + " | ".join("" if pd.isna(v) else format(v, fmt) for v in t.loc[ds]) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/benchmark.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--headline-n", type=int, default=25)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    s = summarise(df)
    out = Path(args.outdir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for old in fig_dir.glob("*.png"):
        old.unlink()
    s.to_csv(out / "summary.csv", index=False)
    h = headline(s)

    n = args.headline_n
    md = ["# Benchmark summary: headline procedures\n",
          "Generated by `scripts/summarize_benchmark.py` from `results/benchmark.csv`. The full ranking of every "
          "design + estimator procedure is in [`leaderboard.md`](leaderboard.md); per-cell numbers for every method "
          "are in `summary.csv`. 100 repeats per cell.\n",
          "Procedures: " + "; ".join(f"`{d}` + `{m}` ({lab})" for d, m, lab in HEADLINE) + ".\n"]
    for nn in sorted(h.n.unique()):
        md += [f"## n = {nn}: coverage of nominal 90% intervals\n", table(h, "coverage", nn, ".2f"), ""]
    md += [f"## n = {n}: relative RMSE (RMSE / true mean)\n", table(h, "rel_rmse", n, ".3f"), "",
           f"## n = {n}: relative bias\n", table(h, "rel_bias", n, "+.3f"), "",
           f"## n = {n}: relative 90% interval width\n", table(h, "rel_width", n, ".3f"), ""]
    diag = s[s.max_rhat.notna()]
    md += ["## MCMC diagnostics (all Stan-based methods)\n",
           f"Worst R-hat over all fits: {diag.max_rhat.max():.3f}. Cells where any fit had a divergent transition: "
           f"{(diag.frac_divergent > 0).sum()} of {len(diag)}.\n"]
    (out / "summary.md").write_text("\n".join(md))

    plot_coverage(h, n, fig_dir / "coverage.png")
    for ds in sorted(h.dataset.unique()):
        plot_dataset(h, ds, fig_dir / f"{ds}.png")
    print(f"wrote {out / 'summary.md'}, {out / 'summary.csv'} and {len(h.dataset.unique()) + 1} figures")


if __name__ == "__main__":
    main()
