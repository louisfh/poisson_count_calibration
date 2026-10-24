#!/usr/bin/env python3
"""Summarise results/benchmark.csv: bias, RMSE, 90% interval coverage and width.

Writes
  results/summary.csv               one row per dataset x design x n x method x estimand
  results/summary.md                the headline tables (also the table view of the figures)
  results/figures/coverage.png      coverage heatmap, datasets x methods, both designs
  results/figures/<dataset>.png     relative RMSE / coverage / width vs labeling budget
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

# Design-based methods that play the same role under the two designs share a label/colour.
METHOD_LABEL = {
    "sample_mean": "labels only", "hh": "labels only",
    "ppi": "difference (PPI)", "hh_cv": "difference (PPI)",
    "ppi_pp": "regression (PPI++)",
    "poisson": "Poisson GLM", "negbin": "NegBin GLM",
    "gen_joint": "generative, joint", "gen_cut": "generative, cut",
}
ORDER = ["labels only", "difference (PPI)", "regression (PPI++)", "Poisson GLM", "NegBin GLM",
         "generative, joint", "generative, cut"]
# Reference categorical palette (validated, light mode), fixed order.
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
MARKER = dict(zip(ORDER, ["o", "s", "D", "^", "v", "P", "X"]))
DESIGN_LABEL = {"srs": "simple random sample", "pps": "PPS on detector count"}
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
    out["label"] = out.method.map(METHOD_LABEL)
    return out


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_dataset(s: pd.DataFrame, name: str, path: Path):
    s = s[(s.dataset == name) & (s.estimand == "mean")]
    metrics = [("rel_rmse", "relative RMSE"), ("coverage", "90% interval coverage"), ("rel_width", "relative interval width")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, facecolor=SURFACE)
    for i, design in enumerate(["srs", "pps"]):
        sd = s[s.design == design]
        for j, (col, title) in enumerate(metrics):
            ax = axes[i, j]
            _style(ax)
            for lab in ORDER:
                sl = sd[sd.label == lab].sort_values("n")
                if sl.empty:
                    continue
                ax.plot(sl.n, sl[col], color=COLOR[lab], marker=MARKER[lab], markersize=5, linewidth=2,
                        label=lab, markeredgecolor=SURFACE, markeredgewidth=1)
            if col == "coverage":
                ax.axhline(NOMINAL, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
                lo = sd[col].min() if len(sd) else 0.5
                ax.set_ylim(max(0, min(0.5, lo - 0.05)), 1.01)
            if col != "coverage":
                ax.set_ylim(bottom=0)
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticks(sorted(s.n.unique()))
            ax.set_xticklabels([str(v) for v in sorted(s.n.unique())])
            if i == 0:
                ax.set_title(title, color=INK, fontsize=10, loc="left")
            if j == 0:
                ax.set_ylabel(DESIGN_LABEL[design], color=INK, fontsize=9)
            if i == 1:
                ax.set_xlabel("hand-labeled images (n)", color=INK2, fontsize=8)
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    order = [labels.index(l) for l in ORDER if l in labels]
    fig.legend([handles[k] for k in order], [labels[k] for k in order], loc="lower center", ncol=len(order),
               frameon=False, fontsize=8, labelcolor=INK)
    d = data.load(name)
    fig.suptitle(f"{name}  (N = {d.N}, mean true count {d.f.mean():.2f}, mean detector count {d.g.mean():.2f})",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_coverage(s: pd.DataFrame, n: int, path: Path):
    s = s[(s.estimand == "mean") & (s.n == n)]
    datasets = sorted(s.dataset.unique(), key=lambda x: (x.startswith("synth"), x))
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    norm = TwoSlopeNorm(vmin=0.5, vcenter=NOMINAL, vmax=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.45 * len(datasets) + 2), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [6, 7]})
    for ax, design in zip(axes, ["pps", "srs"]):
        sd = s[s.design == design]
        labs = [l for l in ORDER if l in set(sd.label)]
        M = np.full((len(datasets), len(labs)), np.nan)
        for a, ds in enumerate(datasets):
            for b, lab in enumerate(labs):
                v = sd[(sd.dataset == ds) & (sd.label == lab)].coverage
                if len(v):
                    M[a, b] = v.iloc[0]
        ax.imshow(np.clip(M, 0.5, 1.0), cmap=cmap, norm=norm, aspect="auto")
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                if not np.isnan(M[a, b]):
                    ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=8, color=INK)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets if design == "pps" else [], fontsize=8, color=INK)
        ax.set_title(DESIGN_LABEL[design], color=INK, fontsize=10, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(labs)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(datasets)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="both", length=0)
    fig.suptitle(f"Coverage of nominal 90% intervals for the population mean, n = {n} hand-labeled images "
                 f"(red = undercovers, blue = conservative)", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def table(s: pd.DataFrame, value: str, estimand: str, n: int, fmt: str) -> str:
    t = s[(s.estimand == estimand) & (s.n == n)].pivot_table(index="dataset", columns=["design", "label"], values=value)
    cols = [(d, l) for d in ["srs", "pps"] for l in ORDER if (d, l) in t.columns]
    t = t[cols]
    head = "| dataset | " + " | ".join(f"{d.upper()}: {l}" for d, l in cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    rows = [f"| {ds} | " + " | ".join("" if pd.isna(v) else format(v, fmt) for v in t.loc[ds]) + " |" for ds in t.index]
    return "\n".join([head, sep, *rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/benchmark.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--headline-n", type=int, default=25)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    s = summarise(df)
    out = Path(args.outdir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    s.to_csv(out / "summary.csv", index=False)

    n = args.headline_n
    md = [f"# Benchmark summary\n\nGenerated by `scripts/summarize_benchmark.py` from `{args.results}` "
          f"({df.groupby(['dataset', 'design', 'n']).rep.nunique().min()}-"
          f"{df.groupby(['dataset', 'design', 'n']).rep.nunique().max()} repeats per cell). "
          "SRS = simple random sample of images to hand-label; PPS = draws with replacement, "
          "probability proportional to detector count + 0.5.\n"]
    for nn in sorted(s.n.unique()):
        md += [f"## n = {nn}: coverage of nominal 90% intervals (population mean)\n",
               table(s, "coverage", "mean", nn, ".2f"), ""]
    md += [f"## n = {n}: relative RMSE (RMSE / true mean)\n", table(s, "rel_rmse", "mean", n, ".3f"), "",
           f"## n = {n}: relative bias\n", table(s, "rel_bias", "mean", n, "+.3f"), "",
           f"## n = {n}: relative 90% interval width\n", table(s, "rel_width", "mean", n, ".3f"), "",
           f"## n = {n}: population max, model-based methods (coverage of 90% interval)\n",
           table(s, "coverage", "max", n, ".2f"), "",
           f"## n = {n}: population max, relative bias\n", table(s, "rel_bias", "max", n, "+.3f"), ""]
    diag = s[s.max_rhat.notna()]
    md += ["## MCMC diagnostics\n",
           f"Worst R-hat over all model fits: {diag.max_rhat.max():.3f}. "
           f"Cells where any fit had a divergent transition: {(diag.frac_divergent > 0).sum()} of {len(diag)}; "
           f"worst cell had divergences in {diag.frac_divergent.max():.0%} of fits.\n"]
    (out / "summary.md").write_text("\n".join(md))

    plot_coverage(s, n, out / "figures" / "coverage.png")
    for ds in s.dataset.unique():
        plot_dataset(s, ds, out / "figures" / f"{ds}.png")
    print(f"wrote {out / 'summary.md'}, {out / 'summary.csv'} and {len(s.dataset.unique()) + 1} figures")


if __name__ == "__main__":
    main()
