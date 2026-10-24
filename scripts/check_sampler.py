#!/usr/bin/env python3
"""Re-run one model-based method on one dataset with a stricter sampler (adapt_delta = 0.95)
on the same labeled samples as the benchmark, and compare with the benchmark's results.

Used to check that divergent transitions in a benchmark cell did not distort its results:
  python scripts/check_sampler.py --dataset synth_zero_inflated --method gen_cut
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from countcal import data, designs, estimators  # noqa: E402
from countcal.benchmark import task_seed  # noqa: E402


def _run(args):
    import cmdstanpy  # noqa: F401

    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    dataset, method, design, n, rep, adapt_delta = args
    d = data.load(dataset)
    seed = task_seed(dataset, design, n, rep)
    s = designs.draw(design, d.g, n, np.random.default_rng(seed))
    r = estimators.fit_model(method, d.f, d.g, s, seed=seed, chains=2, iter_warmup=400, iter_sampling=500,
                             adapt_delta=adapt_delta)
    return {"design": design, "n": n, "rep": rep, "truth": float(d.f.mean()), **r["mean"], **r["diag"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--n", nargs="+", type=int, default=[10, 25, 50, 100])
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--adapt-delta", type=float, default=0.95)
    ap.add_argument("--results", default="results/benchmark.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    estimators.stan_model(estimators.MODEL_BASED[args.method][0])
    tasks = [(args.dataset, args.method, des, n, rep, args.adapt_delta)
             for des in ["srs", "pps"] for n in args.n for rep in range(args.repeats)]
    with mp.Pool(os.cpu_count()) as pool:
        new = pd.DataFrame(pool.map(_run, tasks))
    out = Path(args.out or f"results/sampler_check_{args.dataset}_{args.method}.csv")
    new.to_csv(out, index=False)

    old = pd.read_csv(args.results)
    old = old[(old.dataset == args.dataset) & (old.method == args.method) & (old.estimand == "mean")]

    def metrics(df):
        cov = (df.lo <= df.truth) & (df.truth <= df.hi)
        return pd.Series({"rel_bias": ((df.point - df.truth) / df.truth).mean(), "coverage": cov.mean(),
                          "frac_divergent": (df.n_divergent > 0).mean()})

    cmp = pd.concat({"adapt_delta=0.8": old.groupby(["design", "n"]).apply(metrics, include_groups=False),
                     f"adapt_delta={args.adapt_delta}": new.groupby(["design", "n"]).apply(metrics, include_groups=False)},
                    axis=1)
    print(cmp.round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
