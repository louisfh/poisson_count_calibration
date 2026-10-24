#!/usr/bin/env python3
"""Run every estimator on every dataset x design x labeling budget x repeat.

Results are appended to a CSV as tasks finish, so an interrupted run can be resumed by
re-running the same command (finished tasks are skipped).

  python scripts/run_benchmark.py                                  # everything
  python scripts/run_benchmark.py --datasets fish rays --repeats 20 --out results/quick.csv
"""

import argparse
import csv
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from countcal import data, estimators  # noqa: E402
from countcal.benchmark import FIELDS, methods_for, run_task  # noqa: E402

STAN_KWARGS = {"chains": 2, "iter_warmup": 400, "iter_sampling": 500}


def _quiet():
    import cmdstanpy  # noqa: F401  (import first so its handlers exist)

    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)


def _worker(args):
    _quiet()
    dataset, design, n, rep, methods = args
    try:
        return args, run_task(dataset, design, n, rep, methods=methods, stan_kwargs=STAN_KWARGS), None
    except Exception as e:  # keep going; report at the end
        return args, [], repr(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=None, help="default: all (see countcal.data.available_datasets)")
    ap.add_argument("--designs", nargs="+", default=["srs", "pps"])
    ap.add_argument("--n", nargs="+", type=int, default=[10, 25, 50, 100])
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--methods", nargs="+", default=None, help="default: all applicable")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/benchmark.csv")
    args = ap.parse_args()

    _quiet()
    datasets = args.datasets or data.available_datasets()
    for f in sorted(estimators.STAN_DIR.glob("*.stan")):
        estimators.stan_model(f.stem)  # compile once before forking

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()  # (dataset, design, n, rep, method) already in the output
    if out.exists():
        with open(out) as fp:
            for r in csv.DictReader(fp):
                done.add((r["dataset"], r["design"], int(r["n"]), int(r["rep"]), r["method"]))
    tasks = []
    for ds in datasets:
        N = data.load(ds).N
        for des in args.designs:
            wanted = [m for m in methods_for(ds, des) if not args.methods or m in args.methods]
            for n in args.n:
                for rep in range(args.repeats):
                    todo = [m for m in wanted if (ds, des, n, rep, m) not in done]
                    if todo and n < N:
                        tasks.append((ds, des, n, rep, todo))
    print(f"{len(tasks)} tasks to run with {args.workers} workers -> {out}", flush=True)

    new_file = not out.exists()
    errors = []
    t0 = time.time()
    with open(out, "a", newline="") as fp, mp.Pool(args.workers) as pool:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, (task, rows, err) in enumerate(pool.imap_unordered(_worker, tasks), 1):
            if err:
                errors.append((task, err))
            w.writerows(rows)
            fp.flush()
            if i % 50 == 0 or i == len(tasks):
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(tasks)} tasks, {rate:.2f}/s, ~{(len(tasks) - i) / rate / 60:.0f} min left", flush=True)
    if errors:
        print(f"{len(errors)} tasks failed; first few:")
        for task, err in errors[:10]:
            print("  ", task[:4], err)


if __name__ == "__main__":
    main()
