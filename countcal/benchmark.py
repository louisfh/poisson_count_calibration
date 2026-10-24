"""One benchmark task = one dataset, one labeling design, one labeling budget n, one repeat.

All estimators that apply to the design are run on the same labeled sample, and each
returns a row per estimand (population mean of f; population max of f for model-based
methods).
"""

from __future__ import annotations

import zlib

import numpy as np

from . import data as datamod
from . import designs
from . import estimators as est

FIELDS = [
    "dataset", "design", "n", "rep", "n_unique", "method", "estimand", "truth",
    "point", "lo", "hi", "n_divergent", "max_rhat",
]


def task_seed(dataset: str, design: str, n: int, rep: int) -> int:
    return zlib.crc32(f"{dataset}|{design}|{n}|{rep}".encode()) & 0x7FFFFFFF


def run_task(dataset: str, design: str, n: int, rep: int, methods: list[str] | None = None,
             stan_kwargs: dict | None = None) -> list[dict]:
    d = datamod.load(dataset)
    seed = task_seed(dataset, design, n, rep)
    rng = np.random.default_rng(seed)
    s = designs.draw(design, d.g, n, rng)
    truth_mean, truth_max = float(d.f.mean()), float(d.f.max())
    base = {"dataset": dataset, "design": design, "n": n, "rep": rep, "n_unique": len(s.labeled)}

    rows = []
    for name, fn in est.DESIGN_BASED[design].items():
        if methods and name not in methods:
            continue
        r = fn(d.f, d.g, s)
        rows.append(base | {"method": name, "estimand": "mean", "truth": truth_mean, **r,
                            "n_divergent": None, "max_rhat": None})
    for name in est.MODEL_BASED:
        if methods and name not in methods:
            continue
        r = est.fit_model(name, d.f, d.g, s, seed=seed, **(stan_kwargs or {}))
        for estimand, truth in [("mean", truth_mean), ("max", truth_max)]:
            rows.append(base | {"method": name, "estimand": estimand, "truth": truth, **r[estimand], **r["diag"]})
    return rows
