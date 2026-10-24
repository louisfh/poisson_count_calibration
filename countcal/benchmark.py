"""One benchmark task = one dataset, one labeling design, one labeling budget n, one repeat.

All estimators that apply to the design are run on the same labeled sample, and each
returns a row per estimand (population mean of f; population max of f where the method
provides one).
"""

from __future__ import annotations

import zlib

import numpy as np

from . import data as datamod
from . import designs
from .methods import applicable

FIELDS = [
    "dataset", "design", "n", "rep", "n_unique", "method", "estimand", "truth",
    "point", "lo", "hi", "n_divergent", "max_rhat",
]


def task_seed(dataset: str, design: str, n: int, rep: int) -> int:
    return zlib.crc32(f"{dataset}|{design}|{n}|{rep}".encode()) & 0x7FFFFFFF


def has_sites(d) -> bool:
    return d.site is not None and len(np.unique(d.site)) > 1


def methods_for(dataset: str, design: str) -> list[str]:
    return [m.name for m in applicable(design, has_sites(datamod.load(dataset)))]


def run_task(dataset: str, design: str, n: int, rep: int, methods: list[str] | None = None,
             stan_kwargs: dict | None = None) -> list[dict]:
    d = datamod.load(dataset)
    seed = task_seed(dataset, design, n, rep)
    s = designs.draw(design, d.g, n, np.random.default_rng(seed))
    truth = {"mean": float(d.f.mean()), "max": float(d.f.max())}
    base = {"dataset": dataset, "design": design, "n": n, "rep": rep, "n_unique": len(s.labeled)}

    rows = []
    for m in applicable(design, has_sites(d)):
        if methods and m.name not in methods:
            continue
        r = m.fn(d, s, seed, stan_kwargs)
        diag = r.get("diag", {"n_divergent": None, "max_rhat": None})
        for estimand in ("mean", "max"):
            if estimand in r:
                rows.append(base | {"method": m.name, "estimand": estimand, "truth": truth[estimand],
                                    **r[estimand], **diag})
    return rows
