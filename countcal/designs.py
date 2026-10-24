"""Labeling designs: which images get hand-counted.

A Sample records the sequence of draws (indices into the population, possibly with
repeats) and the per-draw selection probabilities. Model-based estimators use only
the distinct labeled images; design-based estimators use the draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Added to g before normalising so images with g = 0 can still be drawn.
PPS_FLOOR = 0.5


@dataclass
class Sample:
    design: str
    draws: np.ndarray            # indices, length n (repeats allowed for "pps")
    q: np.ndarray | None = None  # per-draw selection probabilities over the population ("pps")

    @property
    def n(self) -> int:
        return len(self.draws)

    @property
    def labeled(self) -> np.ndarray:
        return np.unique(self.draws)


def srs(N: int, n: int, rng: np.random.Generator) -> Sample:
    """Simple random sample without replacement."""
    return Sample("srs", rng.choice(N, size=n, replace=False))


def pps(g: np.ndarray, n: int, rng: np.random.Generator, floor: float = PPS_FLOOR) -> Sample:
    """Draws with replacement, probability proportional to g + floor (DISCount-style)."""
    w = g.astype(float) + floor
    q = w / w.sum()
    return Sample("pps", rng.choice(len(g), size=n, replace=True, p=q), q=q)


DESIGNS = {"srs": srs, "pps": pps}


def draw(design: str, g: np.ndarray, n: int, rng: np.random.Generator) -> Sample:
    if design == "srs":
        return srs(len(g), n, rng)
    if design == "pps":
        return pps(g, n, rng)
    raise ValueError(f"unknown design {design!r}")
