"""Labeling designs: which images get hand-counted.

A Sample records the sequence of draws (indices into the population, possibly with
repeats) and the per-draw selection probabilities. Model-based estimators use only
the distinct labeled images; design-based estimators use the draws.

Every design here selects images using only the detector counts g (and randomness), so
selection is ignorable for any model that conditions on g; model-based methods therefore
apply to every registered design. A design function takes (g, n, rng) and returns a Sample;
register new ones in DESIGNS. Extra design information (strata, inclusion probabilities,
...) goes in Sample.meta for design-based estimators that need it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Added to g before normalising so images with g = 0 can still be drawn.
PPS_FLOOR = 0.5


@dataclass
class Sample:
    design: str
    draws: np.ndarray            # indices, length n (repeats allowed for "pps")
    q: np.ndarray | None = None  # per-draw selection probabilities over the population ("pps")
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.draws)

    @property
    def labeled(self) -> np.ndarray:
        return np.unique(self.draws)


def srs(g: np.ndarray, n: int, rng: np.random.Generator) -> Sample:
    """Simple random sample without replacement."""
    return Sample("srs", rng.choice(len(g), size=n, replace=False))


def pps(g: np.ndarray, n: int, rng: np.random.Generator, floor: float = PPS_FLOOR) -> Sample:
    """Draws with replacement, probability proportional to g + floor (DISCount-style)."""
    w = g.astype(float) + floor
    q = w / w.sum()
    return Sample("pps", rng.choice(len(g), size=n, replace=True, p=q), q=q)


DESIGNS = {"srs": srs, "pps": pps}


def draw(design: str, g: np.ndarray, n: int, rng: np.random.Generator, label=None) -> Sample:
    """label: optional oracle idx -> true counts f[idx], for adaptive designs (functions with
    attribute adaptive = True) that choose later labels using earlier ones."""
    if design not in DESIGNS:
        raise ValueError(f"unknown design {design!r}")
    fn = DESIGNS[design]
    if getattr(fn, "adaptive", False):
        if label is None:
            raise ValueError(f"design {design!r} is adaptive and needs a label oracle")
        return fn(g, n, rng, label)
    return fn(g, n, rng)
