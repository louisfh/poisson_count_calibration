"""Datasets of paired (true count, detector count) per image.

Every dataset is a finite population of N images with a true count f and a
detector (model-predicted) count g for each image, plus an optional site label.
Processed datasets live in data/processed/<name>.csv with columns
    image_id, site, true_count, pred_count
and are built from data/raw by the scripts in scripts/prepare_*.py.
Synthetic populations are generated on the fly (see SYNTHETIC below).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


@dataclass
class CountData:
    name: str
    f: np.ndarray  # true counts, int
    g: np.ndarray  # detector counts, int
    site: np.ndarray | None = None
    description: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.f = np.asarray(self.f, dtype=np.int64)
        self.g = np.asarray(self.g, dtype=np.int64)
        if self.f.shape != self.g.shape:
            raise ValueError("f and g must have the same length")

    @property
    def N(self) -> int:
        return len(self.f)

    def summary(self) -> dict:
        return {
            "dataset": self.name,
            "N": self.N,
            "mean_f": float(self.f.mean()),
            "mean_g": float(self.g.mean()),
            "max_f": int(self.f.max()),
            "max_g": int(self.g.max()),
            "frac_f0": float((self.f == 0).mean()),
            "frac_g0": float((self.g == 0).mean()),
            "corr_fg": float(np.corrcoef(self.f, self.g)[0, 1]),
            "n_sites": 0 if self.site is None else int(len(np.unique(self.site))),
        }


def load_processed(name: str) -> CountData:
    path = PROCESSED_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    site = df["site"].to_numpy() if "site" in df.columns else None
    return CountData(name=name, f=df["true_count"].to_numpy(), g=df["pred_count"].to_numpy(), site=site)


# --- Synthetic populations with controlled misspecification -------------------------------
#
# All share the generative-detection structure (NB true counts, binomial detection, Poisson
# false positives) and then break one assumption. N and seeds are fixed so each is a single
# finite population, like the real datasets.


def _nb2(rng, mu, phi, size):
    lam = rng.gamma(shape=phi, scale=np.asarray(mu) / phi, size=size)
    return rng.poisson(lam)


def synth_well_specified(N=1000, seed=0, mu=6.0, phi=3.0, p=0.8, lambda_fp=0.5):
    rng = np.random.default_rng(seed)
    f = _nb2(rng, mu, phi, N)
    g = rng.binomial(f, p) + rng.poisson(lambda_fp, N)
    return CountData("synth_well_specified", f, g, description="generative-detection model holds exactly")


def synth_crowding(N=1000, seed=1, mu=6.0, phi=1.5, p0=0.95, k=8.0, lambda_fp=0.3):
    """Detection probability falls as the image gets more crowded (occlusion)."""
    rng = np.random.default_rng(seed)
    f = _nb2(rng, mu, phi, N)
    p = p0 / (1.0 + f / k)
    g = rng.binomial(f, p) + rng.poisson(lambda_fp, N)
    return CountData("synth_crowding", f, g, description=f"detection p = {p0}/(1 + f/{k}): undercounts dense images")


def synth_double_count(N=1000, seed=2, mu=4.0, phi=2.0, p=0.85, dup_rate=0.35, lambda_fp=0.2):
    """Each detected animal can produce extra boxes (fragmented / duplicate detections)."""
    rng = np.random.default_rng(seed)
    f = _nb2(rng, mu, phi, N)
    tp = rng.binomial(f, p)
    g = tp + rng.poisson(dup_rate * tp) + rng.poisson(lambda_fp, N)
    return CountData("synth_double_count", f, g, description=f"each detection duplicated at rate {dup_rate}: overcounts")


def synth_zero_inflated(N=1000, seed=3, pi_empty=0.6, mu=8.0, phi=2.0, p=0.8, fp_empty=1.0, fp_occ=0.2):
    """Most images are empty, and empty images attract more false positives (e.g. vegetation)."""
    rng = np.random.default_rng(seed)
    empty = rng.random(N) < pi_empty
    f = np.where(empty, 0, _nb2(rng, mu, phi, N))
    g = rng.binomial(f, p) + rng.poisson(np.where(empty, fp_empty, fp_occ))
    return CountData(
        "synth_zero_inflated", f, g,
        description=f"{pi_empty:.0%} empty images with {fp_empty} false positives each on average",
    )


SYNTHETIC = {
    "synth_well_specified": synth_well_specified,
    "synth_crowding": synth_crowding,
    "synth_double_count": synth_double_count,
    "synth_zero_inflated": synth_zero_inflated,
}


def available_datasets() -> list[str]:
    real = sorted(p.stem for p in PROCESSED_DIR.glob("*.csv"))
    return real + list(SYNTHETIC)


def load(name: str) -> CountData:
    if name in SYNTHETIC:
        return SYNTHETIC[name]()
    return load_processed(name)
