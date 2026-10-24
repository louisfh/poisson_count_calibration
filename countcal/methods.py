"""Registry of estimators.

Every estimator is a Method: a function that takes the population (CountData), the
labeled sample and a seed, and returns at least {"mean": {"point", "lo", "hi"}}; it may
also return "max" (another interval) and "diag" ({"n_divergent", "max_rhat"}). Only the
labeled images' true counts may be used: f[s.labeled] (or f[s.draws]).

Methods are registered by countcal.estimators (the original set) and by the modules in
countcal/approaches/ (one module per family of approaches), so new approaches can be added
without touching the benchmark code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Method:
    name: str
    fn: Callable  # fn(d: CountData, s: Sample, seed: int, stan_kwargs: dict) -> dict
    designs: tuple[str, ...] | None  # labeling designs it applies to; None = any design
    label: str  # short display name
    family: str  # "design-based", "model-based", "calibrated Bayes", "model-assisted", ...
    needs_sites: bool = False  # skip datasets without a site column (or with one site)


REGISTRY: dict[str, Method] = {}


def register(method: Method) -> Method:
    if method.name in REGISTRY:
        raise ValueError(f"method {method.name!r} registered twice")
    REGISTRY[method.name] = method
    return method


def all_methods() -> dict[str, Method]:
    from . import approaches, estimators  # noqa: F401  (import for registration side effects)

    return REGISTRY


def applicable(design: str, has_sites: bool) -> list[Method]:
    return [m for m in all_methods().values()
            if (m.designs is None or design in m.designs) and (has_sites or not m.needs_sites)]
