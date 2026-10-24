"""Estimators of the finite-population mean (and, for model-based methods, the max) of the
true counts f over all N images, given hand counts on a labeled sample and detector
counts g on every image.

Design-based estimators use the labeling design for inference:
  sample_mean  SRS mean of labeled f
  ppi          difference estimator  mean(g_all) + mean(f - g)_lab              (PPI)
  ppi_pp       regression estimator  lam*mean(g_all) + mean(f - lam*g)_lab      (PPI++ / GREG)
  hh           Hansen-Hurwitz under PPS                                        (DISCount)
  hh_cv        Hansen-Hurwitz on residuals f - g, i.e. control variate g       (DISCount-CV / PPI under PPS)
Intervals are normal-theory 90% intervals with the finite-population correction under SRS.

Model-based estimators (Stan) predict the unlabeled true counts:
  poisson      Poisson calibration GLM
  negbin       negative binomial calibration GLM
  gen_joint    generative detection model, full joint likelihood (w = 1)
  gen_cut      generative detection model, conditional likelihood only (w = 0)
Intervals are central 90% posterior intervals.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .designs import Sample

STAN_DIR = Path(__file__).resolve().parent / "stan"
Z90 = norm.ppf(0.95)
CI_LEVEL = 0.90


def _interval(point: float, se: float) -> dict:
    return {"point": float(point), "lo": float(point - Z90 * se), "hi": float(point + Z90 * se)}


# --- design-based ------------------------------------------------------------------------


def sample_mean(f, g, s: Sample) -> dict:
    y = f[s.draws]
    n, N = len(y), len(f)
    fpc = 1 - n / N
    return _interval(y.mean(), np.sqrt(fpc * y.var(ddof=1) / n))


def ppi(f, g, s: Sample) -> dict:
    r = f[s.draws] - g[s.draws]
    n, N = len(r), len(f)
    fpc = 1 - n / N
    return _interval(g.mean() + r.mean(), np.sqrt(fpc * r.var(ddof=1) / n))


def ppi_pp(f, g, s: Sample) -> dict:
    y, x = f[s.draws].astype(float), g[s.draws].astype(float)
    n, N = len(y), len(f)
    vx = x.var(ddof=1)
    lam = np.cov(y, x)[0, 1] / vx if vx > 0 else 0.0
    r = y - lam * x
    fpc = 1 - n / N
    return _interval(lam * g.mean() + r.mean(), np.sqrt(fpc * r.var(ddof=1) / n))


def hh(f, g, s: Sample) -> dict:
    N = len(f)
    z = f[s.draws] / (N * s.q[s.draws])
    return _interval(z.mean(), z.std(ddof=1) / np.sqrt(len(z)))


def hh_cv(f, g, s: Sample) -> dict:
    N = len(f)
    z = (f[s.draws] - g[s.draws]) / (N * s.q[s.draws])
    return _interval(g.mean() + z.mean(), z.std(ddof=1) / np.sqrt(len(z)))


DESIGN_BASED = {
    "srs": {"sample_mean": sample_mean, "ppi": ppi, "ppi_pp": ppi_pp},
    "pps": {"hh": hh, "hh_cv": hh_cv},
}


# --- model-based (Stan) ------------------------------------------------------------------

MODEL_BASED = {
    "poisson": ("calib_poisson", {"g_offset": 1.0}),
    "negbin": ("calib_negbin", {"g_offset": 1.0}),
    "gen_joint": ("generative_detection", {"w_marginal": 1.0}),
    "gen_cut": ("generative_detection", {"w_marginal": 0.0}),
}


@lru_cache(maxsize=None)
def stan_model(name: str):
    from cmdstanpy import CmdStanModel

    return CmdStanModel(stan_file=str(STAN_DIR / f"{name}.stan"))


def stan_data(f, g, s: Sample) -> dict:
    lab = s.labeled
    mask = np.ones(len(f), dtype=bool)
    mask[lab] = False
    g_unique, g_mult = np.unique(g[mask], return_counts=True)
    return {
        "N_lab": len(lab),
        "f_lab": f[lab].astype(int).tolist(),
        "g_lab": g[lab].astype(int).tolist(),
        "U": len(g_unique),
        "g_unique": g_unique.astype(int).tolist(),
        "g_mult": g_mult.astype(int).tolist(),
    }


def _posterior_interval(draws: np.ndarray) -> dict:
    a = (1 - CI_LEVEL) / 2
    lo, hi = np.quantile(draws, [a, 1 - a])
    return {"point": float(draws.mean()), "lo": float(lo), "hi": float(hi)}


def fit_model(method: str, f, g, s: Sample, seed: int, chains: int = 4, iter_warmup: int = 500,
              iter_sampling: int = 500, adapt_delta: float | None = None) -> dict:
    """Fit one model-based estimator. Returns {"mean": interval, "max": interval, "diag": {...}}."""
    model_name, extra = MODEL_BASED[method]
    return run_stan(model_name, stan_data(f, g, s) | extra, seed, chains, iter_warmup, iter_sampling, adapt_delta)


def run_stan(model_name: str, data: dict, seed: int, chains: int = 4, iter_warmup: int = 500,
             iter_sampling: int = 500, adapt_delta: float | None = None) -> dict:
    """Sample a model whose generated quantities include mean_f and max_f."""
    model = stan_model(model_name)
    out_dir = tempfile.mkdtemp(prefix="countcal_")
    try:
        fit = model.sample(
            data=data, seed=seed, chains=chains, parallel_chains=1, iter_warmup=iter_warmup,
            iter_sampling=iter_sampling, adapt_delta=adapt_delta, show_progress=False, show_console=False,
            output_dir=out_dir,
        )
        mean_draws = fit.stan_variable("mean_f")
        max_draws = fit.stan_variable("max_f")
        diag = fit.method_variables()
        n_div = int(np.sum(diag["divergent__"]))
        summ = fit.summary()
        rhat_col = next(c for c in summ.columns if c.lower().replace("_", "") == "rhat")
        max_rhat = float(summ[rhat_col].drop(index=["lp__"], errors="ignore").max())
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    return {
        "mean": _posterior_interval(mean_draws),
        "max": _posterior_interval(max_draws),
        "diag": {"n_divergent": n_div, "max_rhat": max_rhat},
    }


# --- registration ------------------------------------------------------------------------

def _design_based(fn):
    def run(d, s, seed, stan_kwargs):
        return {"mean": fn(d.f, d.g, s)}
    return run


def _model_based(name):
    def run(d, s, seed, stan_kwargs):
        return fit_model(name, d.f, d.g, s, seed=seed, **(stan_kwargs or {}))
    return run


def _register_all():
    from .methods import Method, register

    labels = {"sample_mean": "labels only", "hh": "labels only", "ppi": "difference (PPI)",
              "hh_cv": "difference (PPI)", "ppi_pp": "regression (PPI++)"}
    for design, fns in DESIGN_BASED.items():
        for name, fn in fns.items():
            register(Method(name, _design_based(fn), (design,), labels[name], "design-based"))
    model_labels = {"poisson": "Poisson GLM", "negbin": "NegBin GLM", "gen_joint": "generative, joint",
                    "gen_cut": "generative, cut"}
    for name in MODEL_BASED:
        register(Method(name, _model_based(name), None, model_labels[name], "model-based"))


_register_all()
