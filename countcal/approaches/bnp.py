"""Less-parametric Bayesian models of f given g.

Every design in countcal.designs selects images using g only, so any model that conditions
on g flexibly enough is valid under all of them. Two families:

Pólya posterior on g-strata (Ghosh & Meeden 1997; Bayesian post-stratification with no
parametric model for f). Distinct g values are merged into adjacent strata with at least
K_MIN labeled images each. Within a stratum, the unlabeled images' true counts are drawn by
a Pólya urn seeded with the labeled images. A linear working curve m(g) = a + b g (b >= 0)
is subtracted first and added back, so merged strata do not treat images with different g
as exchangeable; the urn itself makes no distributional assumption. Under PPS the urn is
weighted by the design weights (Dong, Elliott & Raghunathan 2014), because a merged stratum
spans g values with different selection probabilities.
  polya_ps       Dirichlet(1) urn (the standard Pólya posterior)
  polya_ps_cal   Dirichlet((n_h - 2) / n_h) urn: the prior weight that makes the posterior
                 variance of a stratum mean equal s^2 / n_h, the usual design-based
                 variance (choosing the prior for frequentist calibration, in the
                 calibrated-Bayes spirit)

Monotone nonparametric curve (Stan, bnp_mono.stan): E[f | g] nondecreasing in g, with a
partially pooled slope between each pair of adjacent g values on the log-log scale and NB
noise. Reduces to the NB calibration GLM when the slopes are pooled completely.
  mono_nb

Tried and dropped (no gain in tuning on the benchmark samples): a Dirichlet-process urn with
a negative-binomial base measure (it narrowed intervals instead of adding a tail), and a
per-g dispersion in the monotone model (identical to mono_nb on heavy-tailed Serengeti).
"""

from __future__ import annotations

import numpy as np

from ..estimators import CI_LEVEL, run_stan
from ..methods import Method, register
from .calibrated import G_OFFSET

K_MIN = 10
N_DRAWS = 4000


# --- Pólya posterior on g-strata ---------------------------------------------------------


def g_strata(g_levels: np.ndarray, n_lab_per_level: np.ndarray, k_min: int) -> np.ndarray:
    """Merge adjacent (sorted) g levels into strata with >= k_min labeled images each.
    Returns the stratum index of each level. A short final stratum joins the previous one."""
    strata = np.zeros(len(g_levels), dtype=int)
    h, acc = 0, 0
    for i, k in enumerate(n_lab_per_level):
        strata[i] = h
        acc += k
        if acc >= k_min and i < len(g_levels) - 1:
            h, acc = h + 1, 0
    if acc < k_min and h > 0:
        strata[strata == h] = h - 1
    return strata


def linear_curve(g, f, w):
    """Weighted least squares f ~ a + b g with b >= 0. Linear (not log-linear) so that
    extrapolating to unlabeled g values beyond the labeled range cannot explode."""
    gw = np.average(g, weights=w)
    fw = np.average(f, weights=w)
    vg = np.average((g - gw) ** 2, weights=w)
    b = max(np.average((g - gw) * (f - fw), weights=w) / vg, 0.0) if vg > 0 else 0.0
    a = fw - b * gw
    return lambda x: a + b * x


def polya_draws(d, s, seed, alpha_fn, k_min: int = K_MIN, n_draws: int = N_DRAWS) -> np.ndarray:
    """Posterior draws of the population mean of f."""
    rng = np.random.default_rng(seed)
    lab, mult = np.unique(s.draws, return_counts=True)
    unl = np.setdiff1d(np.arange(d.N), lab)
    w_img = 1 / s.q[lab] if s.q is not None else np.ones(len(lab))  # design weight per image
    m = linear_curve(d.g[lab].astype(float), d.f[lab].astype(float), mult * w_img)

    levels = np.unique(d.g)
    n_per = np.array([(d.g[lab] == v).sum() for v in levels])
    stratum_of_level = dict(zip(levels.tolist(), g_strata(levels, n_per, k_min).tolist()))
    st_lab = np.array([stratum_of_level[v] for v in d.g[lab].tolist()])
    st_unl = np.array([stratum_of_level[v] for v in d.g[unl].tolist()])

    total = np.full(n_draws, float(d.f[lab].sum()) + m(d.g[unl].astype(float)).sum())
    resid = d.f[lab] - m(d.g[lab].astype(float))
    for h in np.unique(st_unl):
        in_h = st_lab == h
        r = resid[in_h]
        if len(r) == 0:  # only possible if there are no labels at all
            continue
        wh = w_img[in_h] / w_img[in_h].mean()
        p = rng.dirichlet(alpha_fn(len(r)) * wh, size=n_draws)
        counts = rng.multinomial(int((st_unl == h).sum()), p)
        total += counts @ r
    return total / d.N


def _polya(alpha_fn):
    def run(d, s, seed, stan_kwargs):
        x = polya_draws(d, s, seed, alpha_fn)
        lo, hi = np.quantile(x, [(1 - CI_LEVEL) / 2, (1 + CI_LEVEL) / 2])
        return {"mean": {"point": float(x.mean()), "lo": float(lo), "hi": float(hi)}}
    return run


def _alpha_standard(n_h: int) -> float:
    return 1.0


def _alpha_calibrated(n_h: int) -> float:
    # Var(sum p_i y_i) under Dirichlet(a) is s_pop^2 / (n a + 1); a = (n - 2) / n gives s^2 / n.
    return max((n_h - 2) / n_h, 0.25)


# --- monotone curve ----------------------------------------------------------------------


def _mono_data(d, s) -> dict:
    levels, lev_all = np.unique(d.g, return_inverse=True)
    lab = s.labeled
    mask = np.ones(d.N, dtype=bool)
    mask[lab] = False
    cells, mult = np.unique(lev_all[mask], return_counts=True)
    return {
        "N_lab": len(lab),
        "f_lab": d.f[lab].astype(int).tolist(),
        "L": len(levels),
        "x_lev": np.log(levels + G_OFFSET).tolist(),
        "lev_lab": (lev_all[lab] + 1).tolist(),
        "C": len(cells),
        "lev_unl": (cells + 1).tolist(),
        "mult_unl": mult.tolist(),
    }


def _mono(d, s, seed, stan_kwargs):
    # adapt_delta 0.95: the default 0.8 leaves many divergent transitions in this geometry
    return run_stan("bnp_mono", _mono_data(d, s), seed, adapt_delta=0.95, **(stan_kwargs or {}))


register(Method("polya_ps", _polya(_alpha_standard), None, "Pólya post-strat", "Bayesian nonparametric"))
register(Method("polya_ps_cal", _polya(_alpha_calibrated), None, "Pólya post-strat (calibrated)",
                "Bayesian nonparametric"))
register(Method("mono_nb", _mono, None, "monotone NB curve", "Bayesian nonparametric"))
