"""Design-based estimators with a Bayesian prior on the residual variance.

Design-based intervals (PPI, PPI++, DISCount, model-assisted) estimate the variance of the
residuals f - m(g) from the labeled sample. When the detector is accurate, almost every
residual is exactly 0 (Channel Islands: 96% of images), so at n = 25 the sample variance is
often exactly 0 and the interval collapses to a point: the better the detector, the worse
the coverage. Model-based intervals never collapse because the model's variance has a prior.

Here the point estimate stays design-based, and the variance of the per-draw values z gets a
conjugate prior: K_PRIOR pseudo-observations at the variance a Poisson count would have at
the detector's count, v_i = g_i + 0.5 per image. On the scale of z this is
  SRS:  mean_i v_i;     PPS with replacement:  sum_i v_i / (N^2 q_i)
(which equals mean(g) + 0.5 when q ∝ g + 0.5). The posterior-mean variance
  s2_post = ((n - ddof) s2 + K_PRIOR s2_prior) / (n - ddof + K_PRIOR)
replaces s2, with a t quantile on n - ddof + K_PRIOR degrees of freedom. With K_PRIOR = 2 the
prior is negligible unless the sample variance is near 0. K_PRIOR and v_i were fixed in
advance, not tuned on the benchmark.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from ..designs import Sample
from ..methods import Method, register
from .intervals import LEVEL, Components, comp_hh_cv, comp_ppi, comp_ppi_pp, ma_poisson_comp

K_PRIOR = 2.0
V_FLOOR = 0.5


def prior_variance(d, s) -> float:
    v = d.g.astype(float) + V_FLOOR
    if s.q is None:
        return float(v.mean())
    return float(np.sum(v / (d.N ** 2 * s.q)))


def _interval(point: float, s2: float, n: int, ddof: int, fpc: float, s2_prior: float) -> dict:
    df = n - ddof
    s2_post = (df * s2 + K_PRIOR * s2_prior) / (df + K_PRIOR)
    se = np.sqrt(fpc * s2_post / n)
    q = float(student_t.ppf(0.5 + LEVEL / 2, df + K_PRIOR))
    return {"mean": {"point": float(point), "lo": float(point - q * se), "hi": float(point + q * se)}}


def plugin_vp(comp):
    """Plug-in variance of z with the variance prior."""

    def run(d, s, seed, stan_kwargs):
        c: Components = comp(d, s, seed)
        return _interval(c.point, float(c.z.var(ddof=c.ddof)), c.n, c.ddof, c.fpc, prior_variance(d, s))

    return run


def jackknife_vp(comp):
    """Delete-one jackknife variance (refitting any coefficients), then the variance prior."""

    def run(d, s, seed, stan_kwargs):
        c: Components = comp(d, s, seed)
        n = s.n
        loo = np.array([comp(d, Sample(s.design, np.delete(s.draws, i), q=s.q), seed).point for i in range(n)])
        var_point = (n - 1) / n * ((loo - loo.mean()) ** 2).sum()  # includes the fpc
        s2 = var_point * n / c.fpc  # back to the per-draw scale
        return _interval(c.point, s2, n, c.ddof, c.fpc, prior_variance(d, s))

    return run


FAMILY = "design-based + variance prior"
register(Method("ppi_vp", plugin_vp(comp_ppi), ("srs",), "difference (PPI), variance prior", FAMILY))
register(Method("ppi_pp_jack_vp", jackknife_vp(comp_ppi_pp), ("srs",), "regression (PPI++), jackknife + variance prior",
                FAMILY))
register(Method("hh_cv_vp", plugin_vp(comp_hh_cv), ("pps",), "difference (PPI), variance prior", FAMILY))
register(Method("ma_poisson_jack_vp", jackknife_vp(ma_poisson_comp), ("srs", "pps"),
                "model-assisted (Poisson), jackknife + variance prior", FAMILY))
