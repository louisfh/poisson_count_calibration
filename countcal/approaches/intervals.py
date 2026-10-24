"""Intervals that keep coverage at small n and with heavy-tailed counts.

Every design-based and model-assisted estimator in this repo has the form

    point = A + mean(z_1, ..., z_n)

where A is computed from the detector counts of all images (and the working model) and
z_i is one value per labeled draw (a residual f - m(g), possibly divided by N q under PPS).
The normal-theory interval point ± z_0.95 · sqrt(fpc · var(z) / n) undercovers when n is
small or when z is skewed — e.g. when the detector undercounts a few large herds, so a few
draws have huge positive residuals. Here the same point estimates get better intervals:

  *_t      Student-t quantiles (n - 1 df; n - 2 and a residual variance with ddof = 2 for
           PPI++, whose coefficient is estimated)
  *_hall   Student-t plus Hall's (1992) monotone cubic transformation, which removes the
           first-order skewness of the studentised mean (uses the sample skewness of z)
  *_boott  bootstrap-t (studentised bootstrap) over the labeled draws, B = 999
  *_jack   Student-t with a delete-one jackknife variance that recomputes the whole
           estimator, so the variability of estimated coefficients is included (PPI++,
           ma_poisson)

The working model of ma_poisson / ma_flex is held fixed in the bootstrap (its estimation
error is second order for a difference estimator).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

from ..designs import Sample
from ..methods import Method, register
from .calibrated import K_FOLDS, PoissonCurve, ShrunkLevels

LEVEL = 0.90
B_BOOT = 999


@dataclass
class Components:
    point: float
    z: np.ndarray   # per-draw values; point - A = mean(z)
    fpc: float      # variance multiplier: 1 - n/N under SRS, 1 under PPS with replacement
    ddof: int = 1   # parameters estimated from z beyond its mean (PPI++: 2)

    @property
    def n(self) -> int:
        return len(self.z)

    @property
    def se(self) -> float:
        return float(np.sqrt(self.fpc * self.z.var(ddof=self.ddof) / self.n))


# --- components of the existing estimators -------------------------------------------------


def comp_ppi(d, s, seed):
    z = (d.f[s.draws] - d.g[s.draws]).astype(float)
    return Components(d.g.mean() + z.mean(), z, 1 - s.n / d.N)


def comp_ppi_pp(d, s, seed):
    y, x = d.f[s.draws].astype(float), d.g[s.draws].astype(float)
    lam = _slope(y, x)
    z = y - lam * x
    return Components(lam * d.g.mean() + z.mean(), z, 1 - s.n / d.N, ddof=2)


def comp_hh(d, s, seed):
    z = d.f[s.draws] / (d.N * s.q[s.draws])
    return Components(z.mean(), z, 1.0)


def comp_hh_cv(d, s, seed):
    z = (d.f[s.draws] - d.g[s.draws]) / (d.N * s.q[s.draws])
    return Components(d.g.mean() + z.mean(), z, 1.0)


def comp_model_assisted(model_cls, cross_fit: bool):
    """Same estimator as calibrated._model_assisted (tests check the points agree)."""

    def comp(d, s, seed):
        draws, f, g, N, n = s.draws, d.f, d.g, d.N, s.n
        w_draw = 1 / s.q[draws] if s.q is not None else np.ones(n)
        if cross_fit:
            rng = np.random.default_rng(seed)
            uniq = np.unique(draws)
            fold_of = dict(zip(uniq.tolist(), rng.permutation(len(uniq)) % K_FOLDS))
            fold = np.array([fold_of[i] for i in draws.tolist()])
            pred_lab, m_all = np.empty(n), np.zeros(N)
            folds = [k for k in range(K_FOLDS) if (fold == k).any()]
            for k in folds:
                model = model_cls().fit(g[draws[fold != k]], f[draws[fold != k]], w_draw[fold != k])
                pred_lab[fold == k] = model.predict(g[draws[fold == k]])
                m_all += model.predict(g) / len(folds)
        else:
            model = model_cls().fit(g[draws], f[draws], w_draw)
            pred_lab, m_all = model.predict(g[draws]), model.predict(g)
        r = f[draws] - pred_lab
        if s.q is None:
            return Components(m_all.mean() + r.mean(), r, 1 - n / N)
        z = r / (N * s.q[draws])
        return Components(m_all.mean() + z.mean(), z, 1.0)

    return comp


def _slope(y, x):
    vx = x.var(ddof=1)
    return float(np.cov(y, x)[0, 1] / vx) if vx > 0 else 0.0


# --- interval constructions ----------------------------------------------------------------


def _out(lo, point, hi):
    return {"mean": {"point": float(point), "lo": float(lo), "hi": float(hi)}}


def _tq(c: Components) -> float:
    return float(student_t.ppf(0.5 + LEVEL / 2, max(c.n - c.ddof, 1)))


def interval_t(c: Components):
    q = _tq(c)
    return _out(c.point - q * c.se, c.point, c.point + q * c.se)


def skewness(z: np.ndarray) -> float:
    m2 = z.var()
    return float(((z - z.mean()) ** 3).mean() / m2 ** 1.5) if m2 > 0 else 0.0


def hall_inverse(c: np.ndarray | float, a: float):
    """Inverse of Hall's transformation g(t) = t + a t^2/3 + a^2 t^3/27 + a/6, a = skew/sqrt(n).
    g(T) of the studentised mean T = (mean - mu)/se is N(0,1) to second order; g is monotone."""
    if abs(a) < 1e-10:
        return c
    return 3.0 / a * (np.cbrt(1.0 + a * (c - a / 6.0)) - 1.0)


def interval_hall(c: Components):
    q = _tq(c)
    a = skewness(c.z) / np.sqrt(c.n)
    lo = c.point - c.se * hall_inverse(q, a)
    hi = c.point - c.se * hall_inverse(-q, a)
    return _out(lo, c.point, hi)


def _boot_t_quantiles(stats: np.ndarray, fallback: float):
    """Quantiles of the bootstrap t statistics; resamples with zero spread give non-finite
    values and are dropped. If too few remain, fall back to symmetric t quantiles."""
    stats = stats[np.isfinite(stats)]
    if len(stats) < B_BOOT // 2:
        return -fallback, fallback
    return np.quantile(stats, [(1 - LEVEL) / 2, (1 + LEVEL) / 2])


def interval_boott(c: Components, seed: int):
    """Studentised bootstrap over the draws: T* = (mean* - mean) / (sd* / sqrt(n))."""
    rng = np.random.default_rng(seed + 1)
    n, z = c.n, c.z
    zb = z[rng.integers(0, n, size=(B_BOOT, n))]
    sd = zb.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tb = (zb.mean(axis=1) - z.mean()) / (sd / np.sqrt(n))
    tb[sd < 0.1 * z.std(ddof=1)] = np.nan  # near-degenerate resamples give wild t*
    q_lo, q_hi = _boot_t_quantiles(tb, _tq(c))
    return _out(c.point - c.se * q_hi, c.point, c.point - c.se * q_lo)


def jackknife(comp):
    """t interval with a delete-one jackknife variance over the draws, recomputing the whole
    estimator (including any fitted coefficients) without each draw. For a plain mean this
    equals the usual variance; for PPI++ / model-assisted estimators it adds the variability
    of the estimated coefficients that the plug-in variance ignores."""

    def run(d, s, seed, stan_kwargs):
        c = comp(d, s, seed)
        n = s.n
        loo = np.array([comp(d, Sample(s.design, np.delete(s.draws, i), q=s.q), seed).point for i in range(n)])
        se = np.sqrt(c.fpc * (n - 1) / n * ((loo - loo.mean()) ** 2).sum())
        q = _tq(c)
        return _out(c.point - q * se, c.point, c.point + q * se)

    return run


def ppi_pp_boott(d, s, seed):
    """Bootstrap-t for PPI++ that re-estimates the coefficient in every resample."""
    c = comp_ppi_pp(d, s, seed)
    y, x, gbar, n = d.f[s.draws].astype(float), d.g[s.draws].astype(float), d.g.mean(), s.n
    idx = np.random.default_rng(seed + 1).integers(0, n, size=(B_BOOT, n))
    yb, xb = y[idx], x[idx]
    xc, yc = xb - xb.mean(1, keepdims=True), yb - yb.mean(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where((xc ** 2).sum(1) > 0, (xc * yc).sum(1) / (xc ** 2).sum(1), 0.0)
        zb = yb - lam[:, None] * xb
        theta = lam * gbar + zb.mean(1)
        se_b = np.sqrt(zb.var(axis=1, ddof=2) / n)
        tb = (theta - c.point) / se_b
    # resamples whose detector counts or residuals barely vary give wild t*; treat as failed
    tb[(xc ** 2).sum(1) < 0.1 * ((x - x.mean()) ** 2).sum()] = np.nan
    tb[se_b < 0.1 * np.sqrt(c.z.var(ddof=2) / n)] = np.nan
    q_lo, q_hi = _boot_t_quantiles(tb, _tq(c))
    se = np.sqrt(c.z.var(ddof=2) / n)  # studentise like the replicates; fpc applied to the interval
    scale = np.sqrt(c.fpc) * se
    return _out(c.point - scale * q_hi, c.point, c.point - scale * q_lo)


# --- registration ----------------------------------------------------------------------------


def _wrap(comp, how):
    def run(d, s, seed, stan_kwargs):
        c = comp(d, s, seed)
        if how == "t":
            return interval_t(c)
        if how == "hall":
            return interval_hall(c)
        return interval_boott(c, seed)
    return run


ma_poisson_comp = comp_model_assisted(PoissonCurve, False)
ma_flex_comp = comp_model_assisted(ShrunkLevels, True)

_HOW_LABEL = {"t": "t", "hall": "skew-corrected t", "boott": "bootstrap-t"}
for base, comp, designs, label in [
    ("ppi", comp_ppi, ("srs",), "difference (PPI)"),
    ("hh", comp_hh, ("pps",), "labels only"),
    ("hh_cv", comp_hh_cv, ("pps",), "difference (PPI)"),
    ("ma_poisson", ma_poisson_comp, ("srs", "pps"), "model-assisted (Poisson)"),
    ("ma_flex", ma_flex_comp, ("srs", "pps"), "model-assisted (flexible)"),
]:
    for how in ("t", "hall", "boott"):
        register(Method(f"{base}_{how}", _wrap(comp, how), designs, f"{label}, {_HOW_LABEL[how]}",
                        "small-sample intervals"))

register(Method("ppi_pp_t", _wrap(comp_ppi_pp, "t"), ("srs",), "regression (PPI++), t", "small-sample intervals"))
register(Method("ppi_pp_hall", _wrap(comp_ppi_pp, "hall"), ("srs",), "regression (PPI++), skew-corrected t",
                "small-sample intervals"))
register(Method("ppi_pp_jack", jackknife(comp_ppi_pp), ("srs",), "regression (PPI++), jackknife t",
                "small-sample intervals"))
register(Method("ma_poisson_jack", jackknife(ma_poisson_comp), ("srs", "pps"), "model-assisted (Poisson), jackknife t",
                "small-sample intervals"))
register(Method("ppi_pp_boott", lambda d, s, seed, kw: ppi_pp_boott(d, s, seed), ("srs",),
                "regression (PPI++), bootstrap-t", "small-sample intervals"))
