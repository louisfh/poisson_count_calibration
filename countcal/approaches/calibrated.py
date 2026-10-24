"""Combining model-based and design-based inference.

Three routes, all built on the same calibration curve log E[f | g] = a + b log(g + 1):

calibrated Bayes (Little 1993, 2006; Zheng & Little 2003) — keep inference Bayesian, but
model the outcome flexibly in the design variable g so that selection on g is ignorable
without trusting the parametric curve:
  cb_ps        NB curve + partially pooled deviation for each distinct g (Bayesian
               post-stratification on g)
  cb_ps_site   the same + site random effects (multi-site / hierarchical)

pseudo-likelihood — weight the likelihood by the design weights (Savitsky & Toth 2016):
  poisson_w    weighted Poisson GLM; PPS only (under SRS it equals `poisson`)

model-assisted (Särndal, Swensson & Wretman 1992; Wu & Sitter 2001) — keep inference
design-based, use the model only as a working predictor; design-unbiased whatever the model:
  ma_poisson   GREG / PPI with the Poisson calibration curve as working model
  ma_flex      the same with a flexible working model (curve + empirical-Bayes shrunken
               per-g corrections), cross-fitted so the residuals are honest
"""

from __future__ import annotations

import numpy as np

from ..estimators import Z90, run_stan, stan_data
from ..methods import Method, register

G_OFFSET = 1.0
K_FOLDS = 5


# --- calibrated Bayes --------------------------------------------------------------------


def _hier_data(d, s, use_level: bool, use_site: bool) -> dict:
    levels, lev_all = np.unique(d.g, return_inverse=True)
    if use_site:
        _, site_all = np.unique(d.site.astype(str), return_inverse=True)
    else:
        site_all = np.zeros(d.N, dtype=int)
    lab = s.labeled
    mask = np.ones(d.N, dtype=bool)
    mask[lab] = False
    cells, mult = np.unique(np.stack([lev_all[mask], site_all[mask]], axis=1), axis=0, return_counts=True)
    return {
        "N_lab": len(lab),
        "f_lab": d.f[lab].astype(int).tolist(),
        "L": len(levels),
        "x_lev": np.log(levels + G_OFFSET).tolist(),
        "lev_lab": (lev_all[lab] + 1).tolist(),
        "S": int(site_all.max()) + 1,
        "site_lab": (site_all[lab] + 1).tolist(),
        "C": len(cells),
        "lev_unl": (cells[:, 0] + 1).tolist() if len(cells) else [],
        "site_unl": (cells[:, 1] + 1).tolist() if len(cells) else [],
        "mult_unl": mult.tolist(),
        "use_level": int(use_level),
        "use_site": int(use_site),
    }


def _cb(use_site: bool):
    def run(d, s, seed, stan_kwargs):
        return run_stan("calib_hier", _hier_data(d, s, True, use_site), seed, **(stan_kwargs or {}))
    return run


# --- pseudo-likelihood -------------------------------------------------------------------


def _design_weights(s) -> tuple[np.ndarray, np.ndarray]:
    """Per distinct labeled image: (indices, weight). Weights ∝ (draw count) / q under PPS,
    1 under SRS; normalised to sum to the number of labeled images."""
    lab, counts = np.unique(s.draws, return_counts=True)
    w = counts / s.q[lab] if s.q is not None else counts.astype(float)
    return lab, w * len(lab) / w.sum()


def _poisson_w(d, s, seed, stan_kwargs):
    lab, w = _design_weights(s)
    data = stan_data(d.f, d.g, s) | {"w": w.tolist(), "g_offset": G_OFFSET}
    assert data["f_lab"] == d.f[lab].astype(int).tolist()  # stan_data orders labeled images the same way
    return run_stan("calib_poisson_w", data, seed, **(stan_kwargs or {}))


# --- model-assisted ----------------------------------------------------------------------


SLOPE_MAX = 4.0


def fit_poisson_glm(x, y, w) -> np.ndarray:
    """Weighted Poisson regression log mu = a + b x, MAP with the Bayesian models' prior on the
    slope (b ~ N(1, 1), 0 <= b <= SLOPE_MAX) and a flat intercept; returns (a, b).

    For fixed b the intercept has the closed form a(b) = log(sum w y / sum w exp(b x)); the
    profile log-likelihood is concave in b, so a bounded scalar search finds the optimum.
    Weights are rescaled to sum to the number of observations, so the prior matters only at
    small n. (Unpenalised IRLS diverges at small n when the labeled g are all equal or nearly
    separate the counts, and an unpenalised slope can reach the bound and extrapolate
    (g + 1)^4 to unlabeled images.)"""
    from scipy.optimize import minimize_scalar

    w = w * len(w) / w.sum()
    swy = float(np.sum(w * y))
    if swy <= 0:  # no labeled animals: predict ~0 everywhere, the residual term does the rest
        return np.array([np.log(1e-6), 0.0])

    def intercept(b):
        return np.log(swy) - np.log(np.sum(w * np.exp(b * (x - x.max())))) - b * x.max()

    if np.ptp(x) == 0:
        return np.array([intercept(1.0), 1.0])
    res = minimize_scalar(lambda b: -(swy * intercept(b) + b * np.sum(w * y * x)) + 0.5 * (b - 1.0) ** 2,
                          bounds=(0.0, SLOPE_MAX), method="bounded", options={"xatol": 1e-7})
    return np.array([intercept(res.x), res.x])


class PoissonCurve:
    """Working model m(g) = exp(a + b log(g + 1))."""

    def fit(self, g, f, w):
        self.beta = fit_poisson_glm(np.log(g + G_OFFSET), f.astype(float), w)
        return self

    def predict(self, g):
        return np.exp(np.clip(self.beta[0] + self.beta[1] * np.log(g + G_OFFSET), -30, 30))


class ShrunkLevels(PoissonCurve):
    """Poisson curve plus an additive correction for each distinct g, shrunk by empirical
    Bayes towards 0 (a design-weighted, frequentist analogue of cb_ps)."""

    def fit(self, g, f, w):
        super().fit(g, f, w)
        r = f - super().predict(g)
        levels = np.unique(g)
        n_l = np.array([w[g == v].sum() for v in levels])
        k_l = np.array([(g == v).sum() for v in levels])
        mean_l = np.array([np.average(r[g == v], weights=w[g == v]) for v in levels])
        within = np.average((r - mean_l[np.searchsorted(levels, g)]) ** 2, weights=w)
        within = within * len(r) / max(len(r) - len(levels), 1)
        # method-of-moments between-level variance, over levels with labels
        se2 = within / np.maximum(k_l, 1)
        tau2 = max(0.0, np.average(mean_l ** 2 - se2, weights=n_l))
        shrink = tau2 / (tau2 + se2) if tau2 > 0 else np.zeros_like(se2)
        self.corr = dict(zip(levels.tolist(), (shrink * mean_l).tolist()))
        return self

    def predict(self, g):
        base = super().predict(g)
        return base + np.array([self.corr.get(v, 0.0) for v in g.tolist()])


def _model_assisted(model_cls, cross_fit: bool):
    """Difference / GREG estimator with a working model:
        mean(m(g), all images) + design-weighted mean of (f - m(g)) over the labeled draws,
    with design-based variance. With cross_fit, each labeled image's residual comes from a
    model fit without its fold, and m on the population is the average of the fold models."""

    def run(d, s, seed, stan_kwargs):
        draws = s.draws
        f, g, N, n = d.f, d.g, d.N, len(s.draws)
        w_draw = 1 / s.q[draws] if s.q is not None else np.ones(n)
        if cross_fit:
            rng = np.random.default_rng(seed)
            uniq = np.unique(draws)
            fold_of = dict(zip(uniq.tolist(), rng.permutation(len(uniq)) % K_FOLDS))
            fold = np.array([fold_of[i] for i in draws.tolist()])
            pred_lab = np.empty(n)
            m_all = np.zeros(N)
            folds = [k for k in range(K_FOLDS) if (fold == k).any()]
            for k in folds:
                tr = fold != k
                model = model_cls().fit(g[draws[tr]], f[draws[tr]], w_draw[tr])
                pred_lab[fold == k] = model.predict(g[draws[fold == k]])
                m_all += model.predict(g) / len(folds)
        else:
            model = model_cls().fit(g[draws], f[draws], w_draw)
            pred_lab = model.predict(g[draws])
            m_all = model.predict(g)
        r = f[draws] - pred_lab
        if s.q is None:  # SRS: difference estimator with finite-population correction
            point = m_all.mean() + r.mean()
            se = np.sqrt((1 - n / N) * r.var(ddof=1) / n)
        else:  # PPS with replacement: Hansen-Hurwitz on the residuals
            z = r / (N * s.q[draws])
            point = m_all.mean() + z.mean()
            se = z.std(ddof=1) / np.sqrt(n)
        return {"mean": {"point": float(point), "lo": float(point - Z90 * se), "hi": float(point + Z90 * se)}}

    return run


register(Method("cb_ps", _cb(False), None, "calibrated Bayes", "calibrated Bayes"))
register(Method("cb_ps_site", _cb(True), None, "calibrated Bayes + sites", "calibrated Bayes",
                needs_sites=True))
register(Method("poisson_w", _poisson_w, ("pps",), "weighted Poisson GLM", "pseudo-likelihood"))
register(Method("ma_poisson", _model_assisted(PoissonCurve, False), ("srs", "pps"), "model-assisted (Poisson)",
                "model-assisted"))
register(Method("ma_flex", _model_assisted(ShrunkLevels, True), ("srs", "pps"), "model-assisted (flexible)",
                "model-assisted"))
