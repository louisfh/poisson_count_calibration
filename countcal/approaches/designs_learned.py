"""Learned labeling designs: use the detector counts g to decide which images to hand-count.

All designs select on g only (plus randomness), so model-based methods stay valid under
them; here each design also gets design-based estimators whose inference follows the design.

Designs (registered in countcal.designs.DESIGNS):
  strat_prop   stratified SRS on bins of g, proportional allocation
  strat_sqrt   stratified SRS on bins of g, allocation ∝ N_h sqrt(mean g_h + 1): Neyman
               allocation under a Poisson-like variance heuristic, Var(f | g) ∝ g + 1
  neyman2      two-phase: phase 1 = strat_prop with ~30% of n; phase 2 = Neyman allocation
               of the rest using residual SDs estimated from phase 1
  pps_sqrt     PPS with replacement, q ∝ sqrt(g + 1) (the PPS analogue of strat_sqrt)

Strata are contiguous ranges of g with roughly equal numbers of images (ties stay
together), at most min(8, n // 5) of them and at least 2 labels each.

Estimators (design-based, 90% intervals with t quantiles; "_z" variants use the normal
quantile like the original estimators, to separate the design effect from the t correction):
  st_mean, st_diff, st_ma    stratified mean of f / of f - g / of f - m(g), m = Poisson curve
                             fit on the labels with weights N_h / n_h; Satterthwaite df
  tp_diff, tp_ma             two-phase: phase-1 labels enter exactly; the remaining images'
                             total is mean m1(g) + stratified phase-2 residual mean, where m1 is
                             fit on phase 1 only (m1 = g for tp_diff). Conditionally on phase 1
                             this is a fixed stratified design, so it is design-unbiased and its
                             variance estimate is valid despite the adaptive allocation.
  tp_naive_diff, tp_naive_ma both phases pooled as one stratified sample (ignores adaptivity)
  st_ratio, st_reg           combined ratio (m = R g) and regression (m = lam g, PPI++-like)
                             working models under the stratified designs
  sq_hh, sq_hh_cv, sq_ma     Hansen-Hurwitz on f / f - g / f - m(g) under pps_sqrt; t(n - 1)

Also strat_lin: like strat_sqrt but allocation ∝ N_h (mean g_h + 1) (SD ∝ g, multiplicative
errors). Working-model fits use fit_curve (MAP with the calib_poisson.stan priors), because
plain IRLS diverges on small samples with near-separation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from .. import designs
from ..designs import Sample
from ..estimators import Z90
from ..methods import Method, register


G_OFFSET = 1.0
MAX_STRATA = 8
MIN_PER_STRATUM = 2
PHASE1_FRAC = 0.3


def _poisson_curve(beta, g):
    return np.exp(np.clip(beta[0] + beta[1] * np.log(g + G_OFFSET), -30, 30))


def fit_curve(g, f, w, iters: int = 100) -> np.ndarray:
    """MAP fit of log E[f | g] = a + b log(g + 1) with design weights w (rescaled to sum to the
    number of labels) and the priors of calib_poisson.stan: a ~ N(0, 2), b ~ N(1, 1). Plain
    IRLS diverges when a small sample has (near-)separation, e.g. all-zero counts at low g."""
    x = np.log(g + G_OFFSET)
    y = np.asarray(f, dtype=float)
    w = np.asarray(w, dtype=float) * len(y) / np.sum(w)
    X = np.column_stack([np.ones_like(x), x])
    prior_mean, prior_prec = np.array([0.0, 1.0]), np.diag([1 / 4, 1.0])
    beta = np.array([np.log(max(np.average(y, weights=w), 0.1)), 1.0])
    for _ in range(iters):
        mu = np.exp(np.clip(X @ beta, -30, 30))
        grad = X.T @ (w * (y - mu)) - prior_prec @ (beta - prior_mean)
        H = X.T @ (X * (w * mu)[:, None]) + prior_prec
        step = np.linalg.solve(H, grad)
        step *= min(1.0, 1.0 / max(np.max(np.abs(step)), 1e-12))  # damped Newton
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


# --- strata and allocation --------------------------------------------------------------


def make_strata(g: np.ndarray, H: int) -> np.ndarray:
    """Stratum id (0..H'-1) per image: contiguous ranges of g with ~N/H images each.
    Images with the same g are in the same stratum, so H' <= H when g has heavy ties."""
    values, counts = np.unique(g, return_counts=True)
    target = len(g) / H
    sid_of_value, sid, filled = np.empty(len(values), dtype=int), 0, 0
    for k, c in enumerate(counts):
        if filled > 0 and filled + c / 2 > target and sid < H - 1:
            sid, filled = sid + 1, 0
        sid_of_value[k] = sid
        filled += c
    # merge a tiny last stratum into its neighbour
    last = sid_of_value == sid_of_value.max()
    if sid_of_value.max() > 0 and counts[last].sum() < max(MIN_PER_STRATUM * 2, target / 3):
        sid_of_value[last] -= 1
    return sid_of_value[np.searchsorted(values, g)]


def allocate(n: int, weight: np.ndarray, N_h: np.ndarray, lo: np.ndarray | None = None) -> np.ndarray:
    """Integer allocation n_h ∝ weight, with lo_h <= n_h <= N_h and sum n_h = n
    (largest-remainder rounding, iterated so the bounds hold)."""
    lo = np.minimum(np.full(len(N_h), MIN_PER_STRATUM) if lo is None else lo, N_h)
    n_h = lo.copy()
    free = np.ones(len(N_h), dtype=bool)
    for _ in range(len(N_h) + 1):
        rest = n - n_h.sum()
        if rest <= 0:
            break
        free &= n_h < N_h
        w = np.where(free, weight, 0.0)
        if w.sum() <= 0:
            w = np.where(free, 1.0, 0.0)
        target = n_h + rest * w / w.sum()
        new = np.minimum(np.maximum(np.floor(target).astype(int), n_h), N_h)
        short = n - new.sum()
        if short > 0:
            order = np.argsort(-(target - np.floor(target)))
            for h in order:
                if short == 0:
                    break
                if free[h] and new[h] < N_h[h]:
                    new[h] += 1
                    short -= 1
        n_h = new
        if n_h.sum() >= n:
            break
    return n_h


def _n_strata(n: int) -> int:
    return int(np.clip(n // 5, 2, MAX_STRATA))


def _stratified_draw(rng, strata, n_h, exclude=None) -> np.ndarray:
    out = []
    for h, k in enumerate(n_h):
        pool = np.flatnonzero(strata == h)
        if exclude is not None:
            pool = np.setdiff1d(pool, exclude)
        if k > 0:
            out.append(rng.choice(pool, size=k, replace=False))
    return np.concatenate(out) if out else np.array([], dtype=int)


def _strat_design(name: str, alloc: str):
    def design(g: np.ndarray, n: int, rng: np.random.Generator) -> Sample:
        strata = make_strata(g, _n_strata(n))
        H = strata.max() + 1
        N_h = np.bincount(strata, minlength=H)
        mean_g = np.bincount(strata, weights=g, minlength=H) / N_h
        if alloc == "prop":
            weight = N_h.astype(float)
        elif alloc == "sqrt":  # Neyman under Var(f | g) ∝ g + 1 (Poisson-like errors)
            weight = N_h * np.sqrt(mean_g + 1)
        else:  # "lin": Neyman under SD(f | g) ∝ g + 1 (multiplicative errors)
            weight = N_h * (mean_g + 1)
        n_h = allocate(n, weight, N_h)
        draws = _stratified_draw(rng, strata, n_h)
        return Sample(name, draws, meta={"strata": strata, "N_h": N_h, "n_h": n_h})
    return design


def _neyman2(g: np.ndarray, n: int, rng: np.random.Generator, label) -> Sample:
    """Two-phase adaptive design. Phase 1: strat_prop with ~30% of n. Then fit the Poisson
    working model m1 on phase 1, estimate each stratum's residual SD (shrunk towards the
    pooled SD, since phase-1 strata hold few labels), and Neyman-allocate the rest of n over
    the not-yet-labeled images. `label(idx)` returns true counts for phase-1 images only."""
    strata = make_strata(g, _n_strata(n))
    H = strata.max() + 1
    N_h = np.bincount(strata, minlength=H)
    n1 = max(MIN_PER_STRATUM * H, int(round(PHASE1_FRAC * n)))
    n1 = min(n1, max(n - MIN_PER_STRATUM * H, MIN_PER_STRATUM * H))
    n1_h = allocate(n1, N_h.astype(float), N_h)
    ph1 = _stratified_draw(rng, strata, n1_h)
    f1 = np.asarray(label(ph1), dtype=float)
    w1 = (N_h / np.maximum(n1_h, 1))[strata[ph1]]
    beta1 = fit_curve(g[ph1], f1, w1)
    r1 = f1 - _poisson_curve(beta1, g[ph1])
    # prior guess for each stratum's residual variance: pooled variance scaled by the
    # Poisson-like heuristic Var ∝ mean g + 1; each stratum's own estimate (2-3 labels) is
    # combined with it as if the prior were worth 4 degrees of freedom
    pooled = r1.var(ddof=1) if len(r1) > 1 else 1.0
    gbar = np.bincount(strata, weights=g, minlength=H) / N_h + 1
    prior = pooled * gbar / np.average(gbar, weights=N_h)
    S2 = np.empty(H)
    for h in range(H):
        rh = r1[strata[ph1] == h]
        nu = max(len(rh) - 1, 0)
        S2[h] = ((nu * rh.var(ddof=1) if nu else 0.0) + 4 * prior[h]) / (nu + 4)
    N_rem = N_h - n1_h
    n2_h = allocate(n - len(ph1), N_rem * np.sqrt(np.maximum(S2, 1e-12)), N_rem,
                    lo=np.minimum(np.full(H, MIN_PER_STRATUM), N_rem))
    ph2 = _stratified_draw(rng, strata, n2_h, exclude=ph1)
    return Sample("neyman2", np.concatenate([ph1, ph2]),
                  meta={"strata": strata, "N_h": N_h, "n1_h": n1_h, "n2_h": n2_h, "N_rem": N_rem,
                        "n_h": n1_h + n2_h, "phase1": ph1, "phase2": ph2, "beta1": beta1})


_neyman2.adaptive = True


# --- estimators --------------------------------------------------------------------------


def _ci(point, var, df, use_t=True) -> dict:
    se = np.sqrt(max(var, 0.0))
    q = student_t.ppf(0.95, max(df, 1)) if use_t else Z90
    return {"point": float(point), "lo": float(point - q * se), "hi": float(point + q * se)}


def _stratified_mean_var(y, strata_of_y, N_h, n_h, N_pop):
    """Stratified estimate of the population mean of y and its variance + Satterthwaite df."""
    H = len(N_h)
    W = N_h / N_pop
    means = np.array([y[strata_of_y == h].mean() if n_h[h] else 0.0 for h in range(H)])
    s2 = np.array([y[strata_of_y == h].var(ddof=1) if n_h[h] > 1 else 0.0 for h in range(H)])
    a = W ** 2 * (1 - n_h / N_h) / np.maximum(n_h, 1)
    var = float((a * s2).sum())
    den = ((a * s2) ** 2 / np.maximum(n_h - 1, 1)).sum()
    df = var ** 2 / den if den > 0 else max(n_h.sum() - H, 1)
    return float((W * means).sum()), var, df


def _st(kind: str, use_t=True):
    def run(d, s, seed, stan_kwargs):
        f, g, N = d.f, d.g, d.N
        m = s.meta
        idx = s.draws
        sh = m["strata"][idx]
        w = (m["N_h"] / m["n_h"])[sh]
        if kind == "mean":
            pred_all, pred_lab = np.zeros(N), np.zeros(len(idx))
        elif kind == "diff":
            pred_all, pred_lab = g.astype(float), g[idx].astype(float)
        elif kind == "ratio":  # combined ratio estimator: m(g) = R g, R = weighted sum f / sum g
            R = np.sum(w * f[idx]) / max(np.sum(w * g[idx]), 1e-12)
            pred_all, pred_lab = R * g, R * g[idx]
        elif kind == "reg":  # combined regression (GREG with a linear working model, PPI++)
            xb, yb = np.average(g[idx], weights=w), np.average(f[idx], weights=w)
            vx = np.average((g[idx] - xb) ** 2, weights=w)
            lam = np.average((g[idx] - xb) * (f[idx] - yb), weights=w) / vx if vx > 0 else 0.0
            pred_all, pred_lab = lam * g, lam * g[idx]
        else:  # Poisson working model, weights N_h / n_h
            beta = fit_curve(g[idx], f[idx], w)
            pred_all, pred_lab = _poisson_curve(beta, g), _poisson_curve(beta, g[idx])
        mu, var, df = _stratified_mean_var(f[idx] - pred_lab, sh, m["N_h"], m["n_h"], N)
        return {"mean": _ci(pred_all.mean() + mu, var, df, use_t)}
    return run


def _tp(kind: str, use_t=True):
    def run(d, s, seed, stan_kwargs):
        f, g, N = d.f, d.g, d.N
        m = s.meta
        ph1, ph2 = m["phase1"], m["phase2"]
        rem = np.ones(N, dtype=bool)
        rem[ph1] = False
        if kind == "diff":
            pred = g.astype(float)
        else:
            pred = _poisson_curve(m["beta1"], g)
        strata = m["strata"]
        # total over the remaining images = sum of predictions + stratified residual correction
        N_rem, n2_h = m["N_rem"], m["n2_h"]
        H = len(N_rem)
        r2 = f[ph2] - pred[ph2]
        sh2 = strata[ph2]
        corr, var = 0.0, 0.0
        terms = []
        for h in range(H):
            rh = r2[sh2 == h]
            if n2_h[h] == 0:
                continue
            corr += N_rem[h] * rh.mean()
            s2 = rh.var(ddof=1) if n2_h[h] > 1 else 0.0
            a = N_rem[h] ** 2 * (1 - n2_h[h] / N_rem[h]) / n2_h[h]
            var += a * s2
            terms.append((a * s2, n2_h[h]))
        total = f[ph1].sum() + pred[rem].sum() + corr
        den = sum(t ** 2 / max(k - 1, 1) for t, k in terms)
        df = var ** 2 / den if den > 0 else max(len(ph2) - H, 1)
        return {"mean": _ci(total / N, var / N ** 2, df, use_t)}
    return run


def _pps_sqrt(g: np.ndarray, n: int, rng: np.random.Generator) -> Sample:
    w = np.sqrt(g.astype(float) + 1)
    q = w / w.sum()
    return Sample("pps_sqrt", rng.choice(len(g), size=n, replace=True, p=q), q=q)


def _sq(kind: str, use_t=True):
    def run(d, s, seed, stan_kwargs):
        f, g, N, idx = d.f, d.g, d.N, s.draws
        if kind == "hh":
            pred_all, pred_lab = np.zeros(N), np.zeros(len(idx))
        elif kind == "cv":
            pred_all, pred_lab = g.astype(float), g[idx].astype(float)
        else:
            beta = fit_curve(g[idx], f[idx], 1 / s.q[idx])
            pred_all, pred_lab = _poisson_curve(beta, g), _poisson_curve(beta, g[idx])
        z = (f[idx] - pred_lab) / (N * s.q[idx])
        n = len(z)
        return {"mean": _ci(pred_all.mean() + z.mean(), z.var(ddof=1) / n, n - 1, use_t)}
    return run


# --- registration ------------------------------------------------------------------------

designs.DESIGNS["strat_prop"] = _strat_design("strat_prop", "prop")
designs.DESIGNS["strat_sqrt"] = _strat_design("strat_sqrt", "sqrt")
designs.DESIGNS["strat_lin"] = _strat_design("strat_lin", "lin")
designs.DESIGNS["neyman2"] = _neyman2
designs.DESIGNS["pps_sqrt"] = _pps_sqrt

_ST = ("strat_prop", "strat_sqrt", "strat_lin")
register(Method("st_mean", _st("mean"), _ST, "stratified: labels only", "learned design"))
register(Method("st_diff", _st("diff"), _ST, "stratified: difference", "learned design"))
register(Method("st_diff_z", _st("diff", use_t=False), _ST, "stratified: difference (z)", "learned design"))
register(Method("st_ma", _st("ma"), _ST, "stratified: model-assisted", "learned design"))
register(Method("st_ratio", _st("ratio"), _ST, "stratified: ratio", "learned design"))
register(Method("st_reg", _st("reg"), _ST, "stratified: regression", "learned design"))
register(Method("tp_diff", _tp("diff"), ("neyman2",), "two-phase: difference", "learned design"))
register(Method("tp_ma", _tp("ma"), ("neyman2",), "two-phase: model-assisted", "learned design"))
# the union of both phases analysed as one stratified sample, ignoring that the allocation
# was chosen from phase-1 data (common practice; small bias in principle)
register(Method("tp_naive_diff", _st("diff"), ("neyman2",), "two-phase, pooled: difference", "learned design"))
register(Method("tp_naive_ma", _st("ma"), ("neyman2",), "two-phase, pooled: model-assisted", "learned design"))
register(Method("sq_hh", _sq("hh"), ("pps_sqrt",), "sqrt-PPS: labels only", "learned design"))
register(Method("sq_hh_cv", _sq("cv"), ("pps_sqrt",), "sqrt-PPS: difference", "learned design"))
register(Method("sq_ma", _sq("ma"), ("pps_sqrt",), "sqrt-PPS: model-assisted", "learned design"))
