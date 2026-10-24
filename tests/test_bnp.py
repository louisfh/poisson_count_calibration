import numpy as np
import pytest

from countcal import data, designs
from countcal.approaches import bnp


def test_g_strata_merges_to_min_labels():
    levels = np.arange(8)
    n_lab = np.array([4, 1, 0, 6, 2, 2, 1, 0])
    st = bnp.g_strata(levels, n_lab, k_min=5)
    for h in np.unique(st):
        assert n_lab[st == h].sum() >= 5
    assert np.all(np.diff(st) >= 0)  # strata are runs of adjacent levels


def test_linear_curve_slope_nonnegative():
    g = np.array([0.0, 1, 2, 3])
    m = bnp.linear_curve(g, np.array([3.0, 2, 1, 0]), np.ones(4))
    assert m(10.0) == pytest.approx(1.5)  # slope clipped to 0 -> weighted mean


@pytest.mark.parametrize("design", ["srs", "pps"])
def test_polya_unbiased_and_calibrated(design):
    """Approximately unbiased for the population mean, with roughly nominal coverage, on a
    population where the linear working curve is wrong (crowding)."""
    pop = data.synth_crowding()
    rng = np.random.default_rng(0)
    pts, cov = [], []
    for rep in range(150):
        s = designs.draw(design, pop.g, 60, rng)
        x = bnp.polya_draws(pop, s, rep, bnp._alpha_calibrated, n_draws=1000)
        lo, hi = np.quantile(x, [0.05, 0.95])
        pts.append(x.mean())
        cov.append(lo <= pop.f.mean() <= hi)
    se = np.std(pts) / np.sqrt(len(pts))
    assert abs(np.mean(pts) - pop.f.mean()) < 4 * se + 0.02 * pop.f.mean()
    assert np.mean(cov) > 0.75
