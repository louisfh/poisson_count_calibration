import numpy as np

from countcal import data, designs
from countcal.approaches import hybrid
from countcal.methods import all_methods


def test_variance_prior_keeps_intervals_open_when_residuals_are_all_equal():
    d = data.CountData("perfect", f=np.r_[np.zeros(500, int), np.ones(500, int)], g=np.r_[np.zeros(500, int), np.ones(500, int)])
    s = designs.srs(d.g, 25, np.random.default_rng(0))
    m = all_methods()
    plain = m["ppi"].fn(d, s, 0, None)["mean"]
    vp = m["ppi_vp"].fn(d, s, 0, None)["mean"]
    assert plain["hi"] - plain["lo"] == 0  # every residual is 0
    assert vp["hi"] - vp["lo"] > 0
    assert vp["point"] == plain["point"]


def test_variance_prior_negligible_with_large_sample_variance():
    rng = np.random.default_rng(1)
    g = rng.poisson(5, 2000)
    f = g + rng.poisson(20, 2000)  # residual variance ~20 >> prior variance ~5.5
    d = data.CountData("noisy", f=f, g=g)
    s = designs.srs(d.g, 400, np.random.default_rng(2))
    m = all_methods()
    w_plain = np.diff([m["ppi_t"].fn(d, s, 0, None)["mean"][k] for k in ("lo", "hi")])[0]
    w_vp = np.diff([m["ppi_vp"].fn(d, s, 0, None)["mean"][k] for k in ("lo", "hi")])[0]
    assert abs(w_vp / w_plain - 1) < 0.02


def test_prior_variance_pps_matches_srs_for_g_plus_half():
    d = data.synth_crowding()
    s = designs.pps(d.g, 20, np.random.default_rng(3))
    assert np.isclose(hybrid.prior_variance(d, s), d.g.mean() + 0.5)
