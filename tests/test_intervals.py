import numpy as np
import pytest

from countcal import data, designs
from countcal.approaches import intervals as I
from countcal.estimators import Z90
from countcal.methods import all_methods


@pytest.fixture(scope="module")
def pop():
    return data.synth_double_count()


@pytest.mark.parametrize("design,base,comp", [
    ("srs", "ppi", I.comp_ppi), ("srs", "ppi_pp", I.comp_ppi_pp), ("pps", "hh", I.comp_hh),
    ("pps", "hh_cv", I.comp_hh_cv), ("srs", "ma_poisson", I.ma_poisson_comp), ("pps", "ma_poisson", I.ma_poisson_comp),
    ("srs", "ma_flex", I.ma_flex_comp), ("pps", "ma_flex", I.ma_flex_comp),
])
def test_components_reproduce_registered_estimators(pop, design, base, comp):
    s = designs.draw(design, pop.g, 40, np.random.default_rng(5))
    ref = all_methods()[base].fn(pop, s, 11, None)["mean"]
    c = comp(pop, s, 11)
    assert c.point == pytest.approx(ref["point"], rel=1e-10)
    if base != "ppi_pp":  # ppi_pp's registered interval uses ddof = 1
        assert c.point + Z90 * c.se == pytest.approx(ref["hi"], rel=1e-10)


def test_hall_inverse_inverts_transform():
    for a in [-0.5, -0.1, 0.0, 0.2, 0.7]:
        t = np.linspace(-3, 3, 13)
        gt = t + a * t ** 2 / 3 + a ** 2 * t ** 3 / 27 + a / 6
        assert np.allclose(I.hall_inverse(gt, a), t)


def _coverage(how, n=10, reps=4000):
    rng = np.random.default_rng(0)
    hits = 0
    for r in range(reps):
        z = rng.exponential(size=n)  # right-skewed, mean 1
        c = I.Components(z.mean(), z, 1.0)
        out = (I.interval_hall(c) if how == "hall" else I.interval_t(c))["mean"]
        hits += out["lo"] <= 1.0 <= out["hi"]
    return hits / reps


def test_skew_correction_improves_coverage_for_skewed_data():
    assert _coverage("hall") > _coverage("t") + 0.01


def test_bootstrap_t_is_reproducible_and_contains_point(pop):
    s = designs.draw("srs", pop.g, 25, np.random.default_rng(3))
    a = all_methods()["ppi_boott"].fn(pop, s, 7, None)["mean"]
    b = all_methods()["ppi_boott"].fn(pop, s, 7, None)["mean"]
    assert a == b and a["lo"] < a["point"] < a["hi"]
    j = all_methods()["ppi_pp_boott"].fn(pop, s, 7, None)["mean"]
    assert j["lo"] < j["point"] < j["hi"]


def test_jackknife_equals_usual_variance_for_a_plain_mean(pop):
    s = designs.draw("pps", pop.g, 30, np.random.default_rng(9))
    jack = I.jackknife(I.comp_hh_cv)(pop, s, 1, None)["mean"]
    t = I.interval_t(I.comp_hh_cv(pop, s, 1))["mean"]
    assert jack["lo"] == pytest.approx(t["lo"]) and jack["hi"] == pytest.approx(t["hi"])
