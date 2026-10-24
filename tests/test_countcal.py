import logging

import numpy as np
import pytest

from countcal import data, designs
from countcal import estimators as E


@pytest.fixture(scope="module")
def pop():
    return data.synth_crowding()


@pytest.mark.parametrize("design,method", [(d, m) for d, ms in E.DESIGN_BASED.items() for m in ms])
def test_design_based_unbiased(pop, design, method):
    """Design-based estimators are (approximately) unbiased for the population mean."""
    rng = np.random.default_rng(0)
    fn = E.DESIGN_BASED[design][method]
    est = [fn(pop.f, pop.g, designs.draw(design, pop.g, 50, rng))["point"] for _ in range(4000)]
    se = np.std(est) / np.sqrt(len(est))
    # ppi_pp estimates its coefficient from the sample, so allow its O(1/n) bias.
    tol = 4 * se + (0.02 * pop.f.mean() if method == "ppi_pp" else 0)
    assert abs(np.mean(est) - pop.f.mean()) < tol


def test_pps_probabilities_cover_zero_predictions():
    g = np.array([0, 0, 1, 5])
    s = designs.pps(g, 10, np.random.default_rng(0))
    assert np.all(s.q > 0) and np.isclose(s.q.sum(), 1)


def test_stan_data_collapses_unlabeled(pop):
    s = designs.srs(pop.N, 30, np.random.default_rng(1))
    d = E.stan_data(pop.f, pop.g, s)
    assert d["N_lab"] == 30
    assert sum(d["g_mult"]) == pop.N - 30
    assert len(d["g_unique"]) == len(set(d["g_unique"]))


@pytest.mark.parametrize("name", [n for n in data.available_datasets() if not n.startswith("synth")])
def test_processed_datasets_valid(name):
    d = data.load(name)
    assert d.N > 0 and (d.f >= 0).all() and (d.g >= 0).all()


@pytest.mark.slow
@pytest.mark.parametrize("method", list(E.MODEL_BASED))
def test_models_recover_mean_when_well_specified(method):
    import cmdstanpy  # noqa: F401

    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    pop = data.synth_well_specified()
    s = designs.srs(pop.N, 200, np.random.default_rng(2))
    r = E.fit_model(method, pop.f, pop.g, s, seed=3, chains=2)
    assert r["mean"]["lo"] <= pop.f.mean() <= r["mean"]["hi"]
    assert r["diag"]["max_rhat"] < 1.05
