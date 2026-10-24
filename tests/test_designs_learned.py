import numpy as np
import pytest

from countcal import data, designs
from countcal.approaches.designs_learned import allocate, make_strata
from countcal.methods import all_methods


@pytest.fixture(scope="module")
def pop():
    return data.synth_crowding()


def test_strata_keep_ties_together(pop):
    st = make_strata(pop.g, 5)
    for v in np.unique(pop.g):
        assert len(np.unique(st[pop.g == v])) == 1
    assert np.all(np.diff(st[np.argsort(pop.g, kind="stable")]) >= 0)  # contiguous in g


def test_allocate_respects_bounds():
    N_h = np.array([3, 50, 100, 2])
    n_h = allocate(40, np.array([10.0, 1.0, 1.0, 50.0]), N_h)
    assert n_h.sum() == 40 and np.all(n_h <= N_h) and np.all(n_h >= 2)


@pytest.mark.parametrize("design,method", [
    ("strat_prop", "st_mean"), ("strat_prop", "st_diff"), ("strat_sqrt", "st_diff"), ("strat_sqrt", "st_ma"),
    ("neyman2", "tp_diff"), ("neyman2", "tp_ma"), ("pps_sqrt", "sq_hh_cv"), ("pps_sqrt", "sq_ma"),
])
def test_learned_designs_unbiased(pop, design, method):
    """Design-based estimators under the learned designs are (approximately) unbiased."""
    rng = np.random.default_rng(0)
    fn = all_methods()[method].fn
    est = [fn(pop, designs.draw(design, pop.g, 50, rng, label=lambda i: pop.f[i]), 0, None)["mean"]["point"]
           for _ in range(1500)]
    se = np.std(est) / np.sqrt(len(est))
    tol = 4 * se + (0.02 * pop.f.mean() if method.endswith("_ma") else 0)  # fitted working model: O(1/n) bias
    assert abs(np.mean(est) - pop.f.mean()) < tol


def test_neyman2_phase2_uses_only_phase1_labels(pop):
    seen = []

    def label(idx):
        seen.append(np.asarray(idx).copy())
        return pop.f[idx]

    s = designs.draw("neyman2", pop.g, 50, np.random.default_rng(3), label=label)
    assert len(seen) == 1 and np.array_equal(np.sort(seen[0]), np.sort(s.meta["phase1"]))
    assert len(s.draws) == 50 and len(np.unique(s.draws)) == 50
