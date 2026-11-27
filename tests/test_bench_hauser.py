"""Hauser et al. (2011) mass-spring network, ported from the authors' Matlab toolbox
(github.com/helmuthauser/Matlab_Morphological_Computation_Simulation, commit 1e7a17f)."""
import numpy as np
import pytest

from mibook.bench import make
from mibook.bench.bodies import hauser as h


@pytest.fixture(scope="module")
def vdata():
    return h.volterra_data()


@pytest.fixture(scope="module")
def ndata():
    return h.narma_data()


def test_task_data_match_the_toolbox_files(vdata, ndata):
    """Values read from the toolbox's data_Volterra/volterra.mat and data_NARMA/NARMA-L2.mat
    (Python indices 1000, 123456, 400000); our generators reproduce the whole files to 1e-11."""
    idx = [1000, 123456, 400000]
    np.testing.assert_allclose(vdata["y"][idx], [0.00286918847007049, 1.1427955817093696,
                                                 0.3351338759304431], rtol=1e-9)
    np.testing.assert_allclose(vdata["yn"][idx], [-0.8880950934411574, 0.7024502039834762,
                                                  -0.4244844258799058], rtol=1e-9)
    np.testing.assert_allclose(ndata["yn"][idx], [-0.2478022923478946, 0.04032478925211514,
                                                  0.7032995285164726], rtol=1e-9)
    np.testing.assert_allclose(ndata["un"][idx], [-0.6230881457163788, 1.0343266584656534,
                                                  -1.5658172401904624], rtol=1e-9)
    assert len(vdata["u"]) == 500200 and len(ndata["un"]) == 499001


def test_construction_follows_the_paper_and_toolbox():
    net = h.build(n_nodes=30, seed=3)
    assert net.fixed.sum() == 2
    assert np.argmin(net.pos0[:, 0]) in np.flatnonzero(net.fixed)
    assert np.argmax(net.pos0[:, 0]) in np.flatnonzero(net.fixed)
    assert (net.w_in != 0).sum() <= 6 and np.all(net.w_in[net.fixed] == 0)   # ceil(0.2 N)
    assert np.all(np.abs(net.w_in) <= 1)
    assert np.all((net.pos0 >= 0) & (net.pos0 <= 10))
    assert np.all((net.k1 >= 1) & (net.k1 <= 100) & (net.d1 >= 1) & (net.d1 <= 100))
    assert np.all((net.k3 >= 100) & (net.k3 <= 200) & (net.d3 >= 100) & (net.d3 <= 200))
    assert 3 * 30 - 3 - 30 < net.n_springs <= 3 * 30 - 6         # planar Delaunay graph
    np.testing.assert_allclose(net.l0, net.lengths(net.pos0))   # starts at rest


def test_rest_is_an_equilibrium_and_motion_fades():
    net = h.build(seed=0)
    L, _ = h.simulate(net, np.zeros(2000))
    assert np.abs(L - net.l0).max() < 1e-12
    u = np.zeros(20000)
    u[:200] = 1.0
    L, _ = h.simulate(net, u)
    dev = np.abs(L - net.l0).max(axis=1)
    assert dev[300] > 1e-4 and dev[-1] < 0.05 * dev.max()       # fading memory (paper fn. 15)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_network_emulates_what_a_linear_model_cannot(seed, vdata, ndata):
    """Paper Fig. 9c,e: a linear readout of the spring lengths emulates the Volterra operator
    (Eq. 2) and the tenth-order NARMA system (Eq. 8); a linear model on the raw input does
    not. Shortened runs (learning 50 s, washout 20 s, test 10 s); the full toolbox runs give
    normalized test errors 0.07-0.28 (Volterra) and 0.04-0.16 (NARMA) for these seeds
    against 2.0 and 1.03 for the linear model."""
    kw = dict(n_learn=50001, washout=20000, n_test=10000)
    rv = h.volterra_task(h.build(seed=seed, ranges=h.VOLTERRA), data=vdata, **kw)
    rn = h.narma_task(h.build(seed=seed, ranges=h.NARMA), data=ndata, **kw)
    assert rv["mse_lr"] > 1.9 and rv["mse"] < 0.1 * rv["mse_lr"]
    assert rn["mse_lr"] > 0.95 and rn["mse"] < 0.1 * rn["mse_lr"]


def test_registered_body_matches_the_toolbox_scheme():
    sim = make("hauser_network", seed=2, sigma_T2=0.0)
    net = sim.net
    n = 3000
    u = 0.5 * np.sin(2 * np.pi * 2.0 * np.arange(n) * h.DT)
    out, (q, v) = sim.simulate(n, h.DT, 1, u=u[:, None, None], rng=np.random.default_rng(0),
                               v_init=np.zeros((1, sim.n)))
    P = net.pos0[None].copy()
    P[:, ~net.fixed] = q.reshape(1, -1, 2)
    L_ref, _ = h.simulate(net, u)
    np.testing.assert_allclose(net.lengths(P)[0], L_ref[-1], atol=2e-2 * np.abs(L_ref[-1] - net.l0).max())
    # The two differ at O(dt): exact stretch rate against a one-step backward difference.
