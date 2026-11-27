"""Hauser et al. (2011) mass-spring network: construction and the claims of the main text
that do not depend on details left to their supplementary material (placement area, input
scale). Task errors are compared only qualitatively until that material is read."""
import numpy as np

from mibook.bench import make
from mibook.bench.bodies import hauser as h


def test_construction_follows_section_4():
    net = h.build(n_nodes=30, seed=3)
    assert net.fixed.sum() == 2                                     # leftmost and rightmost
    assert np.argmin(net.pos0[:, 0]) in np.flatnonzero(net.fixed)
    assert np.argmax(net.pos0[:, 0]) in np.flatnonzero(net.fixed)
    assert (net.w_in != 0).sum() == 6                               # 20% of 30 nodes
    assert np.all(np.abs(net.w_in) <= 1)
    assert np.all((net.k1 >= 1) & (net.k1 <= 100) & (net.d1 >= 1) & (net.d1 <= 100))
    assert np.all((net.k3 >= 100) & (net.k3 <= 200) & (net.d3 >= 100) & (net.d3 <= 200))
    # Planar Delaunay graph: L = 3N - 3 - (hull nodes); the paper's example has 30 nodes, 78 springs.
    assert 3 * 30 - 3 - 30 < net.n_springs <= 3 * 30 - 6
    np.testing.assert_allclose(net.l0, net.lengths(net.pos0))      # starts at rest


def test_rest_is_an_equilibrium_and_motion_decays():
    net = h.build(seed=0)
    n = 2000
    L = h.simulate(net, np.zeros(n))
    assert np.abs(L - net.l0).max() < 1e-12
    u = np.zeros(20000)
    u[:200] = 1.0                                                   # a push, then release
    L = h.simulate(net, u)
    dev = np.abs(L - net.l0).max(axis=1)
    # Fading memory (their footnote 15). Nearly collinear springs leave soft modes held
    # only by geometric and cubic stiffness, so the slowest decay takes tens of seconds.
    assert dev[300] > 1e-3 and dev[-1] < 0.1 * dev.max()
    assert np.all(np.diff(dev[5000::1000]) < 0)


def test_network_beats_linear_regression_on_the_volterra_task():
    """Their Fig. 9c: a linear readout of the spring lengths emulates the Volterra operator;
    linear regression on the raw input does not."""
    r = h.multitask(h.build(seed=1), washout=10.0, learn=20.0, test=5.0,
                    tasks=("volterra",))["volterra"]
    assert r["mse"] < 1e-2 * r["mse_lr"]


def test_registered_body_matches_the_direct_simulation():
    sim = make("hauser_network", seed=2, sigma_T2=0.0)
    net = sim.net
    rng = np.random.default_rng(0)
    n = 3000
    u = 0.5 * np.sin(2 * np.pi * 2.0 * np.arange(n) * h.DT)
    out, (q, v) = sim.simulate(n, h.DT, 1, u=u[:, None, None], rng=rng, v_init=np.zeros((1, sim.n)))
    P = np.repeat(net.pos0[None], 1, axis=0)
    P[:, ~net.fixed] = q.reshape(1, -1, 2)
    L_sim = net.lengths(P)[0]
    L_ref = h.simulate(net, u)[-1]
    np.testing.assert_allclose(L_sim, L_ref, atol=2e-4)
