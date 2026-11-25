"""The exact linear engine of mibook.bench (thread T12)."""
import numpy as np
import pytest

from mibook import bench, checks
from mibook.bench import linear as L
from passive_info.systems import proprioceptive, series_elastic


def sdof(m=1.0, b=2.0, k=400.0, sigma_T2=1e-4):
    def build(m=m, b=b, k=k, sigma_T2=sigma_T2):
        return L.LinearBody(H=np.array([[m]]), D=np.array([[b]]), K=np.array([[k]]),
                            inputs={"force": np.array([[1.0]])},
                            parts={"x": np.array([[1.0, 0.0]])}, sigma_T2=sigma_T2,
                            params=dict(m=m, b=b, k=k, sigma_T2=sigma_T2), build=build)
    return build()


def test_registry_lists_the_linear_tier():
    names = {e.name for e in bench.available()}
    assert {"rigid", "sea", "network", "whisker"} <= names


@pytest.mark.parametrize("ks", [None, 70.0, 1e4])
def test_legs_reproduce_the_book(ks):
    b = bench.make("rigid") if ks is None else bench.make("sea", ks=ks)
    s = proprioceptive(readout="energy") if ks is None else series_elastic(ks, readout="energy")
    whole = checks.langevin_bits(s, np.eye(2 * s.n), 0.0)
    rotor = checks.langevin_bits(s, b.part("rotor"), 1e-9)
    assert L.profile_bits(b) == pytest.approx(whole, rel=1e-10)
    assert L.profile_bits(b, "rotor", eps=1e-9) == pytest.approx(rotor, rel=1e-8)


def test_thermal_stationary_law_is_gibbs():
    b = bench.make("sea", ks=1e3)
    P = b.stationary_cov()
    Pg = b.sigma_T2 * np.linalg.inv(b.energy_metric())
    np.testing.assert_allclose(P, Pg, rtol=1e-8, atol=1e-10 * np.abs(Pg).max())


def test_noise_integral_matches_the_stationary_difference():
    b = bench.make("sea", ks=1e3)
    s = b.discretize(1 / 4000.0)
    np.testing.assert_allclose(s.Qd, s.P - s.Ad @ s.P @ s.Ad.T, rtol=1e-6, atol=1e-14)


def test_whitening_equals_inverse_cholesky():
    b = bench.make("sea", ks=1e3)
    s = b.discretize(1 / 4000.0, "force")
    C = b.part("rotor")
    n_t = 30
    cov = checks.langevin_noise_cov(series_elastic(1e3, readout="energy"), C, n_t, 1e-6)
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(n_t, 1, 3))
    Z = L.Filter.build(s, C, n_t, eps=1e-6).whiten(Y)
    Lc = np.linalg.cholesky(cov)
    np.testing.assert_allclose(Z.T @ Z, (np.linalg.solve(Lc, Y.reshape(n_t, 3))).T
                               @ np.linalg.solve(Lc, Y.reshape(n_t, 3)), rtol=1e-6)


def test_white_input_matches_a_direct_computation():
    b = sdof()
    n_t, q, fs = 40, 0.3, 1000.0
    s = b.discretize(1 / fs, "force")
    C = b.part("x")
    # Direct: Y = M u + noise, noise covariance from the stationary Gauss-Markov law.
    Phi = np.eye(n_t)
    M = L.responses(s, C, Phi[:, None, :]).reshape(n_t, n_t)
    cov = np.empty((n_t, n_t))
    Ak = [np.eye(2)]
    for _ in range(n_t):
        Ak.append(s.Ad @ Ak[-1])
    for i in range(n_t):
        for j in range(n_t):
            k, l = max(i, j), min(i, j)
            cov[i, j] = (C @ Ak[k - l] @ s.P @ C.T).item()
    cov += 1e-12 * np.eye(n_t)
    direct = 0.5 * np.linalg.slogdet(np.eye(n_t) + q * np.linalg.solve(cov, M @ M.T))[1] / np.log(2)
    assert L.white_input_bits(b, "x", q=q, n_t=n_t, fs=fs, eps=1e-6) == pytest.approx(direct, rel=1e-6)


def test_spectral_rate_single_mass_closed_form():
    # SNR = level / (2 sigma_T^2 b) at every frequency, for position readout.
    b = sdof()
    level, B = 1e-4, 50.0
    expect = B * np.log2(1 + level / (2 * b.sigma_T2 * 2.0))
    assert L.input_rate(b, "x", band_hz=B, level=level) == pytest.approx(expect, rel=1e-6)


def test_spectral_rate_is_the_same_for_every_leg():
    # The whole body sees the force through J D^-1 J^T alone (Prop. "only the dissipation
    # at the contact matters"), in the frequency domain too.
    rates = [L.input_rate(bench.make(n, **kw), band_hz=100.0, level=1.0)
             for n, kw in (("rigid", {}), ("sea", {"ks": 70.0}), ("sea", {"ks": 1e4}))]
    assert max(rates) - min(rates) < 1e-6 * max(rates)


def test_one_step_empowerment_is_the_sampling_gain():
    b = bench.make("sea", ks=1e3)
    s = b.discretize(1 / 4000.0, "force")
    g = (s.Bd.T @ np.linalg.solve(s.Qd, s.Bd)).item()
    P = 1e3
    assert L.empowerment(b, "force", horizon=1, power=P) == pytest.approx(0.5 * np.log2(1 + P * g), rel=1e-8)
    assert L.empowerment(b, "force", horizon=20, power=P) >= L.empowerment(b, "force", horizon=1, power=P)


def test_memory_capacity_matches_the_bridge_check():
    from mibook import bridges
    r = bridges.check_memory_capacity()
    m = L.memory_capacity(bench.make("sea", ks=1e3), q=1e4)
    assert m["mc"] == pytest.approx(r["mc_sum"], rel=1e-8)
    assert m["nats"] == pytest.approx(r["info_spectrum"], rel=1e-8)


def test_mcw_matches_the_book():
    from passive_info.channel import morphological_bits
    b = bench.make("sea", ks=1e4)
    assert L.mc_w(b) == pytest.approx(morphological_bits(series_elastic(1e4, readout="energy")), rel=1e-6)


def test_contact_point_needs_an_input():
    w = bench.make("whisker")
    assert L.fisher_rate(w, "contact", "follicle", log=False) == pytest.approx(0.0, abs=1e-6)
    assert L.fisher_rate(w, "contact", "follicle", log=False, channel="force", level=1e-8) > 0


def test_network_is_stable_and_informative():
    net = bench.make("network", n_nodes=12, seed=1)
    assert np.linalg.eigvals(net.A).real.max() < 0
    whole = L.profile_bits(net, sigma_a=1.0)
    node = L.profile_bits(net, "node0", sigma_a=1.0, eps=1e-12)
    assert 0 < node <= whole
