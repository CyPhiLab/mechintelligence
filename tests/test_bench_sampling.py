"""The sampling engine of mibook.bench, validated against the exact engine (thread T12)."""
import numpy as np
import pytest

from mibook.bench import linear as L, sampling as S
from test_bench_linear import sdof


def test_baoab_keeps_the_gibbs_law():
    b = sdof()
    sim = S.from_linear(b)
    rng = np.random.default_rng(0)
    q0, v0 = S.gibbs_start(sim, 4000, rng, b.K)
    out, _ = sim.simulate(1000, 1e-3, 4000, rng=rng, q_init=q0, v_init=v0)
    x = out["whole"][-1]
    assert x[:, 0].var() == pytest.approx(b.sigma_T2 / 400.0, rel=0.08)
    assert x[:, 1].var() == pytest.approx(b.sigma_T2 / 1.0, rel=0.08)


def test_lmmse_matches_the_exact_engine_on_a_linear_body():
    b = sdof()
    sim = S.from_linear(b)
    fs, T, band, sa, sub, batch = 200.0, 0.5, 10.0, 0.5, 10, 8000
    n_t, k = int(round(T * fs)) + 1, int(round(2 * band * T))
    Phi = L.dct_basis(n_t, k)
    rng = np.random.default_rng(1)
    W = rng.standard_normal((batch, k)) * sa
    u = np.repeat((W @ Phi.T).T, sub, axis=0)[:, :, None]
    q0, v0 = S.gibbs_start(sim, batch, rng, b.K)
    out, _ = sim.simulate(n_t * sub, 1 / fs / sub, batch, u=u, rng=rng, parts=("x",),
                          q_init=q0, v_init=v0)
    est = S.lmmse_bits(W, out["x"][::sub, :, 0].T)
    exact = L.profile_bits(b, "x", t_window=T, fs=fs, band_hz=band, sigma_a=sa, eps=1e-9)
    assert est == pytest.approx(exact, abs=0.15)


def test_path_fisher_matches_the_stiffness_rate():
    # Whole-body rate about log k is k/(2b) per second (ground-stiffness note, eps = 1).
    b = sdof()
    sim = S.from_linear(b)
    sim.params = {"k": 400.0}
    sim.force = lambda q, p: -p["k"] * q
    fi, se = S.path_fisher(sim, "k", n_steps=1000, dt=1e-3, batch=3000, burn_in=3000,
                           rng=np.random.default_rng(2))
    assert abs(fi - 400.0 / (2 * 2.0) * 1.0) < 4 * se
