"""General handling of unstable bodies (thread T12)."""
import numpy as np
import pytest

from mibook import bench
from mibook.bench import linear as L, sampling as S


def test_mirror_stabilization_reflects_unstable_poles_only():
    b = bench.make("pendulum", k=5.0)
    ev = np.sort(np.linalg.eigvals(b.A).real)
    cl = L.mirror_stabilize(b, "motor")
    evc = np.sort(np.linalg.eigvals(cl.A).real)
    np.testing.assert_allclose(evc, np.sort(-np.abs(ev)), rtol=1e-8)
    assert cl.min_data_rate == pytest.approx(ev.max() / np.log(2), rel=1e-9)
    stable = L.mirror_stabilize(bench.make("pendulum", k=20.0), "motor")
    assert np.abs(stable.gain).max() < 1e-9 and stable.control_power < 1e-12


def test_unstable_body_needs_a_prior():
    b = bench.make("pendulum", k=5.0)
    with pytest.raises(ValueError):
        L.white_input_bits(b, "angle", q=1.0, n_t=50, fs=200.0, eps=1e-9)


def test_steady_rate_matches_windows_for_stable_and_unstable_bodies():
    sea = bench.make("sea", ks=1e3)
    a = L.white_input_bits(sea, "rotor", q=1e4, n_t=200, eps=1e-9)
    c = L.white_input_bits(sea, "rotor", q=1e4, n_t=400, eps=1e-9)
    assert L.steady_white_rate(sea, "rotor", q=1e4) == pytest.approx((c - a) / 200, rel=1e-4)
    p = bench.make("pendulum", k=5.0)
    w = [L.white_input_bits(p, "angle", q=1.0, n_t=n, fs=200.0, eps=1e-9, prior=L.prior_rest(p))
         for n in (200, 400)]
    assert L.steady_white_rate(p, "angle", q=1.0, fs=200.0) == pytest.approx((w[1] - w[0]) / 200, rel=1e-6)


def test_instability_does_not_change_what_the_body_holds_about_a_push():
    rates = [L.steady_white_rate(bench.make("pendulum", k=k), "angle", q=1.0, fs=200.0)
             for k in (20.0, 10.5, 9.0, 0.0)]
    # Equal up to higher-order terms in the sampling step (exact as dt -> 0).
    assert max(rates) - min(rates) < 1e-4 * max(rates)


def test_nonlinear_body_is_held_up_by_mirror_feedback():
    m, l, g, k, b = 1.0, 1.0, 9.81, 5.0, 0.5
    sim = S.MechSim(H=np.array([[m * l * l]]), D=np.array([[b]]),
                    force=lambda q, p: m * g * l * np.sin(q) - k * q,
                    J=lambda q: np.ones((q.shape[0], 1, 1)), sigma_T2=1e-2, params={})
    u, cl = S.mirror_feedback(sim, np.array([[1.0]]))
    rng = np.random.default_rng(0)
    z = np.zeros((100, 1))
    _, (q, _) = sim.simulate(10000, 1e-3, 100, rng=rng, parts=(), q_init=z, v_init=z)
    assert (np.abs(q) > 1).mean() > 0.9                                    # falls
    out, _ = sim.simulate(15000, 1e-3, 100, u=u, rng=rng, parts=("whole",), q_init=z, v_init=z)
    assert out["whole"][5000:, :, 0].var() == pytest.approx(cl.stationary_cov()[0, 0], rel=0.1)
