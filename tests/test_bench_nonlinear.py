"""Nonlinear benchmark bodies of mibook.bench (thread T12)."""
import numpy as np
import pytest

from mibook import bench
from mibook.bench.bodies.nonlinear import mesostate


def _stiffness(sim, eps=1e-7):
    n = sim.n
    K = np.zeros((n, n))
    for j in range(n):
        dq = np.zeros((1, n))
        dq[0, j] = eps
        K[:, j] = -(sim.force(dq, sim.params) - sim.force(-dq, sim.params))[0] / (2 * eps)
    return K


def test_soft_arm_is_a_stable_truss():
    arm = bench.make("soft_arm")
    K = _stiffness(arm)
    np.testing.assert_allclose(K, K.T, atol=1e-6 * np.abs(K).max())
    assert np.linalg.eigvalsh(0.5 * (K + K.T)).min() > 0          # bending stiffness exists


def test_soft_arm_stiffens():
    arm = bench.make("soft_arm")
    n = arm.n
    q = np.zeros((1, n))
    q[0, -1] = 0.01                                               # stretch the tip upward
    soft = bench.make("soft_arm", alpha=0.0)
    assert np.abs(arm.force(q, arm.params)).sum() > np.abs(soft.force(q, soft.params)).sum()


def test_soft_arm_thermal_velocities():
    # Start from the Gibbs law of the linearized arm, so equipartition already holds.
    from mibook.bench.sampling import gibbs_start
    arm = bench.make("soft_arm")
    rng = np.random.default_rng(0)
    q0, v0 = gibbs_start(arm, 400, rng, K=_stiffness(arm))
    out, (q, v) = arm.simulate(300, 2e-4, 400, rng=rng, parts=(), q_init=q0, v_init=v0)
    assert np.all(np.isfinite(q))
    assert v.var() == pytest.approx(arm.sigma_T2 / 0.005, rel=0.1)


def test_bistable_chain_holds_and_switches():
    ch = bench.make("bistable_chain")
    rng = np.random.default_rng(1)
    batch, dt = 200, 0.01
    cache = ch._ou(dt)
    # Without drive, every element stays in its starting well (barriers 6-10 sigma_T^2).
    q = np.tile(ch.q0, (batch, 1))
    v = np.zeros((batch, ch.n))
    a = None
    zero = np.zeros((batch, 1))
    for _ in range(500):
        q, v, a = ch.step(q, v, zero, rng, dt, cache, a)
    assert mesostate(ch, q).mean() < 0.01
    # A slow ramp to 1.3 times the largest critical force switches every element, and the
    # element with the lowest critical force tends to go first.
    Fc = ch.critical_forces
    ramp = np.linspace(0.0, 1.3 * Fc.max(), 2500)
    first = np.full(batch, -1)
    for F in ramp:
        q, v, _ = ch.step(q, v, np.full((batch, 1), F), rng, dt, cache)
        st = mesostate(ch, q)
        hit = (first < 0) & st.any(axis=1)
        first[hit] = np.argmax(st[hit], axis=1)
    assert mesostate(ch, q).all(axis=1).mean() > 0.99
    assert np.mean(first == np.argmin(Fc)) > 0.4
