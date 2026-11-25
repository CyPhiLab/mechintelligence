"""Item 5 of the 2026-11-25 priorities: nonlinear comparisons (threads T7, T12).

1. Soft arm: what a linear readout of a snapshot can reconstruct about the past input,
   split by degree as in information processing capacity (Dambre et al. 2012): degree 1
   (linear memory capacity) and degree 2 (squares and products of past inputs). A linear
   body has no degree-2 capacity beyond what noise and geometry give; stiffening springs
   create it.
2. Bistable chain: hysterons at finite temperature. Each trajectory sees a ramp up to a
   random peak force and back to zero, then waits. The mesostate (which elements have
   switched) records the peak. The study measures how many bits it holds, how they decay
   while waiting, and the work the drive does.
"""
from __future__ import annotations

import numpy as np

from . import make
from .bodies.nonlinear import mesostate
from .sampling import gibbs_start


# ---------------------------------------------------------------------------------------
# Soft arm: capacities by degree
# ---------------------------------------------------------------------------------------

def _stiffness(sim, eps=1e-7):
    n = sim.n
    K = np.zeros((n, n))
    for j in range(n):
        dq = np.zeros((1, n))
        dq[0, j] = eps
        K[:, j] = -(sim.force(dq, sim.params) - sim.force(-dq, sim.params))[0] / (2 * eps)
    return 0.5 * (K + K.T)


def _capacity(X, y, train):
    """Held-out R^2 of the best linear readout of y from X (rows are samples)."""
    Xt, yt = X[train], y[train]
    Xv, yv = X[~train], y[~train]
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-30
    Xt = (Xt - mu) / sd
    Xv = (Xv - mu) / sd
    A = np.hstack([Xt, np.ones((len(Xt), 1))])
    w, *_ = np.linalg.lstsq(A, yt, rcond=None)
    pred = np.hstack([Xv, np.ones((len(Xv), 1))]) @ w
    c = 1.0 - np.mean((yv - pred) ** 2) / np.var(yv)
    return max(c, 0.0)


def arm_capacities(alpha: float = 1e3, amp: float = 0.05, batch: int = 3000,
                   period: float = 0.04, history: int = 40, dt: float = 4e-4,
                   max_lag1: int = 30, max_lag2: int = 12, seed: int = 0,
                   b_spring: float = 0.1) -> dict:
    """Uniform i.i.d. tip-force samples u in [-amp, amp], each held for `period`; the readout
    is a snapshot of every node's position after `history` samples. Returns the degree-1 and
    degree-2 capacities (sums of held-out R^2 over lags and lag pairs)."""
    arm = make("soft_arm", alpha=alpha, b_spring=b_spring)
    rng = np.random.default_rng(seed)
    sub = int(round(period / dt))
    U = rng.uniform(-1.0, 1.0, (history, batch))
    u = np.repeat(U, sub, axis=0)[:, :, None] * amp
    q0, v0 = gibbs_start(arm, batch, rng, K=_stiffness(arm))
    out, (q, v) = arm.simulate(history * sub, dt, batch, u=u, rng=rng, parts=(), q_init=q0,
                               v_init=v0)
    X = q                                              # all node positions, (batch, 2N)
    train = rng.uniform(size=batch) < 0.7
    lag = lambda tau: U[history - 1 - tau]             # input tau samples before the snapshot
    c1 = [_capacity(X, lag(t), train) for t in range(max_lag1)]
    P2 = lambda x: 0.5 * (3 * x ** 2 - 1)
    c2_sq = [_capacity(X, P2(lag(t)), train) for t in range(max_lag2)]
    c2_x = [_capacity(X, lag(t) * lag(s), train)
            for t in range(max_lag2) for s in range(t + 1, max_lag2)]
    # A shuffled target gives the estimator's floor (chance R^2 from finite samples).
    floor = np.mean([_capacity(X, rng.permutation(lag(0)), train) for _ in range(5)])
    return dict(alpha=alpha, amp=amp, deg1=float(np.sum(c1)), deg2=float(np.sum(c2_sq) + np.sum(c2_x)),
                c1=np.array(c1), c2_sq=np.array(c2_sq), floor=float(floor),
                n_targets2=len(c2_sq) + len(c2_x), dims=X.shape[1])


def arm_sweep(alphas=(0.0, 1e3, 1e4, 3e4), amps=(0.02, 0.1), **kw) -> list[dict]:
    return [arm_capacities(alpha=a, amp=A, **kw) for a in alphas for A in amps]


# ---------------------------------------------------------------------------------------
# Bistable chain: bits about the peak force, their lifetime, and the work
# ---------------------------------------------------------------------------------------

def _plugin_mi(x_bins: np.ndarray, states: np.ndarray, n_x: int) -> float:
    """Plug-in mutual information (bits) between a binned variable and a discrete state,
    with the Miller-Madow correction for both entropies."""
    keys = states @ (1 << np.arange(states.shape[1]))
    _, s_idx = np.unique(keys, return_inverse=True)
    n_s = s_idx.max() + 1
    joint = np.zeros((n_x, n_s))
    np.add.at(joint, (x_bins, s_idx), 1)
    N = joint.sum()

    def H(p_counts):
        p = p_counts[p_counts > 0] / N
        return -np.sum(p * np.log2(p)) + (np.count_nonzero(p_counts) - 1) / (2 * N * np.log(2))

    return float(H(joint.sum(1)) + H(joint.sum(0)) - H(joint.ravel()))


def chain_memory(barrier: float = 8.0, waits=(0.0, 50.0, 200.0, 800.0), batch: int = 4000,
                 ramp_time: float = 25.0, dt: float = 0.01, n_bins: int = 14, seed: int = 0,
                 gamma: float = 2.0) -> dict:
    """Random peak forces F_max ~ U[0, 1.3 max F_c]; ramp up and down in `ramp_time` each,
    then wait at zero force. Returns I(F_max; mesostate) after each wait, the ideal value
    (no thermal noise: the mesostate counts thresholds below F_max), and the mean work."""
    ch = make("bistable_chain", barrier=barrier, gamma=gamma, seed=seed)
    rng = np.random.default_rng(seed + 1)
    Fc = ch.critical_forces
    Fmax = rng.uniform(0.0, 1.3 * Fc.max(), batch)
    bins = np.minimum((Fmax / (1.3 * Fc.max()) * n_bins).astype(int), n_bins - 1)
    n_r = int(ramp_time / dt)
    profile = np.concatenate([np.linspace(0, 1, n_r), np.linspace(1, 0, n_r)])
    cache = ch._ou(dt)
    q = np.tile(ch.q0, (batch, 1))
    v = np.zeros((batch, ch.n))
    work = np.zeros(batch)
    x_end_prev = q[:, -1].copy()
    for c in profile:
        F = (c * Fmax)[:, None]
        q, v, _ = ch.step(q, v, F, rng, dt, cache)
        work += F[:, 0] * (q[:, -1] - x_end_prev)       # drive force times end displacement
        x_end_prev = q[:, -1].copy()
    out = {}
    zero = np.zeros((batch, 1))
    t_done = 0.0
    for w in sorted(waits):
        for _ in range(int((w - t_done) / dt)):
            q, v, _ = ch.step(q, v, zero, rng, dt, cache)
        t_done = w
        out[w] = _plugin_mi(bins, mesostate(ch, q).astype(int), n_bins)
    ideal_states = (Fmax[:, None] > Fc[None, :]).astype(int)
    ideal = _plugin_mi(bins, ideal_states, n_bins)
    return dict(barrier=barrier, info=out, ideal=ideal, work=float(work.mean()),
                sigma_T2=ch.sigma_T2, levels=ch.n + 1)


def summary(alphas=(0.0, 1e3, 1e4, 3e4), amps=(0.02, 0.1), barriers=(6.0, 8.0, 12.0),
            arm_batch: int = 2000, chain_batch: int = 3000) -> dict:
    """The sweeps reported in the research note (a few minutes)."""
    arm = [arm_capacities(alpha=a, amp=A, batch=arm_batch, history=30)
           for a in alphas for A in amps]
    chain = [chain_memory(barrier=b, batch=chain_batch) for b in barriers]
    return dict(arm=arm, chain=chain)
