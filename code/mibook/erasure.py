"""Erasing a mechanical bit, and the cost of a write-read-erase cycle (thread T7).

The element of `mibook.bistable` in the book's units (lengths a, energies sigma_T^2, times
sqrt(m a^2 / sigma_T^2)), with the potential written as

    U(x, t) = beta x^4 - a2(t) x^2 - f(t) x,

which is the double well beta ((x^2 - 1)^2 - 1) when a2 = 2 beta and f = 0. The erasure resets the
bit to the left well whatever its value, in four stages of equal length tau (as in the experiment
of Berut et al. 2012):

    1. lower the barrier:  a2 from 2 beta to 0;
    2. tilt to the left:   f from 0 to -f_tilt;
    3. raise the barrier:  a2 from 0 to 2 beta;
    4. remove the tilt:    f from -f_tilt to 0.

The work is the energy put in by changing the potential, W = sum of U(x, t + dt) - U(x, t) at the
current x (Sekimoto's definition), averaged over elements whose starting well is random.
"""
from __future__ import annotations

import numpy as np

from mibook.bistable import GAMMA

BETA = 8.0
F_TILT = 5.0              #: tilt, in sigma_T^2 / a: the right well ends 2 F_TILT above the left


def schedule(t, tau: float, beta: float = BETA, f_tilt: float = F_TILT):
    """(a2, f) at time t of the four-stage protocol, linear within each stage."""
    s = np.clip(t / tau, 0.0, 4.0)
    a2 = np.where(s < 1, 2 * beta * (1 - s), np.where(s < 2, 0.0,
                  np.where(s < 3, 2 * beta * (s - 2), 2 * beta)))
    f = np.where(s < 1, 0.0, np.where(s < 2, -f_tilt * (s - 1),
                 np.where(s < 3, -f_tilt, -f_tilt * (4 - s))))
    return float(a2), float(f)


def potential(x, a2, f, beta: float = BETA):
    return beta * x ** 4 - a2 * x ** 2 - f * x


def erase(tau: float, beta: float = BETA, f_tilt: float = F_TILT, gamma: float = GAMMA,
          n: int = 2000, dt: float = 0.005, seed: int = 0) -> dict:
    """Run the erasure on n elements that start in a random well (thermalised there).

    Returns the mean and spread of the work, and the fraction that end in the left well."""
    rng = np.random.default_rng(seed)
    c = np.exp(-gamma * dt)
    s = np.sqrt(1.0 - c * c)
    x = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    v = rng.normal(0.0, 1.0, n)

    def F(x, a2, f):
        return -(4 * beta * x ** 3 - 2 * a2 * x - f)

    # thermalise inside the starting wells at the full barrier
    a2, f = 2 * beta, 0.0
    Fx = F(x, a2, f)
    for _ in range(int(round(4.0 / dt))):
        v += 0.5 * dt * Fx
        x += 0.5 * dt * v
        v = c * v + s * rng.normal(0.0, 1.0, n)
        x += 0.5 * dt * v
        Fx = F(x, a2, f)
        v += 0.5 * dt * Fx
    W = np.zeros(n)
    steps = int(round(4 * tau / dt))
    for k in range(steps):
        # change the potential at fixed x (work), then move for dt (heat)
        a2n, fn = schedule((k + 1) * dt, tau, beta, f_tilt)
        W += potential(x, a2n, fn, beta) - potential(x, a2, f, beta)
        a2, f = a2n, fn
        Fx = F(x, a2, f)
        v += 0.5 * dt * Fx
        x += 0.5 * dt * v
        v = c * v + s * rng.normal(0.0, 1.0, n)
        x += 0.5 * dt * v
        Fx = F(x, a2, f)
        v += 0.5 * dt * Fx
    return dict(tau=tau, work=float(W.mean()), work_sd=float(W.std()),
                work_err=float(W.std() / np.sqrt(n)), left=float(np.mean(x < 0.0)))


def landauer_bound(p_error: float) -> float:
    """Least mean work to reset a random bit to one well with error p (in sigma_T^2):
    ln 2 - H(p), with H the entropy in nats; ln 2 for a perfect reset."""
    p = np.clip(p_error, 1e-15, 0.5)
    return float(np.log(2.0) + p * np.log(p) + (1 - p) * np.log(1 - p))


def check_erasure(taus=(5.0, 10.0, 20.0, 50.0, 100.0, 200.0), n: int = 2000,
                  seed: int = 0) -> dict:
    """Work and success of the erasure against the protocol's duration, and a fit
    W = W_inf + B / tau to the slower half."""
    rows = [erase(t, n=n, seed=seed) for t in taus]
    tau = np.array([r["tau"] for r in rows])
    W = np.array([r["work"] for r in rows])
    err = np.array([r["work_err"] for r in rows])
    keep = tau >= np.median(tau)
    A = np.vstack([np.ones(keep.sum()), 1.0 / tau[keep]]).T
    w = 1.0 / err[keep] ** 2
    coef = np.linalg.solve(A.T @ (A * w[:, None]), A.T @ (w * W[keep]))
    left = np.array([r["left"] for r in rows])
    return dict(tau=tau, work=W, work_err=err, left=left,
                bound_at_error=np.array([landauer_bound(1.0 - q) for q in left]),
                w_inf=float(coef[0]), b=float(coef[1]), bound=landauer_bound(0.0))
