"""Sampling engine: batched simulation and estimators for nonlinear bodies (thread T12).

A `MechSim` is H q'' + D q' = F(q, params) + J(q)^T u + xi, with thermal forces
E[xi xi^T] = 2 sigma_T^2 D delta: a constant mass matrix and damping, and any conservative
or nonlinear force F. Contact and hybrid bodies override `step`.

The default integrator is BAOAB (Leimkuhler and Matthews): half kick, half drift, an exact
Ornstein-Uhlenbeck step for the velocities under H^-1 D, half drift, half kick. Every array
carries a leading batch axis, so one call simulates many trajectories.

Estimators:
- `lmmse_bits`: for a Gaussian input, a lower bound on I(input; readout) from the error of
  the best linear estimate of the input. Exact for linear bodies.
- `path_fisher`: Fisher information about a drift parameter from the whole state path,
  through the Girsanov score. Exact in the limit of small steps, for any F.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import scipy.linalg as sla

from .linear import LinearBody

LOG2 = np.log(2.0)


@dataclass
class MechSim:
    """A simulated mechanism. `force(q, params) -> (batch, n)`; `J(q) -> (batch, n_u, n)`."""
    H: np.ndarray
    D: np.ndarray
    force: Callable[[np.ndarray, Mapping], np.ndarray]
    J: Callable[[np.ndarray], np.ndarray]
    sigma_T2: float = 1e-4
    params: dict = field(default_factory=dict)
    parts: Mapping[str, np.ndarray] = field(default_factory=dict)
    name: str = "sim"
    q0: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.H.shape[0]

    def readout(self, name: str, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        x = np.concatenate([q, v], axis=-1)
        C = np.eye(2 * self.n) if name == "whole" else np.atleast_2d(self.parts[name])
        return x @ C.T

    # -- integrator ---------------------------------------------------------------------
    def _ou(self, dt: float):
        Hi = np.linalg.inv(self.H)
        Gm = Hi @ self.D
        E = sla.expm(-Gm * dt)
        # Stationary velocity covariance sigma_T^2 H^-1; OU step keeps it exactly.
        S = self.sigma_T2 * (Hi - E @ Hi @ E.T)
        S = 0.5 * (S + S.T)
        lam, V = np.linalg.eigh(S)
        Lc = V * np.sqrt(np.clip(lam, 0.0, None))
        return Hi, E, Lc

    def step(self, q, v, u, rng, dt, cache):
        Hi, E, Lc = cache
        acc = lambda qq: (self.force(qq, self.params)
                          + np.einsum("bun,bu->bn", self.J(qq), u)) @ Hi.T
        v = v + 0.5 * dt * acc(q)
        q = q + 0.5 * dt * v
        v = v @ E.T + rng.standard_normal(v.shape) @ Lc.T
        q = q + 0.5 * dt * v
        v = v + 0.5 * dt * acc(q)
        return q, v

    def simulate(self, n_steps: int, dt: float, batch: int, u=None, rng=None,
                 parts=("whole",), q_init=None, v_init=None, burn_in: int = 0):
        """Simulate `batch` trajectories. `u` is None, an array (n_steps, batch, n_u), or a
        callable (k, q, v) -> (batch, n_u). Returns {part: (n_steps, batch, m)}, and the
        final (q, v)."""
        rng = np.random.default_rng() if rng is None else rng
        n = self.n
        q = (np.zeros((batch, n)) if self.q0 is None else np.tile(self.q0, (batch, 1))) \
            if q_init is None else q_init.copy()
        v = (rng.standard_normal((batch, n)) @ np.linalg.cholesky(
            self.sigma_T2 * np.linalg.inv(self.H)).T) if v_init is None else v_init.copy()
        cache = self._ou(dt)
        n_u = self.J(q).shape[1]
        zero = np.zeros((batch, n_u))
        for _ in range(burn_in):
            q, v = self.step(q, v, zero, rng, dt, cache)
        out = {p: np.empty((n_steps, batch, self.readout(p, q, v).shape[-1])) for p in parts}
        for k in range(n_steps):
            for p in parts:
                out[p][k] = self.readout(p, q, v)
            uk = zero if u is None else (u(k, q, v) if callable(u) else u[k])
            q, v = self.step(q, v, uk, rng, dt, cache)
        return out, (q, v)


def from_linear(body: LinearBody) -> MechSim:
    """Wrap a linear body as a simulation (for validating estimators)."""
    K = body.K.copy()
    Js = {c: np.atleast_2d(J) for c, J in body.inputs.items()}
    J0 = next(iter(Js.values()))
    return MechSim(H=body.H, D=body.D, force=lambda q, p: -q @ K.T,
                   J=lambda q: np.broadcast_to(J0, (q.shape[0],) + J0.shape),
                   sigma_T2=body.sigma_T2, parts=dict(body.parts), name=body.name + " (sim)")


def gibbs_start(sim: MechSim, batch: int, rng, K: np.ndarray | None = None):
    """Exact Gibbs draw for a linear body with stiffness K, else positions at q0."""
    n = sim.n
    v = rng.standard_normal((batch, n)) @ np.linalg.cholesky(sim.sigma_T2 * np.linalg.inv(sim.H)).T
    if K is None:
        q = np.zeros((batch, n)) if sim.q0 is None else np.tile(sim.q0, (batch, 1))
    else:
        q = rng.standard_normal((batch, n)) @ np.linalg.cholesky(sim.sigma_T2 * np.linalg.inv(K)).T
    return q, v


# ---------------------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------------------

def lmmse_bits(U: np.ndarray, Y: np.ndarray, ridge: float = 0.0, unbiased: bool = True) -> float:
    """Lower bound on I(U; Y) in bits for Gaussian U: 1/2 log2 det Cov(U) / det Cov(U - U_hat),
    where U_hat is the best linear estimate of U from Y. Rows are samples.

    The bound holds for any body, linear or not, because the Gaussian has the largest
    entropy at a given error covariance. It is exact when U and Y are jointly Gaussian.
    """
    U = U - U.mean(0)
    Y = Y - Y.mean(0)
    Suu = U.T @ U / len(U)
    Syy = Y.T @ Y / len(Y) + ridge * np.eye(Y.shape[1])
    Suy = U.T @ Y / len(U)
    Serr = Suu - Suy @ np.linalg.solve(Syy, Suy.T)
    if unbiased:                       # residual covariance of a regression on p features
        N, p = Y.shape
        Serr = Serr * N / (N - p - 1)
    return 0.5 * (np.linalg.slogdet(Suu)[1] - np.linalg.slogdet(Serr)[1]) / LOG2


def path_fisher(sim: MechSim, param: str, n_steps: int, dt: float, batch: int,
                rel_step: float = 1e-4, rng=None, log: bool = True, burn_in: int = 0):
    """Fisher information about a force parameter from the whole state path (Girsanov).

    With the state observed, the score is sum_k dF^T (2 sigma_T^2 D)^+ (H dv_k - (F - D v) dt),
    using the velocity increments of a semi-implicit Euler step (velocity first, then
    position), for which this score is exact. Plain Euler-Maruyama pumps energy into
    oscillators and biases the result. Returns the Monte Carlo mean of
    the squared score and its standard error. dF must lie in the range of D.
    """
    rng = np.random.default_rng() if rng is None else rng
    theta = sim.params[param]
    h = rel_step if log else rel_step * max(abs(theta), 1e-12)

    def F_at(q, val):
        return sim.force(q, {**sim.params, param: val})

    up = theta * np.exp(h) if log else theta + h
    dn = theta * np.exp(-h) if log else theta - h
    Dp = np.linalg.pinv(2.0 * sim.sigma_T2 * sim.D)
    Hm, Dm = sim.H, sim.D
    q, v = gibbs_start(sim, batch, rng)
    lamD, VD = np.linalg.eigh(2.0 * sim.sigma_T2 * Dm * dt)
    LD = VD * np.sqrt(np.clip(lamD, 0.0, None))
    Hi = np.linalg.inv(Hm)
    score = np.zeros(batch)
    for k in range(burn_in + n_steps):
        F = sim.force(q, sim.params)
        xi = rng.standard_normal(v.shape) @ LD.T             # impulse of the thermal force
        dv = ((F - v @ Dm.T) * dt + xi) @ Hi.T
        if k >= burn_in:
            dF = (F_at(q, up) - F_at(q, dn)) / (2 * h)
            score += np.einsum("bi,ij,bj->b", dF, Dp, dv @ Hm.T - (F - v @ Dm.T) * dt)
        v = v + dv
        q = q + v * dt                 # semi-implicit Euler: no energy drift for oscillators
    fi = score ** 2
    return float(fi.mean()), float(fi.std(ddof=1) / np.sqrt(batch))
