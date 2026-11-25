"""Nonlinear, smooth benchmark bodies (sampling engine).

- `soft_arm`: a planar arm of point masses joined by stiffening springs, with second-
  neighbour springs for bending: the physical-reservoir body of soft robotics (Nakajima et
  al. 2013), reduced to a spring network.
- `bistable_chain`: bistable elements in series, driven at one end: hysterons at finite
  temperature (Bense and van Hecke 2021), each element a double well as in @sec-memory.
"""
from __future__ import annotations

import numpy as np

from .. import register
from ..sampling import MechSim


def _spring_forces(X, F, pairs, rest, k, alpha):
    """Add the forces of stiffening springs f = -k s (1 + alpha s^2), s = stretch, to F.

    X and F have shape (N, 2, batch): the batch axis is last, so gathering a node is a
    contiguous slab, which is several times faster than batch-first indexing.
    """
    i, j = pairs[:, 0], pairs[:, 1]
    d = X[j] - X[i]
    ln = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    rest = np.asarray(rest, float)
    s = ln - (rest[:, None] if rest.ndim == 1 else rest)
    f = (k * s * (1.0 + alpha * s * s) / np.maximum(ln, 1e-12))[:, None, :] * d
    F[i] += f                          # indices are unique within a spring set (checked at
    F[j] -= f                          # construction), so plain indexing accumulates exactly


@register("soft_arm", "2-nonlinear",
          "Planar soft arm: a ladder truss of point masses with stiffening springs.")
def soft_arm(n_seg: int = 10, length: float = 0.3, width: float = 0.03, mass: float = 0.005,
             k_rail: float = 200.0, k_rung: float = 200.0, k_diag: float = 100.0,
             alpha: float = 1e3, b_spring: float = 0.02, sigma_T2: float = 1e-6) -> MechSim:
    """Two rails of N nodes each (left L_i, right R_i, i = 1..N) above a fixed base pair,
    joined by rail, rung and diagonal springs: a triangulated truss, so the arm has linear
    bending stiffness. Springs stiffen as k s (1 + alpha s^2). Each spring has a damper along
    its rest direction (the damping matrix is fixed at rest, as the sampling engine needs).
    Input: a transverse (x) force at the tip, shared by the two tip nodes."""
    N = n_seg
    h = length / N
    # Node index: L_i -> 2(i-1), R_i -> 2(i-1)+1, for i = 1..N; base nodes are fixed.
    rest = np.zeros((2 * N, 2))
    for i in range(1, N + 1):
        rest[2 * (i - 1)] = (-width / 2, h * i)
        rest[2 * (i - 1) + 1] = (width / 2, h * i)
    L = lambda i: 2 * (i - 1)
    R = lambda i: 2 * (i - 1) + 1
    sets = {
        "railL": np.array([[L(i), L(i + 1)] for i in range(1, N)]),
        "railR": np.array([[R(i), R(i + 1)] for i in range(1, N)]),
        "rung": np.array([[L(i), R(i)] for i in range(1, N + 1)]),
        "diag": np.array([[L(i), R(i + 1)] for i in range(1, N)]),
    }
    kset = {"railL": k_rail, "railR": k_rail, "rung": k_rung, "diag": k_diag}
    for P in sets.values():
        assert len(set(P[:, 0])) == len(P) and len(set(P[:, 1])) == len(P)
    restlen = {k: np.linalg.norm(rest[P[:, 1]] - rest[P[:, 0]], axis=1) for k, P in sets.items()}
    # Springs from the fixed base pair (at height 0) to level 1: rails and a diagonal.
    base = np.array([[-width / 2, 0.0], [width / 2, 0.0]])
    anchors = [(L(1), base[0], k_rail), (R(1), base[1], k_rail), (R(1), base[0], k_diag)]
    nn = 2 * N
    n = 2 * nn
    H = mass * np.eye(n)
    D = np.zeros((n, n))
    for name, P in sets.items():
        for (i, j) in P:
            u = (rest[j] - rest[i]) / np.linalg.norm(rest[j] - rest[i])
            a = np.zeros(n)
            a[2 * i:2 * i + 2], a[2 * j:2 * j + 2] = -u, u
            D += b_spring * np.outer(a, a)
    for (i, p0, _) in anchors:
        u = (rest[i] - p0) / np.linalg.norm(rest[i] - p0)
        a = np.zeros(n)
        a[2 * i:2 * i + 2] = u
        D += b_spring * np.outer(a, a)
    D += 1e-3 * b_spring * np.eye(n)          # a trace of damping on every coordinate
    params = dict(alpha=alpha, k_rail=k_rail, k_rung=k_rung, k_diag=k_diag)
    rest_T = rest[:, :, None]

    def force(q, p):
        X = q.T.reshape(nn, 2, -1) + rest_T
        F = np.zeros_like(X)
        scale = {"railL": p["k_rail"], "railR": p["k_rail"], "rung": p["k_rung"],
                 "diag": p["k_diag"]}
        for name, P in sets.items():
            _spring_forces(X, F, P, restlen[name], scale[name], p["alpha"])
        for (i, p0, k) in anchors:
            d = X[i] - p0[:, None]
            ln = np.sqrt(d[0] ** 2 + d[1] ** 2)
            r0 = np.linalg.norm(rest[i] - p0)
            s_ = ln - r0
            F[i] -= (k * s_ * (1.0 + p["alpha"] * s_ * s_) / ln)[None, :] * d
        return F.reshape(n, -1).T

    Jtip = np.zeros((1, n))
    Jtip[0, 2 * L(N)] = Jtip[0, 2 * R(N)] = 0.5
    def sel(*idx):
        C = np.zeros((len(idx), 2 * n))
        for r, c in enumerate(idx):
            C[r, c] = 1.0
        return C
    parts = {"base_x": sel(2 * L(1)), "mid_x": sel(2 * L(N // 2)), "tip_x": sel(2 * L(N)),
             "base_strain": sel(2 * L(1) + 1, 2 * R(1) + 1)}
    return MechSim(H=H, D=D, force=force,
                   J=lambda q: np.broadcast_to(Jtip, (q.shape[0], 1, n)),
                   sigma_T2=sigma_T2, params=params, parts=parts, name="soft arm")


@register("bistable_chain", "2-nonlinear",
          "Bistable elements in series, driven by a force at the free end (hysterons).")
def bistable_chain(n_el: int = 6, a: float = 1.0, barrier: float = 8.0, spread: float = 0.5,
                   mass: float = 1.0, gamma: float = 2.0, seed: int = 0,
                   sigma_T2: float = 1.0) -> MechSim:
    """Element e has extension s_e = x_e - x_{e-1} and energy dE_e ((s_e/a)^2 - 1)^2, with
    barriers dE_e spread around `barrier` (in units of sigma_T2) so that elements switch at
    different forces, the Preisach-like ordering of hysterons. Each element starts in its
    left well (s = -a). Node 0 is fixed; the drive acts on the last node."""
    rng = np.random.default_rng(seed)
    dE = barrier * sigma_T2 * (1.0 + spread * (rng.uniform(size=n_el) - 0.5))
    n = n_el
    H = mass * np.eye(n)
    # Each node has friction gamma to the ground (keeps every coordinate damped).
    D = gamma * mass * np.eye(n)
    params = dict(barrier=barrier)

    def ext(q):
        x = np.concatenate([np.zeros((q.shape[0], 1)), q], axis=1)
        return np.diff(x, axis=1)

    def force(q, p):
        s = ext(q)
        f_el = -4.0 * dE * (p["barrier"] / barrier) * s * ((s / a) ** 2 - 1.0) / a ** 2  # -dV/ds
        F = f_el.copy()                       # element e pushes node e by +f_e ...
        F[:, :-1] -= f_el[:, 1:]              # ... and node e-1 by -f_e
        return F

    Jend = np.zeros((1, n))
    Jend[0, -1] = 1.0
    q0 = -a * np.arange(1, n + 1)             # every element in its left well
    sim = MechSim(H=H, D=D, force=force,
                  J=lambda q: np.broadcast_to(Jend, (q.shape[0], 1, n)),
                  sigma_T2=sigma_T2, params=params, parts={"end": _sel(n, n - 1)},
                  name="bistable chain", q0=q0)
    sim.extensions = ext
    sim.critical_forces = 8.0 * dE / (3.0 * np.sqrt(3.0) * a)   # F_c of @prp-critical-force
    return sim


def _sel(n, i):
    C = np.zeros((1, 2 * n))
    C[0, i] = 1.0
    return C


def mesostate(sim: MechSim, q: np.ndarray) -> np.ndarray:
    """Which well each element is in: (batch, n_el) booleans, True = right well."""
    return sim.extensions(q) > 0.0
