"""Bodies with two feet on the ground: what the body, and each part of it, holds about each
contact force.

Each contact i is a node c_i held by a ground spring k_i and damper b_i, and takes its own
unknown force f_i. The internal dampers are spring-damper elements between nodes. Results
checked here (research note of 2026-11-21 on two contacts):

* the whole body sees the two forces through the damping matrix B = (J D^{-1} J^T)^{-1},
  independent of frequency; for a tree-shaped body B = B_c + b_x (e1 - e2)(e1 - e2)^T, with
  b_x the dampers on the path between the feet combined in series;
* a one-coordinate part sees one combination of the forces, weighted by how much each foot
  moves when the part is pushed, against every damper weighted by its stretch;
* a static law in which the push splits between the feet by compliance.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from mechinfo.mechanics import bits_from_gram
from passive_info.information import FS, SIGMA_A, response_matrix

from mibook import chain as ch
from mibook import checks


def body(m, edges, contacts):
    """Point masses, spring-damper `edges` (i, j, k, b), and `contacts` (node, k_c, b_c).

    Returns (sys, internal dampers as (b, a), B_c as a diagonal matrix).
    """
    m = np.atleast_1d(np.asarray(m, float))
    N = m.size
    H = np.diag(m)
    K = np.zeros((N, N))
    D = np.zeros((N, N))
    dampers = []
    for i, j, k, b in edges:
        a = np.zeros(N)
        a[i], a[j] = 1.0, -1.0
        K += k * np.outer(a, a)
        D += b * np.outer(a, a)
        dampers.append((float(b), a))
    J = np.zeros((len(contacts), N))
    for r, (c, k_c, b_c) in enumerate(contacts):
        J[r, c] = 1.0
        K[c, c] += k_c
        D[c, c] += b_c
    Bc = np.diag([b for _, _, b in contacts]).astype(float)
    return ch._passive(H, K, D, J), dampers, Bc


def damping(sys) -> np.ndarray:
    return -sys.H @ sys.A[sys.n:, sys.n:]


def whole_body_damping(sys) -> np.ndarray:
    """B = (J D^{-1} J^T)^{-1}: the damping through which the whole body sees the forces."""
    J = np.atleast_2d(sys.J)
    return np.linalg.inv(J @ np.linalg.solve(damping(sys), J.T))


def precision(sys, P, Bc, omega: float) -> np.ndarray:
    """F_P(w) = H^* (H B_c H^* + N)^+ H, the part's information matrix per unit 2 sigma_T^2.

    H = P Z^{-1} J^T is the part's response to the forces, N = P Z^{-1} D_int Z^{-*} P^T the
    noise it receives from the internal dampers. P selects positions (rows of length n).
    """
    J = np.atleast_2d(sys.J)
    D = damping(sys)
    Dint = D - J.T @ Bc @ J
    Z = sys.K - omega ** 2 * sys.H + 1j * omega * D
    Zi = np.linalg.inv(Z)
    P = np.atleast_2d(P)
    Hm = P @ Zi @ J.T
    Ntot = Hm @ Bc @ Hm.conj().T + P @ Zi @ Dint @ Zi.conj().T @ P.T
    return Hm.conj().T @ np.linalg.pinv(Ntot, rcond=1e-12, hermitian=True) @ Hm


def push_precision(sys, p, dampers, Bc, omega: float) -> np.ndarray:
    """The one-coordinate rule: push on the part with x = Z^{-1} p and watch the feet.

    F = conj(h) h^T / (sum_i b_i |h_i|^2 + sum_j b_j |a_j.x|^2), with h = J x.
    """
    J = np.atleast_2d(sys.J)
    Z = sys.K - omega ** 2 * sys.H + 1j * omega * damping(sys)
    x = np.linalg.solve(Z, np.asarray(p, complex))
    h = J @ x
    den = float(np.sum(np.diag(Bc) * np.abs(h) ** 2) + sum(b * abs(a @ x) ** 2 for b, a in dampers))
    return np.outer(h.conj(), h) / den


def series(b) -> float:
    """Dampers in series: 1 / sum(1 / b_j)."""
    return 1.0 / float(np.sum(1.0 / np.asarray(b, float)))


# ---------------------------------------------------------------------------------------
# Information in a finite window about each force and each combination
# ---------------------------------------------------------------------------------------

#: Total load and difference, as an orthonormal change of force variables: f = U g with
#: g = ((f1 + f2) / sqrt 2, (f1 - f2) / sqrt 2), so the g's are independent like the f's.
SUM_DIFF = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)


def force_bits(sys, C, eps: float, U=None, sigma_T2: float | None = None, fs: float = FS,
               band_hz: float = 100.0) -> dict:
    """Bits held by the observation C x about all forces together and about each one alone.

    The noise is the Langevin body's (as in `checks.langevin_bits`). Forces are written
    f = U g with independent g; `U=None` means g = f. Each g_i is scored with the others
    treated as unknown.
    """
    C = np.atleast_2d(C)
    Ms = [response_matrix(dataclasses.replace(sys, C=C, B_F=sys.B_F[:, [i]]), fs=fs,
                          band_hz=band_hz) for i in range(sys.B_F.shape[1])]
    U = np.eye(len(Ms)) if U is None else U
    Mg = [sum(U[i, r] * Ms[i] for i in range(len(Ms))) for r in range(U.shape[1])]
    m = C.shape[0]
    S = checks.langevin_noise_cov(sys, C, Ms[0].shape[0] // m, eps, sigma_T2, fs)
    out = {}
    Mall = np.hstack(Mg)
    L = np.linalg.cholesky(S)
    Z = np.linalg.solve(L, Mall)
    out["all"] = bits_from_gram(SIGMA_A ** 2 * (Z.T @ Z))
    for r in range(len(Mg)):
        others = [Mg[q] for q in range(len(Mg)) if q != r]
        Sr = S + SIGMA_A ** 2 * sum(Mo @ Mo.T for Mo in others)
        Zr = np.linalg.solve(np.linalg.cholesky(Sr), Mg[r])
        out[r] = bits_from_gram(SIGMA_A ** 2 * (Zr.T @ Zr))
    return out


# ---------------------------------------------------------------------------------------
# Bodies used in the note
# ---------------------------------------------------------------------------------------

def random_two_foot(rng: np.random.Generator):
    """A random tree-shaped body with 3 to 8 nodes and two contacts on distinct nodes."""
    N = int(rng.integers(3, 9))
    edges = []
    for j in range(1, N):
        i = int(rng.integers(0, j))                 # attach node j to an earlier node: a tree
        edges.append((i, j, 10 ** rng.uniform(3, 5), 10 ** rng.uniform(-1, 1)))
    c1, c2 = (int(v) for v in rng.choice(N, size=2, replace=False))
    contacts = [(c1, 10 ** rng.uniform(3, 6), 10 ** rng.uniform(0, 2)),
                (c2, 10 ** rng.uniform(3, 6), 10 ** rng.uniform(0, 2))]
    s, dampers, Bc = body(rng.uniform(0.05, 0.5, N), edges, contacts)
    return s, dampers, Bc, edges, (c1, c2)


def tree_path(edges, n: int, a: int, b: int):
    """Edge indices on the path from node a to node b in a tree with n nodes."""
    adj = {v: [] for v in range(n)}
    for e, (i, j, *_) in enumerate(edges):
        adj[i].append((j, e))
        adj[j].append((i, e))
    prev = {a: None}
    stack = [a]
    while stack:
        v = stack.pop()
        for w, e in adj[v]:
            if w not in prev:
                prev[w] = (v, e)
                stack.append(w)
    path, v = [], b
    while prev[v] is not None:
        v, e = prev[v]
        path.append(e)
    return path[::-1]


def two_foot_chain(rng: np.random.Generator):
    """A chain whose two end nodes both touch the ground, with random parameters."""
    N = int(rng.integers(3, 9))
    k = 10 ** rng.uniform(3, 5, N - 1)
    b = 10 ** rng.uniform(-1, 1, N - 1)
    k1, k2 = 10 ** rng.uniform(3, 6, 2)
    b1, b2 = 10 ** rng.uniform(0, 2, 2)
    s, dampers, Bc = body(rng.uniform(0.05, 0.5, N),
                          [(j, j + 1, k[j], b[j]) for j in range(N - 1)],
                          [(0, k1, b1), (N - 1, k2, b2)])
    return s, dampers, Bc, k, b, (k1, k2), (b1, b2)


def two_foot_static_law(k, b, kc, bc, part: int) -> np.ndarray:
    """F(0) for node `part` of a chain whose two ends touch the ground.

    The unit push splits between the two routes to the ground in inverse proportion to their
    compliances; each foot moves F_i / k_ci, and each damper stretches by its load over k_j.
    """
    k, b = np.asarray(k, float), np.asarray(b, float)
    C1 = 1.0 / kc[0] + np.sum(1.0 / k[:part])
    C2 = 1.0 / kc[1] + np.sum(1.0 / k[part:])
    F1, F2 = C2 / (C1 + C2), C1 / (C1 + C2)
    h = np.array([F1 / kc[0], F2 / kc[1]])
    den = (F1 ** 2 * (bc[0] / kc[0] ** 2 + np.sum(b[:part] / k[:part] ** 2))
           + F2 ** 2 * (bc[1] / kc[1] ** 2 + np.sum(b[part:] / k[part:] ** 2)))
    return np.outer(h, h) / den


#: The leg-on-trunk toy of the branching note with swing foot A (node 5) also on the
#: ground, on the same contact as the stance foot. Swing leg B (nodes 6, 7) stays in the air.
def two_foot_stance(p: dict = ch.LEG_TRUNK):
    m = p["m"]
    edges = [(i, j, k, 2.0 * p["zeta"] * np.sqrt(k * min(m[i], m[j]))) for i, j, k in p["edges"]]
    b_c = 2.0 * p["zeta_c"] * np.sqrt(p["k_c"] * sum(m))
    s, dampers, Bc = body(m, edges, [(0, p["k_c"], b_c), (5, p["k_c"], b_c)])
    return s, dampers, Bc, edges


# ---------------------------------------------------------------------------------------
# Checks used by the tests and the research note
# ---------------------------------------------------------------------------------------

def check_whole_body(n_bodies: int = 20, f_hz=(0.5, 8.0, 60.0), seed: int = 11) -> dict:
    """The whole body's information matrix is J D^{-1} J^T at every frequency, and
    (J D^{-1} J^T)^{-1} = B_c + b_x L with b_x the series damping of the path between the feet."""
    rng = np.random.default_rng(seed)
    L = np.array([[1.0, -1.0], [-1.0, 1.0]])
    e_freq = e_series = 0.0
    for _ in range(n_bodies):
        s, _, Bc, edges, (c1, c2) = random_two_foot(rng)
        B = whole_body_damping(s)
        for w in 2 * np.pi * np.asarray(f_hz):
            F = precision(s, np.eye(s.n), Bc, w)
            e_freq = max(e_freq, float(np.max(np.abs(F - np.linalg.inv(B))) / np.max(np.abs(F))))
        bx = series([edges[e][3] for e in tree_path(edges, s.n, c1, c2)])
        e_series = max(e_series, float(np.max(np.abs(B - (Bc + bx * L))) / np.max(np.abs(B))))
    return dict(freq_rel_err=e_freq, series_rel_err=e_series)


def check_push_rule(n_bodies: int = 20, f_hz=(0.5, 8.0, 60.0), seed: int = 12) -> dict:
    """The one-coordinate push rule against the state-space information matrix."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_bodies):
        s, dampers, Bc, _, _ = random_two_foot(rng)
        for w in 2 * np.pi * np.asarray(f_hz):
            for k in range(s.n):
                Fp = precision(s, np.eye(s.n)[[k]], Bc, w)
                Fr = push_precision(s, np.eye(s.n)[k], dampers, Bc, w)
                worst = max(worst, float(np.max(np.abs(Fp - Fr)) / np.max(np.abs(Fp))))
    return dict(max_rel_err=worst)


def check_two_foot_static(n_bodies: int = 20, seed: int = 13) -> dict:
    """The static law for chains with both ends on the ground."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_bodies):
        s, _, Bc, k, b, kc, bc = two_foot_chain(rng)
        for part in range(s.n):
            F = precision(s, np.eye(s.n)[[part]], Bc, 2 * np.pi * 1e-4).real
            law = two_foot_static_law(k, b, kc, bc, part)
            worst = max(worst, float(np.max(np.abs(F - law)) / np.max(np.abs(law))))
    return dict(max_rel_err=worst)


STANCE_PARTS = {"stance foot": 0, "stance hip": 2, "trunk": 3, "second knee": 4,
                "second foot": 5, "swing foot": 7}


def check_stance(scales=(1.0, 10.0, 100.0)) -> dict:
    """Two-foot stance: bits about each force, the total load and the difference.

    `scales` multiply every internal damper, to show the cost of the path between the feet.
    """
    s0, _, Bc, edges = two_foot_stance()
    out = dict(b_c=float(Bc[0, 0]), whole={}, parts={})
    for sc in scales:
        s, _, _ = body(ch.LEG_TRUNK["m"], [(i, j, k, sc * b) for i, j, k, b in edges],
                       [(0, ch.LEG_TRUNK["k_c"], Bc[0, 0]), (5, ch.LEG_TRUNK["k_c"], Bc[1, 1])])
        bx = series([sc * edges[e][3] for e in tree_path(edges, s.n, 0, 5)])
        out["whole"][sc] = dict(b_x=bx, B=whole_body_damping(s),
                                f=force_bits(s, np.eye(2 * s.n), 0.0),
                                sd=force_bits(s, np.eye(2 * s.n), 0.0, U=SUM_DIFF))
    for name, nd in STANCE_PARTS.items():
        P = np.zeros((2, 2 * s0.n))
        P[0, nd], P[1, s0.n + nd] = 1.0, 1.0
        out["parts"][name] = dict(f=force_bits(s0, P, 1e-9), sd=force_bits(s0, P, 1e-9, U=SUM_DIFF))
    return out
