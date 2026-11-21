"""Where the contact force goes in a chain of masses, springs and dampers.

The body is a chain: node 0 touches the ground through a contact spring k_c and damper b_c and
takes the force; element j joins node j to node j+1 with a spring k_j and a damper b_j; the far
end is free. A single node is a part of the body. The research note of 2026-11-21 on chains
derives three results checked here:

* reciprocity: drive the body with a unit force at the part. Each internal damper then costs
  b_j |stretch_j / contact motion|^2 of effective contact damping;
* accumulation: in a chain, the cost of a damper between the contact and the part is the same
  for every part beyond it;
* the static law: rho_k(0) = b_c / (b_c + k_c^2 sum_{j<k} b_j / k_j^2).
"""
from __future__ import annotations

import numpy as np

from passive_info.systems import I_ROT, K_I, M_L, R, ZETA_SERIES, Passive, series_elastic

from mibook import checks
from mibook import propagation as pg

#: The uniform chain used in the note: eight 0.1 kg masses, 1e4 N/m springs, dampers at 5% of
#: critical per element, a contact ten times stiffer than one element, at 15% of critical on
#: the whole chain's mass.
CHAIN = dict(N=8, m=0.1, k=1.0e4, zeta=0.05, k_c=1.0e5, zeta_c=0.15)

#: A free-free axial rod touching the ground at one end, with Kelvin-Voigt material damping
#: D = c K_rod at 5% of critical on the first free-free mode. Contact stiffness EA / L.
ROD_FREE = dict(L=0.2, EA=1.0e3, rhoA=0.1, zeta=0.05, zeta_c=0.15, band_hz=400.0)


def _passive(H, K, D, J) -> Passive:
    n = H.shape[0]
    Hi = np.linalg.inv(H)
    A = np.block([[np.zeros((n, n)), np.eye(n)], [-Hi @ K, -Hi @ D]])
    B = np.vstack([np.zeros((n, J.shape[0])), Hi @ J.T])
    return Passive(A=A, B_F=B, C=np.eye(2 * n), H=H, K=K, J=J, ks=None, label="chain",
                   readout="energy")


def tree_system(m, edges, k_c: float, b_c: float):
    """A body of point masses joined by spring-damper elements; node 0 touches the ground.

    `edges` lists (i, j, k_ij, b_ij). Returns the system and the internal dampers as
    (b, a) with a = e_i - e_j. A chain is the case edges = [(j, j+1, k_j, b_j)].
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
    J = np.zeros((1, N))
    J[0, 0] = 1.0
    K += k_c * (J.T @ J)
    D += b_c * (J.T @ J)
    return _passive(H, K, D, J), dampers


def chain_system(m, k, b, k_c: float, b_c: float):
    """A chain with masses `m` (N), springs `k` and dampers `b` (N-1), contact at node 0."""
    m, k, b = (np.atleast_1d(np.asarray(v, float)) for v in (m, k, b))
    return tree_system(m, [(j, j + 1, k[j], b[j]) for j in range(m.size - 1)], k_c, b_c)


def uniform_chain(p: dict = CHAIN):
    """The uniform chain of `CHAIN`. Returns (system, b_c, dampers, element springs)."""
    N, m, k = p["N"], p["m"], p["k"]
    b = 2.0 * p["zeta"] * np.sqrt(k * m)
    b_c = 2.0 * p["zeta_c"] * np.sqrt(p["k_c"] * N * m)
    s, dampers = chain_system(np.full(N, m), np.full(N - 1, k), np.full(N - 1, b), p["k_c"], b_c)
    return s, b_c, dampers, np.full(N - 1, k)


def random_chain(rng: np.random.Generator):
    """A heterogeneous chain with 2 to 8 nodes and parameters spread over decades."""
    N = int(rng.integers(2, 9))
    k = 10 ** rng.uniform(3, 5, N - 1)
    b = 10 ** rng.uniform(-1, 1, N - 1)
    k_c, b_c = 10 ** rng.uniform(3, 6), 10 ** rng.uniform(0, 2)
    s, dampers = chain_system(rng.uniform(0.05, 0.5, N), k, b, k_c, b_c)
    return s, b_c, k_c, k, b, dampers


def free_rod(N: int, p: dict = ROD_FREE):
    """Free-free rod of N consistent-mass elements, node 0 on the ground. Returns (sys, b_c, c)."""
    L, EA, rhoA = p["L"], p["EA"], p["rhoA"]
    h = L / N
    ke = EA / h * np.array([[1.0, -1.0], [-1.0, 1.0]])
    me = rhoA * h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    n = N + 1
    K = np.zeros((n, n))
    H = np.zeros((n, n))
    for e in range(N):
        K[e:e + 2, e:e + 2] += ke
        H[e:e + 2, e:e + 2] += me
    c = 2.0 * p["zeta"] / (np.pi * np.sqrt(EA / rhoA) / L)
    k_c = EA / L
    b_c = 2.0 * p["zeta_c"] * np.sqrt(k_c * rhoA * L)
    J = np.zeros((1, n))
    J[0, 0] = 1.0
    return _passive(H, K + k_c * (J.T @ J), c * K + b_c * (J.T @ J), J), b_c, c


def node(sys, k: int) -> np.ndarray:
    """The part made of node k's position (by the angle-velocity result, velocity adds nothing)."""
    P = np.zeros((1, 2 * sys.n))
    P[0, k] = 1.0
    return P


def reciprocity_beff(sys, p: np.ndarray, dampers, b_c: float, omegas) -> np.ndarray:
    """b_eff from the reciprocity formula: drive at the part, weigh each damper's stretch.

    w = Z(w)^{-1} p^T with Z = K - w^2 H + i w D;  b_eff = b_c + sum b_j |a_j.w|^2 / |J w|^2.
    """
    D = -sys.H @ sys.A[sys.n:, sys.n:]
    J = np.atleast_2d(sys.J)[0]
    out = []
    for w in np.atleast_1d(omegas):
        Z = sys.K - w ** 2 * sys.H + 1j * w * D
        x = np.linalg.solve(Z, np.asarray(p, complex))
        contact = abs(J @ x) ** 2
        cost = sum(bj * abs(aj @ x) ** 2 for bj, aj in dampers)
        out.append(np.inf if contact == 0 else b_c + cost / contact)
    return np.array(out)


def _drive_responses(sys, P, dampers, omega):
    """W = Z^{-1} P^T, the contact motion h = (J W)^T and the stretches s_j = (a_j W)^T."""
    D = -sys.H @ sys.A[sys.n:, sys.n:]
    Z = sys.K - omega ** 2 * sys.H + 1j * omega * D
    W = np.linalg.solve(Z, np.atleast_2d(P).T.astype(complex))
    h = (np.atleast_2d(sys.J) @ W).ravel()
    S = np.array([a @ W for _, a in dampers])
    return W, h, S


def pattern_cost(sys, P, dampers, omega: float, phi) -> float:
    """sum_j b_j |a_j.x|^2 / |J x|^2 for the response x to the force pattern P^T phi."""
    _, h, S = _drive_responses(sys, P, dampers, omega)
    bb = np.array([b for b, _ in dampers])
    return float(np.sum(bb * np.abs(S @ phi) ** 2) / np.abs(h @ phi) ** 2)


def best_pattern(sys, P, dampers, omega: float):
    """The force pattern on the part that minimizes the cost, and that minimum.

    With N = sum_j b_j s_j s_j^*, the minimum is 1 / (h^* N^+ h), reached by phi = conj(N^+ h).
    Returns (phi, cost); cost is 0 when h lies outside the range of N.
    """
    _, h, S = _drive_responses(sys, P, dampers, omega)
    bb = np.array([b for b, _ in dampers])
    N = (S.T * bb) @ S.conj()
    Np = np.linalg.pinv(N, rcond=1e-12, hermitian=True)
    if np.linalg.norm(h - N @ (Np @ h)) > 1e-9 * np.linalg.norm(h):
        return None, 0.0
    psi = Np @ h
    return psi.conj(), 1.0 / float(np.real(h.conj() @ psi))


def static_law(k_c: float, b_c: float, k, b, part: int) -> float:
    """rho_k(0) = b_c / (b_c + k_c^2 sum_{j<k} b_j / k_j^2)."""
    k, b = np.asarray(k, float), np.asarray(b, float)
    return b_c / (b_c + k_c ** 2 * np.sum(b[:part] / k[:part] ** 2))


def sea_deflection_closed_form(ks: float, omega, zeta_series: float = ZETA_SERIES):
    """rho for the SEA's spring deflection: [1 + (b_s/b_c)|z_g - I w^2|^2 / (r^2 I^2 w^4)]^{-1}."""
    w = np.asarray(omega, float)
    b_s = 2.0 * zeta_series * np.sqrt(ks * I_ROT)
    z_g = K_I * R ** 2 - M_L * R ** 2 * w ** 2 + 1j * w * pg.B_CONTACT * R ** 2
    ratio = np.abs(z_g - I_ROT * w ** 2) ** 2 / (R ** 2 * I_ROT ** 2 * w ** 4)
    return 1.0 / (1.0 + (b_s / pg.B_CONTACT) * ratio)


#: The SEA's spring deflection theta_j - theta_m as a part (angle row, then with velocity).
DEFLECTION = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])


# ---------------------------------------------------------------------------------------
# Checks used by the tests and the research note
# ---------------------------------------------------------------------------------------

def check_reciprocity(n_chains: int = 20, f_hz=(0.7, 6.0, 33.0, 90.0), seed: int = 0) -> dict:
    """The reciprocity formula against the state-space b_eff, on random chains and the SEA."""
    rng = np.random.default_rng(seed)
    w = 2 * np.pi * np.asarray(f_hz)
    worst = 0.0
    for _ in range(n_chains):
        s, b_c, _, _, _, dampers = random_chain(rng)
        for k in range(s.n):
            num = pg.spectral_propagation(s, node(s, k), w, b_contact=b_c)
            rec = b_c / reciprocity_beff(s, np.eye(s.n)[k], dampers, b_c, w)
            worst = max(worst, float(np.max(np.abs(num - rec))))
    sea_worst = 0.0
    for ks in (70.0, 1e3, 1e4):
        s = series_elastic(ks, readout="energy")
        b_s = 2.0 * ZETA_SERIES * np.sqrt(ks * I_ROT)
        dampers = [(b_s, np.array([1.0, -1.0]))]
        for p in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, -1.0])):
            P = np.hstack([p, np.zeros(2)])[None, :]
            num = pg.spectral_propagation(s, P, w)
            # J = (-r, 0) already carries the lever arm, so b_c stays in N s/m
            rec = pg.B_CONTACT / reciprocity_beff(s, p, dampers, pg.B_CONTACT, w)
            sea_worst = max(sea_worst, float(np.max(np.abs(num - rec))))
    return dict(chain_max_abs_err=worst, sea_max_abs_err=sea_worst)


def check_accumulation(f_hz=(0.7, 6.0, 33.0, 90.0)) -> dict:
    """The cost of damper j, |stretch_j / contact motion|^2, is the same for every part beyond j."""
    s, b_c, dampers, _ = uniform_chain()
    D = -s.H @ s.A[s.n:, s.n:]
    spread = 0.0
    for w in 2 * np.pi * np.asarray(f_hz):
        Z = s.K - w ** 2 * s.H + 1j * w * D
        for j, (bj, aj) in enumerate(dampers):
            costs = []
            for k in range(j + 1, s.n):
                x = np.linalg.solve(Z, np.eye(s.n)[k].astype(complex))
                costs.append(abs(aj @ x) ** 2 / abs(x[0]) ** 2)
            costs = np.array(costs)
            spread = max(spread, float(np.max(np.abs(costs / costs[0] - 1.0))))
    return dict(max_rel_spread=spread)


def check_static_law(n_chains: int = 20, seed: int = 1, f_hz: float = 1e-4) -> dict:
    """rho_k at very low frequency against the static law, on random chains."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_chains):
        s, b_c, k_c, k, b, _ = random_chain(rng)
        for part in range(s.n):
            r = float(pg.spectral_propagation(s, node(s, part), [2 * np.pi * f_hz],
                                              b_contact=b_c)[0])
            worst = max(worst, abs(r - static_law(k_c, b_c, k, b, part)))
    return dict(max_abs_err=worst)


def check_chain_shares(p: dict = CHAIN, band_hz: float = 100.0) -> dict:
    """Finite-window share and band-average rho of each node of the uniform chain."""
    s, b_c, _, _ = uniform_chain(p)
    f = np.linspace(0.0, band_hz, 401)[1:]
    whole = checks.langevin_bits(s, np.eye(2 * s.n), 0.0, band_hz=band_hz)
    share, rho_mean, static = [], [], []
    k = np.full(p["N"] - 1, p["k"])
    b = np.full(p["N"] - 1, 2.0 * p["zeta"] * np.sqrt(p["k"] * p["m"]))
    for n in range(s.n):
        P = np.zeros((2, 2 * s.n))
        P[0, n], P[1, s.n + n] = 1.0, 1.0
        share.append(checks.langevin_bits(s, P, pg.JITTER, band_hz=band_hz) / whole)
        rho_mean.append(float(np.mean(pg.spectral_propagation(s, node(s, n), 2 * np.pi * f,
                                                               b_contact=b_c))))
        static.append(static_law(p["k_c"], b_c, k, b, n))
    return dict(share=np.array(share), rho_mean=np.array(rho_mean), static=np.array(static),
                whole=whole, b_c=b_c, modes_hz=s.modes_hz())


def check_rod_mesh(Ns=(4, 8, 16, 32, 64), fracs=(0.0, 0.25, 0.5, 0.75, 1.0),
                   p: dict = ROD_FREE, n_freq: int = 200) -> dict:
    """Band-average rho at fixed positions along the rod, under mesh refinement."""
    f = np.linspace(0.0, p["band_hz"], n_freq + 1)[1:]
    rows, static = [], []
    for N in Ns:
        s, b_c, c = free_rod(N, p)
        rows.append([float(np.mean(pg.spectral_propagation(s, node(s, int(round(x * N))),
                                                           2 * np.pi * f, b_contact=b_c)))
                     for x in fracs])
        static.append([float(pg.spectral_propagation(s, node(s, int(round(x * N))),
                                                     [2 * np.pi * 1e-4], b_contact=b_c)[0])
                       for x in fracs])
    _, b_c, c = free_rod(Ns[0], p)
    k_c = p["EA"] / p["L"]
    law = [1.0 / (1.0 + c * k_c ** 2 * x * p["L"] / (b_c * p["EA"])) for x in fracs]
    return dict(N=np.asarray(Ns), fracs=np.asarray(fracs), rho_mean=np.array(rows),
                static=np.array(static), static_law=np.array(law))


def check_deflection(ks_list=(70.0, 1e3, 1e4), f_hz=(1.0, 10.0, 50.0, 100.0)) -> dict:
    """The SEA's spring deflection: closed form, and its share falling with stiffness."""
    w = 2 * np.pi * np.asarray(f_hz)
    worst, share, predicted = 0.0, {}, {}
    for ks in ks_list:
        s = series_elastic(ks, readout="energy")
        num = pg.spectral_propagation(s, DEFLECTION[:1], w)
        worst = max(worst, float(np.max(np.abs(num - sea_deflection_closed_form(ks, w)))))
        share[ks] = pg.information_share(s, DEFLECTION)
        predicted[ks] = pg.predicted_share(s, DEFLECTION)
    low = float(sea_deflection_closed_form(70.0, [2 * np.pi * 1e-3])[0])
    return dict(max_abs_err=worst, share=share, predicted=predicted, rho_near_dc=low)


def check_contact_dips(n_freq: int = 4000) -> dict:
    """The contact node's rho dips at the natural frequencies of the body held fixed there."""
    import scipy.linalg as sla
    from scipy.signal import argrelmin
    out = {}
    for name, (s, b_c, band) in {"rod": free_rod(32)[:2] + (400.0,),
                                 "chain": uniform_chain()[:2] + (100.0,)}.items():
        f = np.linspace(0.5, band, n_freq)
        r = pg.spectral_propagation(s, node(s, 0), 2 * np.pi * f, b_contact=b_c)
        i = argrelmin(r)[0]
        i = i[r[i] < 0.5]                                 # real dips, not round-off ripples
        clamped = np.sqrt(sla.eigh(s.K[1:, 1:], s.H[1:, 1:], eigvals_only=True)) / (2 * np.pi)
        out[name] = dict(dips_hz=f[i], rho_at_dips=r[i], clamped_hz=clamped[clamped < band])
    return out


# ---------------------------------------------------------------------------------------
# Parts with several coordinates, and branching bodies
# ---------------------------------------------------------------------------------------

def node_set(sys, nodes) -> np.ndarray:
    """Position rows for a set of nodes, as a selector on x_q."""
    P = np.zeros((len(nodes), 2 * sys.n))
    P[np.arange(len(nodes)), list(nodes)] = 1.0
    return P


def check_matrix_reciprocity(n_chains: int = 15, f_hz=(0.8, 7.0, 40.0), n_random: int = 200,
                             seed: int = 3) -> dict:
    """Several-coordinate parts: b_eff = b_c + the cost of the best force pattern on the part.

    Compares the state-space b_eff with the best-pattern formula, checks that random patterns
    never do better, and that the predicted pattern reaches the minimum.
    """
    rng = np.random.default_rng(seed)
    w = 2 * np.pi * np.asarray(f_hz)
    rel, beaten, reach = 0.0, 0, 0.0
    for _ in range(n_chains):
        s, b_c, _, _, _, dampers = random_chain(rng)
        if s.n < 3:
            continue
        nodes = sorted(rng.choice(s.n, size=min(3, s.n - 1), replace=False))
        P = node_set(s, nodes)[:, :s.n]
        num = pg.effective_contact_damping(s, node_set(s, nodes), w, b_contact=b_c)
        for wi, bn in zip(w, num):
            phi, c = best_pattern(s, P, dampers, wi)
            rel = max(rel, abs(b_c + c - bn) / bn)
            reach = max(reach, abs(pattern_cost(s, P, dampers, wi, phi) - c) / c)
            for _ in range(n_random):
                z = rng.normal(size=len(nodes)) + 1j * rng.normal(size=len(nodes))
                beaten += pattern_cost(s, P, dampers, wi, z) < c * (1 - 1e-9)
    return dict(max_rel_err=rel, n_beaten=int(beaten), reach_rel_err=reach)


def check_set_static_law(n_chains: int = 20, seed: int = 5) -> dict:
    """At low frequency a set of nodes holds what its node nearest the contact holds."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_chains):
        s, b_c, k_c, k, b, _ = random_chain(rng)
        if s.n < 3:
            continue
        nodes = sorted(rng.choice(np.arange(1, s.n), size=2, replace=False))
        r = float(pg.spectral_propagation(s, node_set(s, nodes), [2 * np.pi * 1e-4],
                                          b_contact=b_c)[0])
        worst = max(worst, abs(r - static_law(k_c, b_c, k, b, nodes[0])))
    return dict(max_abs_err=worst)


#: A toy vertical leg on a trunk: foot (0, on the ground), knee (1), hip (2), trunk (3), and
#: two swing legs hanging from the trunk, knee (4, 6) and foot (5, 7). Masses in kg.
LEG_TRUNK = dict(m=(0.2, 0.5, 1.0, 5.0, 0.5, 0.2, 0.5, 0.2),
                 edges=((0, 1, 2e4), (1, 2, 2e4), (2, 3, 5e4),
                        (3, 4, 2e4), (4, 5, 2e4), (3, 6, 2e4), (6, 7, 2e4)),
                 swing=(3, 4, 5, 6), zeta=0.05, k_c=1e5, zeta_c=0.15)
#: The path from the foot to each node, as edge indices into LEG_TRUNK["edges"].
LEG_TRUNK_PATHS = {0: (), 1: (0,), 2: (0, 1), 3: (0, 1, 2), 4: (0, 1, 2, 3),
                   5: (0, 1, 2, 3, 4), 6: (0, 1, 2, 5), 7: (0, 1, 2, 5, 6)}


def leg_trunk(zeta_swing: float = 0.05, p: dict = LEG_TRUNK):
    """The leg-on-trunk body; elements are damped at `zeta` (swing legs at `zeta_swing`)
    of critical on the lighter of their two masses. Returns (sys, b_c, dampers, edges)."""
    m = p["m"]
    edges = []
    for e, (i, j, k) in enumerate(p["edges"]):
        z = zeta_swing if e in p["swing"] else p["zeta"]
        edges.append((i, j, k, 2.0 * z * np.sqrt(k * min(m[i], m[j]))))
    b_c = 2.0 * p["zeta_c"] * np.sqrt(p["k_c"] * sum(m))
    s, dampers = tree_system(m, edges, p["k_c"], b_c)
    return s, b_c, dampers, edges


def check_tree(zetas=(0.05, 0.2), p: dict = LEG_TRUNK) -> dict:
    """Shares on the leg-on-trunk body; the static law follows the path to the contact."""
    out = {}
    for z in zetas:
        s, b_c, _, edges = leg_trunk(z, p)
        whole = checks.langevin_bits(s, np.eye(2 * s.n), 0.0)
        share, static, path_law = [], [], []
        for n in range(s.n):
            P = np.zeros((2, 2 * s.n))
            P[0, n], P[1, s.n + n] = 1.0, 1.0
            share.append(checks.langevin_bits(s, P, pg.JITTER) / whole)
            static.append(float(pg.spectral_propagation(s, node(s, n), [2 * np.pi * 1e-4],
                                                        b_contact=b_c)[0]))
            path_law.append(b_c / (b_c + p["k_c"] ** 2 * sum(edges[e][3] / edges[e][2] ** 2
                                                             for e in LEG_TRUNK_PATHS[n])))
        out[z] = dict(whole=whole, share=np.array(share), static=np.array(static),
                      path_law=np.array(path_law))
    return out


def check_bridge(f_hz=(3.0, 25.0, 60.0), n_random: int = 20, seed: int = 7) -> dict:
    """Across a bridge (an element whose removal separates contact and part), every damper on
    the contact side costs the same for any part, and any force pattern, beyond the bridge.

    On the leg-on-trunk body the hip-trunk element is a bridge; the contact side holds the
    foot-knee and knee-hip dampers and the bridge damper itself (indices 0, 1, 2).
    """
    rng = np.random.default_rng(seed)
    s, _, dampers, _ = leg_trunk()
    far = [3, 4, 5, 6, 7]
    spread = 0.0
    for w in 2 * np.pi * np.asarray(f_hz):
        ref = None
        trials = [np.eye(s.n)[[n]] for n in far]
        for _ in range(n_random):
            P = np.eye(s.n)[far]
            trials.append((rng.normal(size=len(far)) + 1j * rng.normal(size=len(far))) @ P)
        for P in trials:
            _, h, S = _drive_responses(s, np.atleast_2d(P), dampers, w)
            c = np.abs(S[:3, 0]) ** 2 / abs(h[0]) ** 2
            ref = c if ref is None else ref
            spread = max(spread, float(np.max(np.abs(c / ref - 1.0))))
    return dict(max_rel_spread=spread)


def check_pairs(pairs=((1,), (7,), (1, 7), (6,), (6, 7)), band_hz: float = 100.0) -> dict:
    """Finite-window share of single nodes and pairs on the uniform chain."""
    s, _, _, _ = uniform_chain()
    whole = checks.langevin_bits(s, np.eye(2 * s.n), 0.0, band_hz=band_hz)
    out = {}
    for nodes in pairs:
        rows = [n for n in nodes] + [s.n + n for n in nodes]
        P = np.eye(2 * s.n)[rows]
        out[nodes] = checks.langevin_bits(s, P, pg.JITTER, band_hz=band_hz) / whole
    return out
