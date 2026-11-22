"""A mechanical bit inside a body: what the rest of the body holds about it (thread T7).

The body is the clamped SEA in joint coordinates q = (theta_j, theta_m) on the book's ground. A
bistable element with energy V(z) = dE ((z/a)^2 - 1)^2 acts on z = a_e^T q, in one of two places:

* 'series': it replaces the series spring, z = theta_j - theta_m;
* 'latch' : it holds the rotor against the frame, z = theta_m, alongside the series spring.

Near either well the element is a spring of stiffness k_w = 8 dE / a^2 whose rest point is +a or
-a, so the stored bit acts on the rest of the body as a static force 2 a k_w a_e. Everything else
is linear, with the Langevin body's thermal noise at sigma_T^2.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from mechinfo.mechanics import ground_damping
from passive_info.information import FS, SIGMA_ENERGY
from passive_info.systems import I_ROT, K_I, M_L, R, ZETA_CONTACT, Passive, reference_m_e

from mibook import checks

SIGMA_T2 = SIGMA_ENERGY ** 2
#: Stiffness of the element alone at the bottom of a well (N m/rad). The latch is ten times
#: stiffer than the series spring it works against, so that element and body stay bistable.
K_W = {"series": 1.0e3, "latch": 1.0e4}
BETA = 8.0                            #: barrier over sigma_T^2
ZETA = 0.10                           #: damping ratio of the element, as for the series spring
K_S = 1.0e3                           #: series spring in the latch placement (N m/rad)


def element(placement: str):
    """Direction a_e of the element's coordinate in q."""
    return {"series": np.array([1.0, -1.0]), "latch": np.array([0.0, 1.0])}[placement]


def well_half_width(beta: float, k_w: float) -> float:
    """a, from k_w = 8 dE / a^2 with dE = beta sigma_T^2."""
    return float(np.sqrt(8.0 * beta * SIGMA_T2 / k_w))


def matrices(placement: str, k_w: float | None = None):
    """H, the linear stiffness without the element, D (element damper included), and a_e."""
    k_w = K_W[placement] if k_w is None else k_w
    H = np.diag([M_L * R ** 2, I_ROT])
    b_c = float(ground_damping(K_I, reference_m_e(), ZETA_CONTACT))
    K0 = np.diag([K_I * R ** 2, 0.0])
    D = np.diag([b_c * R ** 2, 0.0])
    ae = element(placement)
    b_e = 2.0 * ZETA * np.sqrt(k_w * I_ROT)
    D = D + b_e * np.outer(ae, ae)
    if placement == "latch":
        couple = np.array([[1.0, -1.0], [-1.0, 1.0]])
        K0 = K0 + K_S * couple
        D = D + 2.0 * ZETA * np.sqrt(K_S * I_ROT) * couple
    return H, K0, D, ae


def linearized(placement: str, k_w: float | None = None, beta: float | None = None) -> Passive:
    """The body with the element replaced by its stiffness at the bottom of a true well."""
    k_w = K_W[placement] if k_w is None else k_w
    H, K0, D, ae = matrices(placement, k_w)
    beta = beta_for_barrier(placement, BETA, k_w) if beta is None else beta
    K = K0 + effective(placement, beta, k_w)["k_local"] * np.outer(ae, ae)
    n = 2
    Hi = np.linalg.inv(H)
    A = np.block([[np.zeros((n, n)), np.eye(n)], [-Hi @ K, -Hi @ D]])
    J = np.array([[-R, 0.0]])
    B = np.vstack([np.zeros((n, 1)), Hi @ J.T])
    return Passive(A=A, B_F=B, C=np.eye(2 * n), H=H, K=K, J=J, ks=None, label=placement,
                   readout="energy")


def effective(placement: str, beta: float = BETA, k_w: float | None = None) -> dict:
    """The element together with the body it pushes against.

    With the element held at coordinate z, the rest of the body settles where it stores least
    energy: q(z) = z u, with u minimizing u^T K0 u subject to a_e . u = 1 (a KKT solve, which
    also handles a free rotor). The body then adds k_eff z^2 / 2 to the element's energy, with
    k_eff = u^T K0 u, so the wells move to z_w^2 = a^2 (1 - k_eff / (k_w / 2)) and the barrier
    drops to dE (1 - k_eff / (k_w / 2))^2. The stored bit moves the body by 2 z_w u.
    """
    k_w = K_W[placement] if k_w is None else k_w
    H, K0, D, ae = matrices(placement, k_w)
    kkt = np.block([[K0, ae[:, None]], [ae[None, :], np.zeros((1, 1))]])
    sol = np.linalg.solve(kkt, np.r_[np.zeros(2), 1.0])
    u = sol[:2]
    k_eff = float(u @ K0 @ u)
    a = well_half_width(beta, k_w)
    dE = beta * SIGMA_T2
    f = 1.0 - k_eff / (k_w / 2.0)
    if f <= 0:
        return dict(bistable=False, k_eff=k_eff)
    z_w = a * np.sqrt(f)
    curv = 4.0 * dE / a ** 2 * (3.0 * z_w ** 2 / a ** 2 - 1.0)       # V''(z_w), element alone
    return dict(bistable=True, k_eff=k_eff, z_w=float(z_w), u=u, barrier=float(dE * f * f),
                barrier_over_T=float(beta * f * f), k_local=float(curv), shift=2.0 * z_w * u)


def beta_for_barrier(placement: str, target: float = BETA, k_w: float | None = None) -> float:
    """The element's own barrier that gives an effective barrier of `target` sigma_T^2."""
    k_w = K_W[placement] if k_w is None else k_w
    k_eff = effective(placement, 1.0, k_w)["k_eff"]
    # f does not depend on beta when k_w is fixed: f = 1 - k_eff / (k_w / 2)
    f = 1.0 - k_eff / (k_w / 2.0)
    return float(target / (f * f))


PARTS = {"link": 0, "rotor": 1}


def static_reading(placement: str, part: str, k_w: float | None = None) -> dict:
    """What the part reads about the bit, with the element's barrier set so that the element
    and body together have a barrier of BETA sigma_T^2.

    The bit moves the part by Delta = 2 z_w u_p (see `effective`). Linearized about a well
    with stiffness matrix K, a snapshot separates the two bits by d^2 = Delta^2 / (sigma^2 p.x),
    x = K^{-1} p, and over a long window the separation grows at
    Delta^2 / (2 sigma^2 x^T D x) per second.
    """
    k_w = K_W[placement] if k_w is None else k_w
    beta = beta_for_barrier(placement, BETA, k_w)
    e = effective(placement, beta, k_w)
    s = linearized(placement, k_w, beta)
    _, _, D, ae = matrices(placement, k_w)
    p = np.zeros(2)
    p[PARTS[part]] = 1.0
    x = np.linalg.solve(s.K, p)
    delta = float(e["shift"][PARTS[part]])
    return dict(shift=delta, snapshot_d2=delta ** 2 / (SIGMA_T2 * float(p @ x)),
                rate_d2=delta ** 2 / (2.0 * SIGMA_T2 * float(x @ D @ x)),
                beta_element=beta, barrier_over_T=e["barrier_over_T"], k_eff=e["k_eff"])


def window_d2(placement: str, part: str, t_window: float, k_w: float | None = None,
              fs: float = FS) -> float:
    """Exact Gaussian separation d^2 = mu^T Sigma^{-1} mu between the two bits, for the part's
    sampled position over a window, in the linearized body."""
    k_w = K_W[placement] if k_w is None else k_w
    s = linearized(placement, k_w)
    n_t = int(round(t_window * fs)) + 1
    C = np.zeros((1, 4))
    C[0, PARTS[part]] = 1.0
    S = checks.langevin_noise_cov(s, C, n_t)
    mu = np.full(n_t, static_reading(placement, part, k_w)["shift"])
    return float(mu @ np.linalg.solve(S, mu))


def binary_gaussian_bits(d2: float, n_grid: int = 4001) -> float:
    """Bits about an equally likely binary choice between two Gaussians at separation d."""
    if d2 < 1e-14:
        return 0.0
    d = np.sqrt(d2)
    L = np.linspace(d2 / 2 - 8 * d, d2 / 2 + 8 * d, n_grid)
    w = np.exp(-(L - d2 / 2) ** 2 / (2 * d2))
    w /= w.sum()
    return float(1.0 - np.sum(w * np.log2(1.0 + np.exp(-L))))


def simulate_window_means(placement: str, part: str, bit: int, t_window: float,
                          beta: float | None = None, k_w: float | None = None, n: int = 2000,
                          dt: float = 5e-5, t_burn: float = 0.05, seed: int = 0) -> np.ndarray:
    """Nonlinear simulation: window means of the part's angle, element started in well `bit`
    (+1 or -1), and which elements stayed in that well throughout the window. BAOAB with the
    full damping matrix in the O step."""
    k_w = K_W[placement] if k_w is None else k_w
    rng = np.random.default_rng(seed)
    H, K0, D, ae = matrices(placement, k_w)
    beta = beta_for_barrier(placement, BETA, k_w) if beta is None else beta
    a = well_half_width(beta, k_w)
    dE = beta * SIGMA_T2
    Hi = np.linalg.inv(H)
    Cdt = sla.expm(-Hi @ D * dt)
    Lo = np.linalg.cholesky(SIGMA_T2 * (Hi - Cdt @ Hi @ Cdt.T) + 1e-30 * np.eye(2))

    def force(q):
        z = q @ ae
        dV = 4.0 * dE / a ** 2 * z * (z * z / a ** 2 - 1.0)
        return -q @ K0.T - np.outer(dV, ae)

    # start at the bottom of the chosen well of element and body together
    q = np.tile(0.5 * bit * effective(placement, beta, k_w)["shift"], (n, 1))
    v = rng.normal(size=(n, 2)) @ np.linalg.cholesky(SIGMA_T2 * Hi).T
    F = force(q)
    n_burn, n_win = int(round(t_burn / dt)), int(round(t_window / dt))
    acc = np.zeros(n)
    stayed = np.ones(n, bool)
    for k in range(n_burn + n_win):
        v += 0.5 * dt * F @ Hi.T
        q += 0.5 * dt * v
        v = v @ Cdt.T + rng.normal(size=(n, 2)) @ Lo.T
        q += 0.5 * dt * v
        F = force(q)
        v += 0.5 * dt * F @ Hi.T
        if k >= n_burn:
            acc += q[:, PARTS[part]]
            stayed &= (q @ ae) * bit > 0.0
    return acc / n_win, stayed


def window_mean_variance(placement: str, part: str, t_window: float, k_w: float | None = None,
                         fs: float = FS) -> float:
    """Variance of the part's window-mean angle in the linearized body (sampled at fs)."""
    k_w = K_W[placement] if k_w is None else k_w
    s = linearized(placement, k_w)
    n_t = int(round(t_window * fs)) + 1
    C = np.zeros((1, 4))
    C[0, PARTS[part]] = 1.0
    S = checks.langevin_noise_cov(s, C, n_t)
    return float(S.sum() / n_t ** 2)


def check_simulation(cases=(("latch", "link"), ("latch", "rotor"), ("series", "link"),
                            ("series", "rotor")), t_window: float = 0.01, n: int = 2000,
                     seed: int = 0) -> dict:
    """Nonlinear simulation against the linearized predictions: the difference of the
    window-mean angle between the two bits, and its spread."""
    rows = {}
    for placement, part in cases:
        up, su = simulate_window_means(placement, part, +1, t_window, n=n, seed=seed)
        dn, sd = simulate_window_means(placement, part, -1, t_window, n=n, seed=seed + 1)
        u, d = up[su], dn[sd]                      # elements that kept their bit
        rows[(placement, part)] = dict(
            shift_sim=float(np.mean(u) - np.mean(d)),
            shift_pred=static_reading(placement, part)["shift"],
            sd_sim=float(np.sqrt(0.5 * (np.var(u) + np.var(d)))),
            sd_all=float(np.sqrt(0.5 * (np.var(up) + np.var(dn)))),
            sd_pred=float(np.sqrt(window_mean_variance(placement, part, t_window))),
            flipped=float(1.0 - 0.5 * (su.mean() + sd.mean())),
            sd_of_shift=float(np.sqrt((np.var(u) / u.size + np.var(d) / d.size))))
    return rows


WINDOWS = (0.0, 0.002, 0.005, 0.01, 0.02)
CASES = (("latch", "link"), ("latch", "rotor"), ("series", "link"), ("series", "rotor"))


def check_reading(windows=WINDOWS, cases=CASES) -> dict:
    """Bits each part reads about the bit from a window of its history (linearized body):
    exact d^2, and the snapshot-plus-rate approximation d^2 ~ d^2_snap + T rate."""
    out = {}
    for placement, part in cases:
        r = static_reading(placement, part)
        d2 = np.array([window_d2(placement, part, T) for T in windows])
        approx = r["snapshot_d2"] + np.asarray(windows) * r["rate_d2"]
        out[(placement, part)] = dict(windows=np.asarray(windows), d2=d2, approx=approx,
                                      bits=np.array([binary_gaussian_bits(x) for x in d2]),
                                      **r)
    return out

