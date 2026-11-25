"""Bodies in a flow: a foil in a (possibly vortical) stream, and a bird's gust suspension.

Both are linear reduced-order models built as `StateSpaceBody`: mechanics plus first-order
aerodynamic lag states. No CFD.

Noise and resolution: the flow's turbulence and vortices are treated as part of the
environment (input channels), not as noise. The resolution is the body's own structural
thermal noise at a declared sigma_T2, entering through its structural dampers. Aerodynamic
damping is a deterministic coupling to the flow. This is one answer to the open modeling
question recorded under thread T12, adopted so the benchmarks can run; it should be
revisited when they are first used for a study.
"""
from __future__ import annotations

import numpy as np

from .. import register
from ..linear import StateSpaceBody

# R. T. Jones's approximation of the Wagner function, and Sears's of the Kussner function.
WAGNER = ((0.165, 0.0455), (0.335, 0.3))
KUSSNER = ((0.5, 0.13), (0.5, 1.0))


@register("foil", "4-fluid",
          "Typical-section foil (plunge and pitch springs) in a stream, unsteady thin-airfoil "
          "aerodynamics with Wagner and Kussner lags; input: transverse gust.")
def foil(U: float = 6.0, b: float = 0.15, a: float = -0.4, x_alpha: float = 0.2,
         r_alpha2: float = 0.25, mu: float = 20.0, rho: float = 1.225,
         omega_h: float = 10.0, omega_alpha: float = 25.0, zeta: float = 0.02,
         sigma_T2: float = 1e-6) -> StateSpaceBody:
    """Theodorsen's typical section: plunge h (positive down) and pitch alpha (nose up) about
    an elastic axis at a*b from mid-chord; semi-chord b; mass ratio mu = m / (pi rho b^2).
    Circulatory lift 2 pi rho U b [phi * w_3/4] with w_3/4 = U alpha + h' + b(1/2 - a) alpha',
    gust lift 2 pi rho U b [psi * w_g], acting at the quarter chord; apparent-mass terms
    included. States: (h, alpha, h', alpha', two Wagner lags, two Kussner lags).
    Water instead of air: rho = 1000 with a smaller mu. The default stream (6 m/s) is below
    the flutter speed of the default section (9.08 m/s; `flutter_speed`). On the classic
    case of Hodges and Pierce (mu = 20, r^2 = 6/25, x_alpha = 1/10, sigma = 2/5, a = -1/5)
    the model flutters at U / (b omega_alpha) = 2.170."""
    pr = np.pi * rho * b ** 2
    m = mu * pr
    S = m * x_alpha * b
    I = m * r_alpha2 * b ** 2
    k_h, k_a = m * omega_h ** 2, I * omega_alpha ** 2
    b_h, b_a = 2 * zeta * m * omega_h, 2 * zeta * I * omega_alpha
    Mt = np.array([[m + pr, S - pr * b * a], [S - pr * b * a, I + pr * b ** 2 * (1 / 8 + a ** 2)]])
    Mi = np.linalg.inv(Mt)
    Ks = np.diag([k_h, k_a])
    Ds = np.diag([b_h, b_a])
    c_l = 2 * np.pi * rho * U * b
    arm = b * (0.5 + a)
    s_rate = U / b                          # reduced time s = U t / b
    A0 = 1.0 - WAGNER[0][0] - WAGNER[1][0]
    # w_3/4 = W @ (h, alpha, h', alpha')
    W = np.array([0.0, U, 1.0, b * (0.5 - a)])
    n = 8
    A = np.zeros((n, n))
    A[0, 2] = A[1, 3] = 1.0
    # Generalized aerodynamic force vector per unit lift: (-1 on h, +arm on alpha).
    e_L = np.array([-1.0, arm])
    # Circulatory lift = c_l (A0 w + z1 + z2); gust lift = c_l (g1 + g2).
    F_q = np.zeros((2, n))
    F_q[:, :4] += c_l * A0 * np.outer(e_L, W)
    F_q[:, 4] += c_l * e_L
    F_q[:, 5] += c_l * e_L
    F_q[:, 6] += c_l * e_L
    F_q[:, 7] += c_l * e_L
    # Apparent-mass velocity terms: -pi rho b^2 U alpha' on h, -pi rho b^3 U (1/2 - a) alpha'.
    F_q[0, 3] += -pr * U
    F_q[1, 3] += -pr * b * U * (0.5 - a)
    F_q[:, :2] -= Ks
    F_q[:, 2:4] -= Ds
    A[2:4, :] = Mi @ F_q
    for i, (Ai, bi) in enumerate(WAGNER):
        r = bi * s_rate
        A[4 + i, 4 + i] = -r
        A[4 + i, :4] += Ai * r * W
    for i, (Ai, ci) in enumerate(KUSSNER):
        r = ci * s_rate
        A[6 + i, 6 + i] = -r
    B_g = np.zeros((n, 1))
    for i, (Ai, ci) in enumerate(KUSSNER):
        B_g[6 + i, 0] = Ai * ci * s_rate
    GG = np.zeros((n, n))
    Bx = np.zeros((n, 2))
    Bx[2:4] = Mi
    GG = Bx @ (2 * sigma_T2 * Ds) @ Bx.T
    def sel(*idx):
        C = np.zeros((len(idx), n))
        for r_, c_ in enumerate(idx):
            C[r_, c_] = 1.0
        return C
    params = dict(U=U, b=b, a=a, x_alpha=x_alpha, r_alpha2=r_alpha2, mu=mu, rho=rho,
                  omega_h=omega_h, omega_alpha=omega_alpha, zeta=zeta, sigma_T2=sigma_T2)
    body = StateSpaceBody(A_=A, Bs={"gust": B_g}, GG_=GG,
                          parts={"plunge": sel(0), "pitch": sel(1), "structure": sel(0, 1)},
                          positions=sel(0, 1), sigma_T2=sigma_T2, name=f"foil U={U:g}",
                          params=params, build=foil)
    return body


def flutter_speed(U_max: float = 60.0, **kw) -> float:
    """Lowest stream speed at which the undamped (zeta = 0) foil becomes unstable."""
    from scipy.optimize import brentq
    kw = {**kw, "zeta": 0.0}
    g = lambda U: np.linalg.eigvals(foil(U=U, **kw).A).real.max()
    Us = np.linspace(0.2, U_max, 300)
    vals = np.array([g(U) for U in Us])
    i = int(np.argmax(vals > 0))
    if vals[i] <= 0:
        return float("inf")
    return float(brentq(g, Us[i - 1], Us[i]))


def karman_gust(n_t: int, dt: float, batch: int, U: float = 6.0, d: float = 0.05,
                St: float = 0.2, amp: float = 0.5, turbulence: float = 0.0, rng=None):
    """Transverse velocity seen by a body downstream of a cylinder of diameter d: the
    vortex street's shedding at f = St U / d with a random phase, plus optional white
    turbulence. Shape (n_t, batch, 1), for `simulate_linear` on the "gust" channel."""
    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(n_t) * dt
    f = St * U / d
    ph = rng.uniform(0, 2 * np.pi, batch)
    w = amp * np.sin(2 * np.pi * f * t[:, None] + ph[None, :])
    if turbulence > 0:
        w = w + turbulence * rng.standard_normal((n_t, batch))
    return w[:, :, None]


@register("bird", "4-fluid",
          "Bird in gusty air: body and wings joined by a compliant shoulder (a suspension); "
          "inputs: gust velocity or white turbulence through a Dryden-like filter.")
def bird(m_b: float = 0.5, m_w: float = 0.05, k_s: float = 50.0, b_s: float = 0.3,
         U: float = 10.0, rho: float = 1.225, S: float = 0.1, cla: float = 5.0,
         chord: float = 0.1, L_turb: float = 2.0, sigma_T2: float = 1e-6) -> StateSpaceBody:
    """Vertical motion only. The wings (effective mass m_w) move relative to the body (m_b)
    on a shoulder spring k_s and damper b_s; quasi-steady lift on the wings
    c (w_g - y_w') with c = rho U S cla / 2, the gust reaching the lift through Kussner's lag
    (reduced time with semi-chord chord/2). The body's height is a free drift, so the state
    uses the shoulder deflection and the two vertical velocities:
    (y_w - y_b, y_b', y_w', two Kussner lags, a turbulence filter state). A stiff shoulder
    (large k_s) is the rigid-winged comparison."""
    c = 0.5 * rho * U * S * cla
    s_rate = U / (0.5 * chord)
    n = 6
    A = np.zeros((n, n))
    # d/dt (y_w - y_b) = y_w' - y_b'
    A[0, 1], A[0, 2] = -1.0, 1.0
    # body: m_b y_b'' = k_s d + b_s (y_w' - y_b')
    A[1, 0] = k_s / m_b
    A[1, 1] = -b_s / m_b
    A[1, 2] = b_s / m_b
    # wing: m_w y_w'' = -k_s d - b_s (y_w' - y_b') - c y_w' + c (g1 + g2)
    A[2, 0] = -k_s / m_w
    A[2, 1] = b_s / m_w
    A[2, 2] = -(b_s + c) / m_w
    A[2, 3] = A[2, 4] = c / m_w
    # Kussner lags driven by the gust velocity w_g (a channel, or the filter state 5).
    for i, (Ai, ci) in enumerate(KUSSNER):
        r = ci * s_rate
        A[3 + i, 3 + i] = -r
        A[3 + i, 5] = Ai * r
    # Dryden-like first-order turbulence filter: tau w' = -w + white, tau = L/U.
    tau = L_turb / U
    A[5, 5] = -1.0 / tau
    B_gust = np.zeros((n, 1))
    for i, (Ai, ci) in enumerate(KUSSNER):
        B_gust[3 + i, 0] = Ai * ci * s_rate
    B_turb = np.zeros((n, 1))
    B_turb[5, 0] = 1.0 / tau
    # Thermal noise through the shoulder damper: force xi on the wing, -xi on the body.
    e = np.zeros((n, 1))
    e[1, 0] = -1.0 / m_b
    e[2, 0] = 1.0 / m_w
    GG = 2 * sigma_T2 * b_s * (e @ e.T)
    def sel(*idx):
        C = np.zeros((len(idx), n))
        for r_, c_ in enumerate(idx):
            C[r_, c_] = 1.0
        return C
    params = dict(m_b=m_b, m_w=m_w, k_s=k_s, b_s=b_s, U=U, rho=rho, S=S, cla=cla,
                  chord=chord, L_turb=L_turb, sigma_T2=sigma_T2)
    body = StateSpaceBody(A_=A, Bs={"gust": B_gust, "turbulence": B_turb}, GG_=GG,
                          parts={"shoulder": sel(0), "body_velocity": sel(1),
                                 "wing_velocity": sel(2)},
                          positions=sel(0), sigma_T2=sigma_T2, name=f"bird k_s={k_s:g}",
                          params=params, build=bird)
    body_accel = A[1].copy()                # y_b'' = A[1] @ x (no direct input term)
    object.__setattr__(body, "_accel_row", body_accel)
    return body


def body_acceleration_rms(body: StateSpaceBody, channel: str = "turbulence",
                          level: float = 1.0) -> float:
    """RMS of the body's vertical acceleration driven by white input of two-sided density
    `level` on `channel` (the measure of gust rejection): sqrt(c P c^T) with P the input's
    stationary covariance."""
    import scipy.linalg as sla
    Bc = body.B(channel)
    P = sla.solve_continuous_lyapunov(body.A, -level * Bc @ Bc.T)
    c = body._accel_row
    return float(np.sqrt(c @ P @ c))
