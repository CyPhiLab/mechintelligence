"""A mechanical bit: a bistable element in the Langevin body (thread T7).

One coordinate x with mass m in the double well V(x) = dE ((x/a)^2 - 1)^2, damping b and the
Langevin body's thermal force, so the Gibbs law is p(x, v) ~ exp(-(V(x) + m v^2 / 2) / sigma_T^2).
Units: lengths in a, masses in m, energies in sigma_T^2, so times are in sqrt(m a^2 / sigma_T^2)
and the barrier is beta = dE / sigma_T^2. gamma = b / m.

The environment writes a bit: either it pushed the element into the right well at t = 0 or it
did not. The question is how long the body keeps that bit.
"""
from __future__ import annotations

import numpy as np

#: Damping rate gamma = b / m used throughout, the same for the bistable and harmonic elements.
GAMMA = 2.0


def force(x, beta: float):
    """-V'(x) for V = beta (x^2 - 1)^2 in the book's units."""
    return -4.0 * beta * x * (x * x - 1.0)


def kramers_rate(beta: float, gamma: float) -> float:
    """Escape rate from one well (Kramers, moderate to strong damping, beta >> 1):
    r = omega_0 / (2 pi omega_b) (sqrt(gamma^2/4 + omega_b^2) - gamma/2) exp(-beta),
    with omega_0^2 = V''(1) = 8 beta and omega_b^2 = |V''(0)| = 4 beta."""
    w0, wb = np.sqrt(8.0 * beta), np.sqrt(4.0 * beta)
    return float(w0 / (2 * np.pi * wb) * (np.sqrt(gamma ** 2 / 4 + wb ** 2) - gamma / 2)
                 * np.exp(-beta))


def simulate_wells(beta: float, gamma: float, t_end: float, n: int = 4000, dt: float = 0.005,
                   n_out: int = 60, seed: int = 0):
    """Start n elements at rest at the bottom of the right well (a perfect write) and record
    the fraction found in the left well (x < 0) at n_out times, with the BAOAB integrator."""
    rng = np.random.default_rng(seed)
    x = np.ones(n)
    v = rng.normal(0.0, 1.0, n)                   # velocities from the Gibbs law (m = sigma = 1)
    c = np.exp(-gamma * dt)
    s = np.sqrt(1.0 - c * c)
    steps = int(round(t_end / dt))
    every = max(1, steps // n_out)
    times, frac = [], []
    F = force(x, beta)
    for k in range(1, steps + 1):
        v += 0.5 * dt * F
        x += 0.5 * dt * v
        v = c * v + s * rng.normal(0.0, 1.0, n)
        x += 0.5 * dt * v
        F = force(x, beta)
        v += 0.5 * dt * F
        if k % every == 0:
            times.append(k * dt)
            frac.append(float(np.mean(x < 0.0)))
    return np.array(times), np.array(frac)


def fitted_rate(times, frac, skip: float = 0.0) -> float:
    """Fit p(t) = (1 - exp(-2 r t)) / 2 by least squares on -log(1 - 2 p) = 2 r t."""
    keep = (times > skip) & (frac < 0.45)
    y = -np.log(1.0 - 2.0 * frac[keep])
    return float(np.sum(times[keep] * y) / np.sum(times[keep] ** 2) / 2.0)


def binary_entropy(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def bit_information(t, rate: float):
    """Bits the well holds about a perfectly written bit after time t: 1 - h((1 - e^{-2rt})/2)."""
    return 1.0 - binary_entropy(0.5 * (1.0 - np.exp(-2.0 * rate * np.asarray(t, float))))


def linear_information(t, gamma: float, beta: float, n_grid: int = 4001):
    """The same push recorded by a harmonic element with the curvature of one well (no barrier).

    The push moves the element by 2 (from one well's bottom to the other's, the same
    displacement the bistable element records). Its memory of the push is the separation of the
    two Gaussian state distributions (with and without the push) against the Gibbs covariance;
    for a binary input with equal priors, I = 1 - E[log2(1 + exp(-L))] over the log-likelihood
    ratio L, which for two Gaussians with Mahalanobis separation d is N(d^2/2, d^2).
    """
    k = 8.0 * beta                                # V''(1)
    A = np.array([[0.0, 1.0], [-k, -gamma]])
    out = []
    from scipy.linalg import expm
    for tt in np.atleast_1d(t):
        mean = expm(A * tt) @ np.array([2.0, 0.0])
        d2 = mean[0] ** 2 * k + mean[1] ** 2       # Gibbs covariance diag(1/k, 1)
        if d2 < 1e-14:
            out.append(0.0)
            continue
        d = np.sqrt(d2)
        L = np.linspace(d2 / 2 - 8 * d, d2 / 2 + 8 * d, n_grid)
        w = np.exp(-(L - d2 / 2) ** 2 / (2 * d2))
        w /= w.sum()
        out.append(float(1.0 - np.sum(w * np.log2(1.0 + np.exp(-L)))))
    return np.array(out)


def half_life_bistable(beta: float, gamma: float) -> float:
    """Time until the well holds half a bit: 1 - h(p) = 1/2 at p = 0.110, so
    t = -log(1 - 2 p) / (2 r)."""
    from scipy.optimize import brentq
    p = brentq(lambda q: binary_entropy(q) - 0.5, 1e-6, 0.5 - 1e-9)
    return float(-np.log(1.0 - 2.0 * p) / (2.0 * kramers_rate(beta, gamma)))


def half_life_linear(beta: float, gamma: float) -> float:
    """Time until the harmonic element holds half a bit about the same push."""
    from scipy.optimize import brentq
    f = lambda t: linear_information([t], gamma, beta)[0] - 0.5
    hi = 1.0
    while f(hi) > 0:
        hi *= 2.0
    return float(brentq(f, 0.0, hi, xtol=1e-6))


# ---------------------------------------------------------------------------------------
# Checks used by the tests and the research note
# ---------------------------------------------------------------------------------------

def check_kramers(betas=(3.0, 4.0, 5.0, 6.0), n: int = 4000, seed: int = 0,
                  gamma: float = GAMMA) -> dict:
    """Simulated escape rate against Kramers' formula."""
    rows = {}
    for beta in betas:
        g = gamma
        r = kramers_rate(beta, g)
        t, f = simulate_wells(beta, g, min(0.8 / r, 3000.0), n=n, seed=seed)
        rows[beta] = dict(kramers=r, simulated=fitted_rate(t, f, skip=5.0))
    return rows


def check_lifetimes(betas=(2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0), gamma: float = GAMMA) -> dict:
    """Half-lives of the bit in the bistable element and in a harmonic element."""
    out = dict(betas=np.asarray(betas), bistable=[], linear=[])
    for beta in betas:
        g = gamma
        out["bistable"].append(half_life_bistable(beta, g))
        out["linear"].append(half_life_linear(beta, g))
    return {k: np.asarray(v) for k, v in out.items()}


def check_window(s_list=(0.002, 0.01, 0.03), window: float = 0.02) -> dict:
    """Markov check on the rigid leg: a window of the whole state from s to s + window holds
    exactly what the state at s holds about the amplitude of a force impulse at t = 0."""
    from mechinfo.mechanics import energy_metric, zoh
    from passive_info.information import FS, SIGMA_A, SIGMA_ENERGY
    from passive_info.systems import proprioceptive
    from mibook import checks
    s = proprioceptive(readout="energy")
    n2 = 2 * s.n
    Ad, Bd = zoh(s.A, s.B_F, 1.0 / FS)
    P = SIGMA_ENERGY ** 2 * np.linalg.inv(energy_metric(s.K, s.H))
    rows = {}
    for t0 in s_list:
        k0 = int(round(t0 * FS))
        kw = int(round(window * FS))
        x = Bd[:, 0].copy()                        # state one step after a unit impulse
        means = []
        for k in range(1, k0 + kw + 1):
            if k >= k0:
                means.append(x.copy())
            x = Ad @ x
        M = np.concatenate(means)
        S = checks.langevin_noise_cov(s, np.eye(n2), len(means))
        I_win = 0.5 * np.log2(1 + SIGMA_A ** 2 * M @ np.linalg.solve(S, M))
        m0 = means[0]
        I_state = 0.5 * np.log2(1 + SIGMA_A ** 2 * m0 @ np.linalg.solve(P, m0))
        rows[t0] = dict(window=float(I_win), state=float(I_state))
    return rows
