"""A thermal bath with a finite correlation time: every damper relaxes in time tau.

Each damper becomes a Maxwell element, a damper in series with a spring k = b / tau. The
thermal noise sits inside the element, on its internal extension s, and reaches the masses only
through the spring, so the velocities are smooth. Writing the damping matrix as
D = sum_i lam_i u_i u_i^T (its eigen-decomposition), the body with internal states s_i is

    H qdd = -K q - sum_i (lam_i / tau) u_i s_i + J^T f,
    ds_i  = (u_i^T qd - s_i / tau) dt - dW_i / lam_i,   E[dW_i^2] = 2 sigma_T^2 lam_i dt.

The memory kernel is D e^{-t/tau} / tau whatever decomposition is used, so tau is the only new
parameter, and tau -> 0 gives back the Langevin body with white noise. The stored energy gains
the springs' terms sum_i (lam_i / tau) s_i^2 / 2, and the Gibbs law is Gaussian with covariance
sigma_T^2 diag(K^{-1}, H^{-1}, tau / lam_i).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from mechinfo.mechanics import bits_from_gram
from passive_info.information import FS, SIGMA_A, SIGMA_ENERGY

from mibook import checks


def extend(sys, tau: float, sigma_T2: float | None = None):
    """The bath-extended linear system. Returns (A, B, Gc, P, n, m): drift, force input,
    noise intensity (dx = A x dt + B f dt + noise, E[noise noise^T] = Gc dt), the stationary
    Gibbs covariance, the number of mechanical coordinates and of bath states."""
    sigma_T2 = SIGMA_ENERGY ** 2 if sigma_T2 is None else sigma_T2
    n = sys.n
    D = checks._damping(sys)
    lam, U = np.linalg.eigh(0.5 * (D + D.T))
    keep = lam > 1e-12 * lam.max()
    lam, U = lam[keep], U[:, keep]
    m = lam.size
    Hi = np.linalg.inv(sys.H)
    N = 2 * n + m
    A = np.zeros((N, N))
    A[:n, n:2 * n] = np.eye(n)
    A[n:2 * n, :n] = -Hi @ sys.K
    A[n:2 * n, 2 * n:] = -Hi @ U * (lam / tau)
    A[2 * n:, n:2 * n] = U.T
    A[2 * n:, 2 * n:] = -np.eye(m) / tau
    B = np.zeros((N, sys.B_F.shape[1]))
    B[:2 * n] = sys.B_F
    Gc = np.zeros((N, N))
    Gc[2 * n:, 2 * n:] = np.diag(2.0 * sigma_T2 / lam)
    P = sla.block_diag(sigma_T2 * np.linalg.inv(sys.K), sigma_T2 * Hi, np.diag(sigma_T2 * tau / lam))
    return A, B, Gc, P, n, m


def lyapunov_residual(sys, tau: float) -> float:
    """|A P + P A^T + Gc| / |Gc|: zero when the Gibbs law is the stationary law (FDT holds)."""
    A, _, Gc, P, _, _ = extend(sys, tau)
    return float(np.abs(A @ P + P @ A.T + Gc).max() / np.abs(Gc).max())


def discretize(A, Gc, dt: float):
    """A_d = e^{A dt} and Q_d = int_0^dt e^{As} Gc e^{A^T s} ds, by Van Loan's method."""
    N = A.shape[0]
    M = np.block([[-A, Gc], [np.zeros((N, N)), A.T]]) * dt
    E = sla.expm(M)
    Ad = E[N:, N:].T
    Qd = Ad @ E[:N, N:]
    return Ad, 0.5 * (Qd + Qd.T)


def whittle_rate(build, tau: float, rows, fs: float = FS, velocities: bool = True,
                 dtheta: float = 1e-3, n_freq: int = 3000, white: bool = False,
                 band_hz: float | None = None) -> float:
    """Long-window Fisher information per second about theta, observing the positions in `rows`
    (and their velocities) of the body build(theta) with a bath of correlation time tau.

    `white=True` uses the white-noise Langevin body instead. `band_hz` limits the integral to
    frequencies below that band (the information about motion resolved up to band_hz). The integral over the sampled
    process's frequencies uses a geometric grid, so that low-frequency features stay resolved
    at high sampling rates.
    """
    def model(theta):
        s = build(theta)
        if white:
            n = s.n
            D = checks._damping(s)
            Hi = np.linalg.inv(s.H)
            Gc = np.zeros((2 * n, 2 * n))
            Gc[n:, n:] = 2.0 * SIGMA_ENERGY ** 2 * Hi @ D @ Hi
            return s.A, Gc, n
        A, _, Gc, _, n, _ = extend(s, tau)
        return A, Gc, n

    def spectra(theta, nus):
        A, Gc, n = model(theta)
        Ad, Qd = discretize(A, Gc, 1.0 / fs)
        idx = list(rows) + ([n + r for r in rows] if velocities else [])
        C = np.eye(A.shape[0])[idx]
        out = []
        I = np.eye(A.shape[0])
        for nu in nus:
            R = np.linalg.inv(np.exp(1j * nu) * I - Ad)
            out.append(C @ R @ Qd @ R.conj().T @ C.T)
        return out

    top = np.pi if band_hz is None else min(np.pi, 2 * np.pi * band_hz / fs)
    nus = np.geomspace(2 * np.pi * 0.05 / fs, top, n_freq)
    S0, Sp, Sm = spectra(0.0, nus), spectra(dtheta, nus), spectra(-dtheta, nus)
    vals = np.array([np.real(np.trace(np.linalg.matrix_power(
        np.linalg.solve(a, (b - c) / (2 * dtheta)), 2))) for a, b, c in zip(S0, Sp, Sm)])
    # (1 / 2 pi) int_{-pi}^{pi} (1/2) tr[...] dnu per sample = (1 / 2 pi) int_0^pi tr dnu
    return float(np.trapezoid(vals, nus) / (2 * np.pi) * fs)


def force_bits(sys, tau: float, C, eps: float = 0.0, band_hz: float = 100.0,
               fs: float = FS) -> float:
    """Bits about a force profile held by C x_q (x_q = positions and velocities), with the bath."""
    from passive_info.information import response_matrix
    import dataclasses
    A, B, Gc, P, n, m = extend(sys, tau)
    C = np.atleast_2d(C)
    Cx = np.hstack([C, np.zeros((C.shape[0], m))])
    M = response_matrix(dataclasses.replace(sys, C=C), fs=fs, band_hz=band_hz)
    mm = C.shape[0]
    n_t = M.shape[0] // mm
    Ad, _ = discretize(A, Gc, 1.0 / fs)
    powers = [np.eye(A.shape[0])]
    for _ in range(n_t):
        powers.append(Ad @ powers[-1])
    S = np.zeros((n_t * mm, n_t * mm))
    CP = Cx @ P
    for k in range(n_t):
        for l in range(k + 1):
            blk = Cx @ powers[k - l] @ CP.T
            S[k * mm:(k + 1) * mm, l * mm:(l + 1) * mm] = blk
            S[l * mm:(l + 1) * mm, k * mm:(k + 1) * mm] = blk.T
    S += eps ** 2 * np.eye(n_t * mm)
    Z = np.linalg.solve(np.linalg.cholesky(S), M)
    return bits_from_gram(SIGMA_A ** 2 * (Z.T @ Z))


# ---------------------------------------------------------------------------------------
# Checks used by the tests and the research note on payload mass
# ---------------------------------------------------------------------------------------

BANDS = (100.0, 400.0, 1600.0, 3200.0)


def check_bath(taus=(1e-6, 1e-5, 1e-4)) -> dict:
    """The bath is consistent with fluctuation-dissipation and gives back the white body."""
    from mibook import parameters as pm
    s = pm.leg_mass(None, 0.0)
    s70 = pm.leg_mass(70.0, 0.0)
    return dict(lyapunov=max(lyapunov_residual(s, 1e-4), lyapunov_residual(s70, 1e-4)),
                white=checks.langevin_bits(s, np.eye(2), 0.0),
                force={t: force_bits(s, t, np.eye(2)) for t in taus})


def check_band(bands=BANDS, fs: float = 64000.0, n_freq: int = 3000) -> dict:
    """Fisher information per second on the rigid leg from positions resolved up to each band:
    the link mass (white noise, and baths with tau = 0.1 and 1 ms), the ground damping and the
    ground stiffness (white noise)."""
    from mibook import parameters as pm
    from mibook import propagation as pg
    mass = lambda t: pm.leg_mass(None, t)
    damp = lambda t: pm.leg_at(None, 0.0, pg.B_CONTACT * np.exp(t))
    stiff = lambda t: pm.leg_at(None, t)

    def run(build, tau, white):
        return np.array([whittle_rate(build, tau, [0], fs, velocities=False, white=white,
                                      band_hz=b, n_freq=n_freq) for b in bands])

    return dict(bands=np.asarray(bands),
                mass_white=run(mass, 0.0, True), mass_tau4=run(mass, 1e-4, False),
                mass_tau3=run(mass, 1e-3, False), damp_white=run(damp, 0.0, True),
                stiff_white=run(stiff, 0.0, True),
                eps=pm.mass_velocity_snapshot(None)["eps"])
