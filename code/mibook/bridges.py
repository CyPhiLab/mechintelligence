"""Bridges between the book's measures and other approaches (research note of 2026-11-25).

Each `check_*` returns plain numbers that `tests/test_bridges.py` asserts on.

- `check_sampling_bound`: the per-sample signal-to-noise ratio g of the Langevin body never
  exceeds the continuous-time (Girsanov) value dt J D^+ J^T / (2 sigma_T^2), and reaches
  it as dt -> 0 with a relative error O(dt). This proves @prp-contact (thread T3).
- `check_memory_capacity`: linear memory capacity from reservoir computing and the book's
  Gaussian information are two functions of one noise-whitened Gramian spectrum.
- `check_excess_entropy`: the past-future information of the sampled thermal motion equals
  I(x_k; x_{k+1}), the closed form of MC_W.
- `check_series_rule`: the energy-share rule for a spring in series with the ground.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from mechinfo.mechanics import energy_metric, sym_sqrt
from passive_info.information import FS, SIGMA_ENERGY
from passive_info.systems import proprioceptive, series_elastic

SIGMA_T2 = SIGMA_ENERGY ** 2


# ---------------------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------------------

def mechanism(H: np.ndarray, D: np.ndarray, K: np.ndarray, J: np.ndarray):
    """State-space matrices (A, B) for H q'' + D q' + K q = J^T f."""
    n = H.shape[0]
    Hi = np.linalg.inv(H)
    A = np.block([[np.zeros((n, n)), np.eye(n)], [-Hi @ K, -Hi @ D]])
    B = np.vstack([np.zeros((n, J.shape[0])), Hi @ J.T])
    return A, B


def from_design(s):
    """(H, D, K, J) of a book design. D is read off A, J^T off B_F."""
    n = s.n
    D = -s.H @ s.A[n:, n:]
    JT = s.H @ s.B_F[n:, :1]
    return s.H, 0.5 * (D + D.T), s.K, JT.T


def zoh(A: np.ndarray, B: np.ndarray, dt: float):
    """Zero-order-hold discretization: A_d = e^{A dt}, B_d = int_0^dt e^{As} B ds."""
    n, m = B.shape
    M = sla.expm(np.block([[A, B], [np.zeros((m, n + m))]]) * dt)
    return M[:n, :n], M[:n, n:]


def noise_intensity(H, D, sigma_T2: float = SIGMA_T2) -> np.ndarray:
    """G G^T = diag(0, 2 sigma_T^2 H^-1 D H^-1), the thermal noise on the state."""
    n = H.shape[0]
    Hi = np.linalg.inv(H)
    GG = np.zeros((2 * n, 2 * n))
    GG[n:, n:] = 2.0 * sigma_T2 * Hi @ D @ Hi
    return GG


def discrete_noise(A: np.ndarray, GG: np.ndarray, dt: float) -> np.ndarray:
    """Q_d = int_0^dt e^{As} G G^T e^{A^T s} ds by Van Loan's method.

    Under the fluctuation-dissipation theorem this equals P - A_d P A_d^T (`gibbs_pair`),
    but the integral avoids the cancellation in that difference at small dt.
    """
    n = A.shape[0]
    M = sla.expm(np.block([[-A, GG], [np.zeros((n, n)), A.T]]) * dt)
    Q = M[n:, n:].T @ M[:n, n:]
    return 0.5 * (Q + Q.T)


def sampling_gain(H, D, K, J, dt: float, sigma_T2: float = SIGMA_T2) -> np.ndarray:
    """g = B_d^T Q_d^{-1} B_d: the per-sample signal-to-noise matrix of the Langevin body."""
    A, B = mechanism(H, D, K, J)
    _, Bd = zoh(A, B, dt)
    Q = discrete_noise(A, noise_intensity(H, D, sigma_T2), dt)
    return Bd.T @ np.linalg.solve(Q, Bd)


def continuum_gain(D, J, sigma_T2: float = SIGMA_T2) -> np.ndarray:
    """The Girsanov rate J D^+ J^T / (2 sigma_T^2), per unit time."""
    return J @ np.linalg.pinv(D) @ J.T / (2.0 * sigma_T2)


def chain(n: int, seed: int, damped: str = "all"):
    """A fixed-free chain of n masses, force at the free end, which has a ground damper.

    damped="all": every element has a damper (D full rank).
    damped="ends": only the first element and the ground damper at the tip (D singular,
    J^T still in the range of D). damped="none-at-tip": only the first element is damped,
    so for n >= 2 the force acts on a coordinate outside the range of D.
    """
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.5, 2.0, n)
    k = rng.uniform(1e2, 1e3, n)
    b = rng.uniform(0.5, 3.0, n)
    H = np.diag(m)
    K = np.zeros((n, n))
    D = np.zeros((n, n))
    for i in range(n):
        a = np.zeros(n)
        a[i] = 1.0
        if i > 0:
            a[i - 1] = -1.0
        K += k[i] * np.outer(a, a)
        if damped == "all" or (damped in ("ends", "none-at-tip") and i == 0):
            D += b[i] * np.outer(a, a)
    if damped != "none-at-tip":
        D[-1, -1] += rng.uniform(1.0, 4.0)
    J = np.zeros((1, n))
    J[0, -1] = 1.0
    return H, D, K, J


# ---------------------------------------------------------------------------------------
# Thread T3: the sampling bound
# ---------------------------------------------------------------------------------------

DTS = (1e-2, 1e-3, 1e-4, 1e-5)


def check_sampling_bound(dts=DTS) -> dict:
    """ratio = g / (dt * J D^+ J^T / 2 sigma_T^2) for book designs and random chains.

    The theorem says ratio <= 1 at every dt and 1 - ratio = O(dt).
    """
    cases = {
        "rigid": from_design(proprioceptive(readout="energy")),
        "SEA 70": from_design(series_elastic(70.0, readout="energy")),
        "SEA 1e4": from_design(series_elastic(1e4, readout="energy")),
        "chain 3, all damped": chain(3, 1, "all"),
        "chain 5, all damped": chain(5, 2, "all"),
        "chain 4, singular D": chain(4, 3, "ends"),
    }
    rows = {}
    for name, (H, D, K, J) in cases.items():
        target = continuum_gain(D, J).item()
        ratio = np.array([sampling_gain(H, D, K, J, dt).item() / (dt * target) for dt in dts])
        rows[name] = dict(ratio=ratio, rank_D=int(np.linalg.matrix_rank(D)), n=H.shape[0])
    # Outside the hypothesis: the force drives a coordinate with no damper of its own.
    # g/dt then grows without bound as dt -> 0 (Q_d becomes too ill-conditioned below 1e-3).
    H, D, K, J = chain(2, 4, "none-at-tip")
    dts_out = (1e-2, 3e-3, 1e-3)
    g = np.array([sampling_gain(H, D, K, J, dt).item() / dt for dt in dts_out])
    return dict(rows=rows, dts=np.array(dts), outside_dts=np.array(dts_out), outside_g_per_dt=g)


# ---------------------------------------------------------------------------------------
# Memory capacity and information share one spectrum
# ---------------------------------------------------------------------------------------

def check_memory_capacity(ks: float = 1e3, q: float = 1e4, n_delays: int = 4000) -> dict:
    """Linear memory capacity of the SEA's full state against its thermal noise.

    Input: an i.i.d. force sample u_k ~ N(0, q) held over each step. State: the signal part
    x_t = sum_j A_d^j B_d u_{t-1-j} plus stationary thermal noise of covariance P. The
    memory function C_j is the R^2 of the best linear readout of u_{t-1-j} from x_t.
    """
    s = series_elastic(ks, readout="energy")
    H, D, K, J = from_design(s)
    A, B = mechanism(H, D, K, J)
    Ad, Bd = zoh(A, B, 1.0 / FS)
    P = SIGMA_T2 * np.linalg.inv(energy_metric(K, H))
    b = Bd[:, 0]
    W = sla.solve_discrete_lyapunov(Ad, np.outer(b, b))     # controllability Gramian
    Sx = q * W + P
    Sxi = np.linalg.inv(Sx)
    C = np.empty(n_delays)
    I_delay = np.empty(n_delays)
    v = b.copy()
    for j in range(n_delays):
        c = q * v                                             # Cov(x_t, u_{t-1-j})
        rho2 = c @ Sxi @ c / q
        C[j] = rho2
        I_delay[j] = -0.5 * np.log(1.0 - rho2)
        v = Ad @ v
    Pih = np.linalg.inv(sym_sqrt(P))
    lam = np.linalg.eigvalsh(q * Pih @ W @ Pih)
    return dict(
        mc_sum=float(C.sum()), mc_spectrum=float(np.sum(lam / (1.0 + lam))),
        info_direct=float(0.5 * np.linalg.slogdet(np.eye(len(b)) + np.linalg.solve(P, q * W))[1]),
        info_spectrum=float(0.5 * np.sum(np.log1p(lam))),
        identity_err=float(np.max(np.abs(C - (1.0 - np.exp(-2.0 * I_delay))))),
        n_modes=len(b), lam_max=float(lam.max()), tail=float(C[-1]))


# ---------------------------------------------------------------------------------------
# MC_W is the excess entropy of the sampled thermal motion
# ---------------------------------------------------------------------------------------

def _block_cov(Ad: np.ndarray, P: np.ndarray, n_t: int) -> np.ndarray:
    n = P.shape[0]
    pw = [np.eye(n)]
    for _ in range(n_t):
        pw.append(Ad @ pw[-1])
    S = np.zeros((n_t * n, n_t * n))
    for k in range(n_t):
        for l in range(k + 1):
            blk = pw[k - l] @ P
            S[k * n:(k + 1) * n, l * n:(l + 1) * n] = blk
            S[l * n:(l + 1) * n, k * n:(k + 1) * n] = blk.T
    return S


def _gauss_mi(S: np.ndarray, na: int) -> float:
    ld = lambda M: np.linalg.slogdet(M)[1]
    return 0.5 * (ld(S[:na, :na]) + ld(S[na:, na:]) - ld(S))


def check_excess_entropy(windows=((1, 1), (2, 3), (4, 4))) -> dict:
    """I(past m samples; future n samples) of the thermal motion, against MC_W."""
    rows = {}
    for name, s in (("rigid", proprioceptive(readout="energy")),
                    ("SEA 70", series_elastic(70.0, readout="energy"))):
        H, D, K, J = from_design(s)
        A, _ = mechanism(H, D, K, J)
        Ad, _ = zoh(A, np.zeros((A.shape[0], 1)), 1.0 / FS)
        E = energy_metric(K, H)
        P = SIGMA_T2 * np.linalg.inv(E)
        Eh = sym_sqrt(E)
        sv = np.linalg.svd(Eh @ Ad @ np.linalg.inv(Eh), compute_uv=False)
        mcw = float(-0.5 * np.sum(np.log(1.0 - sv ** 2)))
        n = P.shape[0]
        mi = {}
        for m_past, n_fut in windows:
            S = _block_cov(Ad, P, m_past + n_fut)
            mi[(m_past, n_fut)] = _gauss_mi(S, m_past * n)
        rows[name] = dict(mcw_nats=mcw, past_future=mi)
    return rows


# ---------------------------------------------------------------------------------------
# A spring in series with the ground
# ---------------------------------------------------------------------------------------

def check_series_rule(kg: float = 1e6, kl_grid=(1e3, 1e4, 1e5, 1e6, 1e7), h: float = 1e-6) -> dict:
    """d log K_S / d log k_g against the ground spring's share of the stored energy."""
    rows = []
    for kl in kl_grid:
        ks = lambda g: 1.0 / (1.0 / g + 1.0 / kl)
        slope = (np.log(ks(kg * np.exp(h))) - np.log(ks(kg * np.exp(-h)))) / (2 * h)
        share = (1.0 / kg) / (1.0 / kg + 1.0 / kl)            # energy F^2/2k in the ground spring
        rows.append(dict(kl=kl, slope=float(slope), share=float(share)))
    return dict(rows=rows)
