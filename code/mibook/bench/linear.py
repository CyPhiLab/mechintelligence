"""Exact engine for linear bodies with Gaussian noise (thread T12).

A `LinearBody` is H q'' + D q' + K q = sum_c J_c^T u_c + noise, with named environment
input channels u_c, named parts (rows of the state read out), and named parameters that
rebuild the matrices. The default noise is the body's own thermal noise, covariance
2 sigma_T^2 D (fluctuation-dissipation); a body may declare another noise intensity.

Every window measure here runs in time linear in the window length. The key step is
`Filter.whiten`: the Kalman filter of the noise-only model maps any deterministic response
to its innovations, which is the same as multiplying by the inverse Cholesky factor of the
stacked noise covariance, without ever forming that matrix.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import scipy.linalg as sla

LOG2 = np.log(2.0)


# ---------------------------------------------------------------------------------------
# The body
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearBody:
    """A linear mechanism with named environment channels, parts and parameters.

    `inputs[name]` is J_c, of shape (n_c, n): the channel enters as J_c^T u_c.
    `parts[name]` is a selector C of shape (m, 2n) on the state x = (q, q').
    `noise_intensity`, if given, replaces the thermal G G^T (shape 2n x 2n).
    `build` and `params` let a parameter be changed: build(**params) -> LinearBody.
    """
    H: np.ndarray
    D: np.ndarray
    K: np.ndarray
    inputs: Mapping[str, np.ndarray]
    parts: Mapping[str, np.ndarray] = field(default_factory=dict)
    sigma_T2: float = 1e-4
    noise_intensity: np.ndarray | None = None
    name: str = "body"
    params: Mapping[str, float] = field(default_factory=dict)
    build: Callable[..., "LinearBody"] | None = None

    # -- sizes and matrices ---------------------------------------------------------------
    @property
    def n(self) -> int:
        return self.H.shape[0]

    @property
    def A(self) -> np.ndarray:
        n = self.n
        Hi = np.linalg.inv(self.H)
        return np.block([[np.zeros((n, n)), np.eye(n)], [-Hi @ self.K, -Hi @ self.D]])

    def B(self, channel: str) -> np.ndarray:
        J = np.atleast_2d(self.inputs[channel])
        return np.vstack([np.zeros((self.n, J.shape[0])), np.linalg.solve(self.H, J.T)])

    @property
    def GG(self) -> np.ndarray:
        if self.noise_intensity is not None:
            return self.noise_intensity
        n = self.n
        Hi = np.linalg.inv(self.H)
        GG = np.zeros((2 * n, 2 * n))
        GG[n:, n:] = 2.0 * self.sigma_T2 * Hi @ self.D @ Hi
        return 0.5 * (GG + GG.T)

    def energy_metric(self) -> np.ndarray:
        n = self.n
        E = np.zeros((2 * n, 2 * n))
        E[:n, :n] = self.K
        E[n:, n:] = self.H
        return E

    def stationary_cov(self) -> np.ndarray:
        """Solves A P + P A^T + G G^T = 0. Equals sigma_T^2 E^{-1} for thermal noise."""
        P = sla.solve_continuous_lyapunov(self.A, -self.GG)
        return 0.5 * (P + P.T)

    def part(self, name: str) -> np.ndarray:
        if name == "whole":
            return np.eye(2 * self.n)
        return np.atleast_2d(self.parts[name])

    def spectral_part(self, name: str) -> np.ndarray:
        """Selector for spectral rates. For "whole", the positions only: in continuous time
        the velocities are their derivatives and carry no further information, while
        including them makes the noise spectrum singular."""
        if name == "whole":
            return np.hstack([np.eye(self.n), np.zeros((self.n, self.n))])
        return self.part(name)

    def coordinate(self, i: int, velocity: bool = False) -> np.ndarray:
        """Selector for one position (or velocity) coordinate."""
        C = np.zeros((1, 2 * self.n))
        C[0, i + (self.n if velocity else 0)] = 1.0
        return C

    def with_params(self, **changes) -> "LinearBody":
        if self.build is None:
            raise ValueError(f"{self.name}: no build function, so parameters cannot change")
        return self.build(**{**self.params, **changes})

    def discretize(self, dt: float, channel: str | None = None) -> "Sampled":
        return Sampled.of(self, dt, channel)


def discrete_noise(A: np.ndarray, GG: np.ndarray, dt: float) -> np.ndarray:
    """Q_d = int_0^dt e^{As} G G^T e^{A^T s} ds.

    Van Loan's block exponential contains e^{-A dt}, which overflows for heavily damped fast
    modes. So the integral is taken over a substep short enough for Van Loan to be safe,
    and then doubled exactly: Q(2h) = Q(h) + e^{Ah} Q(h) e^{A^T h}.
    """
    n = A.shape[0]
    rate = float(np.max(np.abs(np.linalg.eigvals(A)))) if n else 0.0
    m = max(0, int(np.ceil(np.log2(max(rate * dt, 1.0)))))
    h = dt / 2 ** m
    M = sla.expm(np.block([[-A, GG], [np.zeros((n, n)), A.T]]) * h)
    F = M[n:, n:].T
    Q = F @ M[:n, n:]
    for _ in range(m):
        Q = Q + F @ Q @ F.T
        F = F @ F
    return 0.5 * (Q + Q.T)


def zoh(A: np.ndarray, B: np.ndarray, dt: float):
    """A_d = e^{A dt}, B_d = int_0^dt e^{As} B ds (input held over the step)."""
    n, m = B.shape
    M = sla.expm(np.block([[A, B], [np.zeros((m, n + m))]]) * dt)
    return M[:n, :n], M[:n, n:]


@dataclass(frozen=True)
class Sampled:
    """x_{k+1} = Ad x_k + Bd u_k + w_k, w_k ~ N(0, Qd), stationary covariance P."""
    Ad: np.ndarray
    Bd: np.ndarray | None
    Qd: np.ndarray
    P: np.ndarray
    dt: float

    @classmethod
    def of(cls, body: LinearBody, dt: float, channel: str | None = None) -> "Sampled":
        A = body.A
        Bd = None
        if channel is not None:
            Ad, Bd = zoh(A, body.B(channel), dt)
        else:
            Ad = sla.expm(A * dt)
        return cls(Ad=Ad, Bd=Bd, Qd=discrete_noise(A, body.GG, dt),
                   P=body.stationary_cov(), dt=dt)


# ---------------------------------------------------------------------------------------
# The Kalman whitening filter
# ---------------------------------------------------------------------------------------

@dataclass
class Filter:
    """Kalman filter of y_k = C x_k + v_k for the noise-only model, k = 0..N-1.

    Gains and innovation covariances do not depend on the data, so they are computed once
    and reused to whiten any number of signals.
    """
    gains: list
    chol: list                 # lower Cholesky factors of the innovation covariances
    logdet: np.ndarray         # log det S_k
    C: np.ndarray
    Ad: np.ndarray

    @classmethod
    def build(cls, s: Sampled, C: np.ndarray, n_t: int, eps: float = 0.0,
              P0: np.ndarray | None = None) -> "Filter":
        C = np.atleast_2d(C)
        m = C.shape[0]
        R = eps ** 2 * np.eye(m)
        Sig = s.P.copy() if P0 is None else P0.copy()
        gains, chol, logdet = [], [], np.empty(n_t)
        for k in range(n_t):
            S = C @ Sig @ C.T + R
            S = 0.5 * (S + S.T)
            L = np.linalg.cholesky(S)
            Kp = s.Ad @ Sig @ C.T @ np.linalg.inv(S)          # predictor gain
            gains.append(Kp)
            chol.append(L)
            logdet[k] = 2.0 * np.sum(np.log(np.diag(L)))
            Sig = s.Ad @ Sig @ s.Ad.T + s.Qd - Kp @ S @ Kp.T
            Sig = 0.5 * (Sig + Sig.T)
        return cls(gains=gains, chol=chol, logdet=logdet, C=C, Ad=s.Ad)

    def whiten(self, Y: np.ndarray) -> np.ndarray:
        """Whiten signals Y of shape (n_t, m, d): returns (n_t * m, d) innovations."""
        n_t, m, d = Y.shape
        xi = np.zeros((self.Ad.shape[0], d))
        out = np.empty((n_t, m, d))
        for k in range(n_t):
            z = Y[k] - self.C @ xi
            out[k] = sla.solve_triangular(self.chol[k], z, lower=True)
            xi = self.Ad @ xi + self.gains[k] @ z
        return out.reshape(n_t * m, d)


# ---------------------------------------------------------------------------------------
# Inputs: deterministic responses
# ---------------------------------------------------------------------------------------

def dct_basis(n_t: int, k: int) -> np.ndarray:
    """First k orthonormal DCT-II modes on n_t samples (the book's force basis)."""
    t = (np.arange(n_t) + 0.5) / n_t
    Bm = np.cos(np.pi * np.outer(t, np.arange(k)))
    return Bm / np.linalg.norm(Bm, axis=0, keepdims=True)


def responses(s: Sampled, C: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Readout of the response to each input profile.

    U has shape (n_t, n_u, d): d profiles of an n_u-dimensional input. The input at sample
    k drives x_{k+1}; the state starts at rest. Returns Y of shape (n_t, m, d).
    """
    C = np.atleast_2d(C)
    n_t, _, d = U.shape
    x = np.zeros((s.Ad.shape[0], d))
    Y = np.empty((n_t, C.shape[0], d))
    for k in range(n_t):
        Y[k] = C @ x
        x = s.Ad @ x + s.Bd @ U[k]
    return Y


# ---------------------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------------------

def bits_from_gram(G: np.ndarray) -> float:
    """1/2 log2 det(I + G)."""
    return 0.5 * np.linalg.slogdet(np.eye(G.shape[0]) + G)[1] / LOG2


def profile_bits(body: LinearBody, part: str | np.ndarray = "whole", channel: str = "force",
                 t_window: float = 0.085, fs: float = 4000.0, band_hz: float = 100.0,
                 sigma_a: float = 100.0, eps: float = 0.0) -> float:
    """Bits about a band-limited input profile with i.i.d. N(0, sigma_a^2) DCT coefficients.

    This is the book's measure (`langevin_bits`), computed in linear time.
    """
    C = body.part(part) if isinstance(part, str) else np.atleast_2d(part)
    s = body.discretize(1.0 / fs, channel)
    n_t = int(round(t_window * fs)) + 1
    k = max(1, int(round(2.0 * band_hz * t_window)))
    Phi = dct_basis(n_t, k)
    n_u = s.Bd.shape[1]
    if n_u != 1:
        raise NotImplementedError("profile_bits: one-dimensional channels only")
    Y = responses(s, C, Phi[:, None, :])
    Z = Filter.build(s, C, n_t, eps).whiten(Y)
    return bits_from_gram(sigma_a ** 2 * (Z.T @ Z))


def white_input_bits(body: LinearBody, part: str | np.ndarray = "whole",
                     channel: str = "force", q: float = 1.0, n_t: int = 400,
                     fs: float = 4000.0, eps: float = 0.0) -> float:
    """Bits about an i.i.d. N(0, q I) input sequence over n_t samples.

    Computed as h(Y) - h(Y | U) with two Kalman filters: one with the input folded into the
    process noise, one without. With no feedback from body to input this is also the
    directed information and the accumulated transfer entropy from input to readout.
    """
    C = body.part(part) if isinstance(part, str) else np.atleast_2d(part)
    s = body.discretize(1.0 / fs, channel)
    noisy = dataclasses.replace(s, Qd=s.Qd + q * s.Bd @ s.Bd.T)
    with_u = Filter.build(noisy, C, n_t, eps, P0=s.P)
    without = Filter.build(s, C, n_t, eps)
    return 0.5 * float(np.sum(with_u.logdet - without.logdet)) / LOG2


def one_step_te(body, part: str | np.ndarray = "whole", channel: str = "force", q: float = 1.0,
                fs: float = 4000.0, eps: float = 0.0) -> float:
    """One-step transfer entropy I(u_{k-1}; y_k | y_{k-1}) in bits per sample, stationary,
    for an i.i.d. N(0, q) input: the form used in sensorimotor studies. It equals the
    directed-information rate only when y is the whole (Markov) state."""
    C = body.part(part) if isinstance(part, str) else np.atleast_2d(part)
    s = body.discretize(1.0 / fs, channel)
    W = sla.solve_discrete_lyapunov(s.Ad, s.Bd @ s.Bd.T)
    Sx = s.P + q * W
    m = C.shape[0]
    R = eps ** 2 * np.eye(m)
    Syy = C @ Sx @ C.T + R
    S10 = C @ s.Ad @ Sx @ C.T
    Su1 = q * (C @ s.Bd).T                                  # Cov(u_{k-1}, y_k)
    p = s.Bd.shape[1]
    S = np.block([[q * np.eye(p), np.zeros((p, m)), Su1],
                  [np.zeros((m, p)), Syy, S10.T],
                  [Su1.T, S10, Syy]])
    ld = lambda idx: np.linalg.slogdet(S[np.ix_(idx, idx)])[1]
    iu, i0, i1 = list(range(p)), list(range(p, p + m)), list(range(p + m, p + 2 * m))
    return 0.5 * (ld(iu + i0) + ld(i1 + i0) - ld(iu + i0 + i1) - ld(i0)) / LOG2


def di_rate(body, part: str | np.ndarray = "whole", channel: str = "force", q: float = 1.0,
            fs: float = 4000.0, eps: float = 0.0, n1: int = 200, n2: int = 400) -> float:
    """Directed-information rate (bits per sample) about an i.i.d. N(0, q) input: the slope of
    the window information, which with no feedback is the information itself."""
    a = white_input_bits(body, part, channel, q, n1, fs, eps)
    b = white_input_bits(body, part, channel, q, n2, fs, eps)
    return (b - a) / (n2 - n1)


# -- Spectral (infinite-window) rates ------------------------------------------------------

def _transfer(body: LinearBody, w: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """C (i w I - A)^{-1} B on a frequency grid: shape (len(w), m, p)."""
    A = body.A
    I = np.eye(A.shape[0])
    return np.stack([C @ np.linalg.solve(1j * wi * I - A, B) for wi in w])


def spectra(body: LinearBody, C: np.ndarray, w: np.ndarray,
            channel: str | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Noise spectral density of C x, and the channel's transfer function, on grid w."""
    GG = body.GG
    lam, V = np.linalg.eigh(GG)
    keep = lam > lam.max() * 1e-14
    G = V[:, keep] * np.sqrt(lam[keep])
    Tn = _transfer(body, w, G, C)
    Sn = np.einsum("wij,wkj->wik", Tn, Tn.conj())
    Tu = _transfer(body, w, body.B(channel), C) if channel else None
    return Sn, Tu


def input_rate(body: LinearBody, part: str | np.ndarray = "whole", channel: str = "force",
               band_hz: float = 100.0, level: float = 1.0, n_w: int = 2000) -> float:
    """Information rate (bits/s) about a stationary input with flat two-sided spectral
    density `level` on |f| < band_hz:  int_0^B log2 det(I + Sn^-1 T level T^*) df."""
    C = body.spectral_part(part) if isinstance(part, str) else np.atleast_2d(part)
    f = np.linspace(0.0, band_hz, n_w)
    Sn, Tu = spectra(body, C, 2 * np.pi * f, channel)
    vals = np.empty(n_w)
    for i in range(n_w):
        M = np.eye(Sn.shape[1]) + level * np.linalg.solve(Sn[i], Tu[i] @ Tu[i].conj().T)
        vals[i] = np.linalg.slogdet(M)[1].real / LOG2
    return float(np.trapezoid(vals, f))


def observed_spectrum(body: LinearBody, C: np.ndarray, w: np.ndarray,
                      channel: str | None = None, level: float = 0.0) -> np.ndarray:
    """Spectral density of the readout: thermal noise plus, if a channel is given, a
    stationary input with flat two-sided density `level` on that channel."""
    Sn, Tu = spectra(body, C, w, channel)
    if channel is None or level == 0.0:
        return Sn
    return Sn + level * np.einsum("wij,wkj->wik", Tu, Tu.conj())


def fisher_rate(body: LinearBody, param: str, part: str | np.ndarray = "whole",
                band_hz: float = 100.0, rel_step: float = 1e-4, n_w: int = 2000,
                log: bool = True, channel: str | None = None, level: float = 0.0) -> float:
    """Whittle Fisher-information rate (per second) about a parameter, on |f| < band_hz:
    int_0^B tr[(S^-1 dS)^2] df, with S the readout spectrum (`observed_spectrum`).

    With no input, only the body's own noise reveals the parameter (the parameter notes).
    A property that shapes how an input enters, such as a contact point, is revealed only
    through that input: pass `channel` and `level`. `log=True` gives information about
    log(theta); otherwise about theta itself.
    """
    C_of = ((lambda b: b.spectral_part(part)) if isinstance(part, str)
            else (lambda b: np.atleast_2d(part)))
    theta = body.params[param]
    h = rel_step if log else rel_step * max(abs(theta), 1e-12)
    lo = body.with_params(**{param: theta * np.exp(-h) if log else theta - h})
    hi = body.with_params(**{param: theta * np.exp(h) if log else theta + h})
    f = np.linspace(0.0, band_hz, n_w)
    w = 2 * np.pi * f
    S0 = observed_spectrum(body, C_of(body), w, channel, level)
    Sl = observed_spectrum(lo, C_of(lo), w, channel, level)
    Sh = observed_spectrum(hi, C_of(hi), w, channel, level)
    vals = np.empty(n_w)
    for i in range(n_w):
        X = np.linalg.solve(S0[i], (Sh[i] - Sl[i]) / (2 * h))
        vals[i] = np.trace(X @ X).real
    return float(np.trapezoid(vals, f))


# -- Memory measures -------------------------------------------------------------------------

def memory_capacity(body: LinearBody, part: str | np.ndarray = "whole", channel: str = "force",
                    q: float = 1.0, fs: float = 4000.0) -> dict:
    """Linear memory capacity and snapshot information about an i.i.d. input history.

    Both come from the spectrum of the noise-whitened controllability Gramian of the readout
    (Prop. "Memory capacity and information share one spectrum"). Scalar channels.
    """
    C = body.part(part) if isinstance(part, str) else np.atleast_2d(part)
    s = body.discretize(1.0 / fs, channel)
    W = sla.solve_discrete_lyapunov(s.Ad, s.Bd @ s.Bd.T)
    N = C @ s.P @ C.T
    Sg = q * C @ W @ C.T
    lam = np.clip(sla.eigvalsh(Sg, N), 0.0, None)
    return dict(mc=float(np.sum(lam / (1 + lam))), nats=float(0.5 * np.sum(np.log1p(lam))),
                lam=lam)


def mc_w(body: LinearBody, fs: float = 4000.0) -> float:
    """Zahedi-Ay MC_W of the passive body, I(x_k; x_{k+1}) in bits (= excess entropy)."""
    s = body.discretize(1.0 / fs)
    # I = 1/2 log det P - 1/2 log det Qd, the one-step Gaussian channel from x_k.
    return 0.5 * (np.linalg.slogdet(s.P)[1] - np.linalg.slogdet(s.Qd)[1]) / LOG2


def empowerment(body: LinearBody, actuator: str, part: str | np.ndarray = "whole",
                horizon: int = 40, power: float = 1.0, fs: float = 4000.0,
                eps: float = 0.0) -> float:
    """n-step empowerment (bits): capacity from an actuator sequence to the readout at the
    horizon, under total input energy sum |a_k|^2 <= power, by water-filling.

    Channel: y = C x_n = C [A^{n-1}B ... B] a + noise, noise covariance C P_n C^T + eps^2 I,
    with P_n the covariance the thermal noise builds from a known start in n steps.
    """
    C = body.part(part) if isinstance(part, str) else np.atleast_2d(part)
    s = body.discretize(1.0 / fs, actuator)
    cols, M = [], s.Bd
    Pn = np.zeros_like(s.Qd)
    for _ in range(horizon):
        cols.append(C @ M)
        M = s.Ad @ M
        Pn = s.Ad @ Pn @ s.Ad.T + s.Qd
    Gn = np.hstack(cols[::-1])
    Nn = C @ Pn @ C.T + eps ** 2 * np.eye(C.shape[0])
    Li = np.linalg.inv(np.linalg.cholesky(Nn))
    g2 = np.linalg.svd(Li @ Gn, compute_uv=False) ** 2
    g2 = g2[g2 > g2.max() * 1e-14]
    # water-filling: p_i = (nu - 1/g2_i)^+, sum p_i = power
    inv = np.sort(1.0 / g2)
    for k in range(len(inv), 0, -1):
        nu = (power + inv[:k].sum()) / k
        if nu > inv[k - 1]:
            break
    p = np.clip(nu - 1.0 / g2, 0.0, None)
    return float(0.5 * np.sum(np.log2(1.0 + p * g2)))


# ---------------------------------------------------------------------------------------
# General state-space bodies (mechanics plus first-order states, such as aerodynamic lags)
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class StateSpaceBody:
    """x' = A x + sum_c B_c u_c + G xi. Same interface as `LinearBody` for every measure.

    Use it when the body has states that are not mechanical positions and velocities, such
    as the lag states of unsteady aerodynamics or a shaping filter for turbulence. `positions`
    selects the rows used for spectral rates of the whole body (rows whose noise spectrum is
    not singular). The noise intensity GG must be declared.
    """
    A_: np.ndarray
    Bs: Mapping[str, np.ndarray]
    GG_: np.ndarray
    parts: Mapping[str, np.ndarray] = field(default_factory=dict)
    positions: np.ndarray | None = None
    sigma_T2: float = 1e-4
    name: str = "state-space body"
    params: Mapping[str, float] = field(default_factory=dict)
    build: Callable[..., "StateSpaceBody"] | None = None

    @property
    def A(self) -> np.ndarray:
        return self.A_

    @property
    def GG(self) -> np.ndarray:
        return self.GG_

    @property
    def dim(self) -> int:
        return self.A_.shape[0]

    def B(self, channel: str) -> np.ndarray:
        return np.atleast_2d(self.Bs[channel]).reshape(self.dim, -1)

    def stationary_cov(self) -> np.ndarray:
        P = sla.solve_continuous_lyapunov(self.A_, -self.GG_)
        return 0.5 * (P + P.T)

    def part(self, name: str) -> np.ndarray:
        if name == "whole":
            return np.eye(self.dim)
        return np.atleast_2d(self.parts[name])

    def spectral_part(self, name: str) -> np.ndarray:
        if name == "whole":
            if self.positions is None:
                raise ValueError(f"{self.name}: declare `positions` for whole-body spectra")
            return np.atleast_2d(self.positions)
        return self.part(name)

    def with_params(self, **changes) -> "StateSpaceBody":
        if self.build is None:
            raise ValueError(f"{self.name}: no build function, so parameters cannot change")
        return self.build(**{**self.params, **changes})

    def discretize(self, dt: float, channel: str | None = None) -> Sampled:
        return Sampled.of(self, dt, channel)


def simulate_linear(body, n_t: int, dt: float, batch: int, u: np.ndarray | None = None,
                    channel: str | None = None, parts=("whole",), rng=None,
                    stationary: bool = True) -> dict:
    """Exact sampled simulation of any linear body: x_{k+1} = Ad x_k + Bd u_k + w_k.

    `u` has shape (n_t, batch, n_u). Starts from the stationary law (or rest). Returns
    {part: (n_t, batch, m)} with the readout taken before each input sample acts.
    """
    rng = np.random.default_rng() if rng is None else rng
    s = body.discretize(dt, channel)
    d = s.Ad.shape[0]
    lam, V = np.linalg.eigh(s.Qd)
    Lq = V * np.sqrt(np.clip(lam, 0.0, None))
    lam, V = np.linalg.eigh(s.P)
    Lp = V * np.sqrt(np.clip(lam, 0.0, None))
    x = rng.standard_normal((batch, d)) @ Lp.T if stationary else np.zeros((batch, d))
    Cs = {p: body.part(p) for p in parts}
    out = {p: np.empty((n_t, batch, C.shape[0])) for p, C in Cs.items()}
    for k in range(n_t):
        for p, C in Cs.items():
            out[p][k] = x @ C.T
        x = x @ s.Ad.T + rng.standard_normal((batch, d)) @ Lq.T
        if u is not None:
            x = x + u[k] @ s.Bd.T
    return out


# ---------------------------------------------------------------------------------------
# Exact finite-window Fisher information, in linear time (thread T11 check)
# ---------------------------------------------------------------------------------------

def window_kl(b0, b1, part, n_t: int, fs: float, eps: float = 0.0) -> float:
    """KL(p0 || p1) in nats between the laws of the part's sampled thermal motion over n_t
    samples under bodies b0 and b1, both started from their stationary laws.

    KL = 1/2 [tr(S1^-1 S0) - N m + log det S1 - log det S0] for the stacked covariances.
    The log-determinants come from each body's own Kalman filter. The trace is the expected
    squared whitened innovation of body 1's filter run on body 0's data, computed exactly by
    propagating the joint covariance of body 0's state and body 1's estimate. Linear in n_t.
    """
    C = b0.part(part) if isinstance(part, str) else np.atleast_2d(part)
    s0 = b0.discretize(1.0 / fs)
    s1 = b1.discretize(1.0 / fs)
    m = C.shape[0]
    R = eps ** 2 * np.eye(m)
    f0 = Filter.build(s0, C, n_t, eps)
    f1 = Filter.build(s1, C, n_t, eps)
    d = s0.Ad.shape[0]
    Z = np.zeros((2 * d, 2 * d))
    Z[:d, :d] = s0.P
    M = np.hstack([C, -C])
    total = 0.0
    for k in range(n_t):
        Ce = M @ Z @ M.T + R
        L1 = f1.chol[k]
        X = sla.solve_triangular(L1, Ce, lower=True)
        X = sla.solve_triangular(L1, X.T, lower=True)
        total += np.trace(X) - m + f1.logdet[k] - f0.logdet[k]
        K1 = f1.gains[k]
        F = np.block([[s0.Ad, np.zeros((d, d))], [K1 @ C, s1.Ad - K1 @ C]])
        N = np.zeros((2 * d, 2 * d))
        N[:d, :d] = s0.Qd
        N[d:, d:] = K1 @ R @ K1.T
        Z = F @ Z @ F.T + N
        Z = 0.5 * (Z + Z.T)
    return 0.5 * total


def window_fisher(body, param: str, part="whole", n_t: int = 400, fs: float = 4000.0,
                  eps: float = 0.0, rel_step: float = 1e-3, log: bool = True) -> float:
    """Exact Fisher information about a parameter from the part's thermal motion over n_t
    samples: [KL(theta || theta + h) + KL(theta || theta - h)] / h^2, to second order in h."""
    theta = body.params[param]
    h = rel_step if log else rel_step * max(abs(theta), 1e-12)
    up = body.with_params(**{param: theta * np.exp(h) if log else theta + h})
    dn = body.with_params(**{param: theta * np.exp(-h) if log else theta - h})
    return (window_kl(body, up, part, n_t, fs, eps) + window_kl(body, dn, part, n_t, fs, eps)) / h ** 2
