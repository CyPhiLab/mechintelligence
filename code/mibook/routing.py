"""Routing efficiency: how much of the body's information about the contact force reaches a sensor.

The setting is the Langevin body of the book: thermal noise injected through every damper,
with intensity set by the fluctuation-dissipation theorem. The contact damper acts through
the same Jacobian as the contact force, so its noise enters exactly like the force. A sensor
therefore sees the force through an *effective contact damping* b_eff(w) >= b_c, and the
spectral routing efficiency is rho(w) = b_c / b_eff(w).

Every function here takes a `passive_info.systems.Passive` (or one built the same way) and
a readout row or matrix `C` acting on the state x_q = (q, qdot).
"""
from __future__ import annotations

import numpy as np

from mechinfo.mechanics import ground_damping
from passive_info.information import FS, SIGMA_A, SIGMA_ENERGY, SIGMA_THETA
from passive_info.systems import (I_ROT, K_I, M_L, R, ZETA_CONTACT, ZETA_SERIES,
                                  proprioceptive, reference_m_e, series_elastic)

from mibook import checks

#: The contact damper b_i of the ground model (N s/m); identical for every design.
B_CONTACT = float(ground_damping(K_I, reference_m_e(), ZETA_CONTACT))


def damping_matrix(sys) -> np.ndarray:
    """D, recovered from A: the lower-right block of A is -H^{-1} D."""
    n = sys.n
    return -sys.H @ sys.A[n:, n:]


def internal_damping(sys, b_contact: float = B_CONTACT) -> np.ndarray:
    """D_int = D - b_c J^T J: every damper except the one at the contact.

    Clipped to its positive-semidefinite part, with eigenvalues below 1e-12 of the scale of
    D treated as zero, so that round-off in a body with no internal damper gives exactly 0.
    """
    J = np.atleast_2d(sys.J)
    D = damping_matrix(sys)
    Dint = D - b_contact * (J.T @ J)
    lam, V = np.linalg.eigh(0.5 * (Dint + Dint.T))
    lam[lam < 1e-12 * max(np.abs(D).max(), 1e-300)] = 0.0
    return (V * lam) @ V.T


def effective_contact_damping(sys, C, omegas, eps: float = 0.0,
                              sigma_T2: float | None = None, dt: float = 1.0 / FS,
                              b_contact: float = B_CONTACT) -> np.ndarray:
    """b_eff(w) = b_c + 1 / (2 sigma_T^2 q(w)), with q = h^* M^+ h.

    h(w) = C (iw - A)^{-1} B_F is the sensor's response to the contact force. M(w) is the
    noise the sensor sees from everything except the contact damper:
    M = 2 sigma_T^2 G D_int G^* + eps^2 dt I, with G = C (iw - A)^{-1} [0; H^{-1}]. The
    readout noise eps (per sample, variance eps^2) has two-sided PSD eps^2 dt. When h is
    outside the range of M the sensor sees the force free of internal noise, q is infinite,
    and b_eff = b_c.
    """
    sigma_T2 = SIGMA_ENERGY ** 2 if sigma_T2 is None else sigma_T2
    C = np.atleast_2d(C)
    n = sys.n
    Dint = internal_damping(sys, b_contact)
    Bxi = np.vstack([np.zeros((n, n)), np.linalg.inv(sys.H)])
    m = C.shape[0]
    out = []
    for w in np.atleast_1d(omegas):
        Rw = np.linalg.inv(1j * w * np.eye(2 * n) - sys.A)
        h = (C @ Rw @ sys.B_F).reshape(m)
        G = C @ Rw @ Bxi
        M = 2.0 * sigma_T2 * (G @ Dint @ G.conj().T) + eps ** 2 * dt * np.eye(m)
        Mp = np.linalg.pinv(M, rcond=1e-12, hermitian=True)
        resid = h - M @ (Mp @ h)
        if np.linalg.norm(resid) > 1e-9 * max(np.linalg.norm(h), 1e-300):
            out.append(b_contact)
            continue
        q = float(np.real(h.conj() @ Mp @ h))
        out.append(b_contact + (np.inf if q <= 0 else 1.0 / (2.0 * sigma_T2 * q)))
    return np.array(out)


def spectral_routing(sys, C, omegas, **kw) -> np.ndarray:
    """rho(w) = b_c / b_eff(w), in [0, 1]."""
    return B_CONTACT / effective_contact_damping(sys, C, omegas, **kw)


def sea_routing_closed_form(ks: float, omega, sensor: str,
                            zeta_series: float = ZETA_SERIES) -> np.ndarray:
    """Closed-form rho(w) for the clamped SEA with a noiseless motor or joint encoder.

    motor: rho = 1 / (1 + (b_s/b_i) |k_i r^2 - m_j w^2 + i w b_i r^2|^2 / (r^2 |k_s + i b_s w|^2))
    joint: rho = 1 / (1 + (b_s/b_i) I^2 w^4 / (r^2 |k_s - I w^2 + i b_s w|^2))
    """
    w = np.asarray(omega, float)
    b_s = 2.0 * zeta_series * np.sqrt(ks * I_ROT)
    m_j = M_L * R ** 2
    c = ks + 1j * b_s * w
    if sensor == "motor":
        z_g = K_I * R ** 2 - m_j * w ** 2 + 1j * w * B_CONTACT * R ** 2
        ratio = np.abs(z_g) ** 2 / (R ** 2 * np.abs(c) ** 2)
    elif sensor == "joint":
        ratio = I_ROT ** 2 * w ** 4 / (R ** 2 * np.abs(c - I_ROT * w ** 2) ** 2)
    else:
        raise ValueError(sensor)
    return 1.0 / (1.0 + (b_s / B_CONTACT) * ratio)


def snr_per_band(sigma_T2: float | None = None, dt: float = 1.0 / FS,
                 sigma_w: float = SIGMA_A) -> float:
    """S / (2 sigma_T^2 b_c): the full-state SNR. The DCT prior has two-sided PSD sigma_w^2 dt."""
    sigma_T2 = SIGMA_ENERGY ** 2 if sigma_T2 is None else sigma_T2
    return sigma_w ** 2 * dt / (2.0 * sigma_T2 * B_CONTACT)


def predicted_efficiency(sys, C, band_hz: float = 100.0, eps: float = 0.0,
                         snr: float | None = None, n_freq: int = 2000) -> float:
    """Infinite-window routing efficiency: int log(1 + snr rho) df / (B log(1 + snr))."""
    snr = snr_per_band() if snr is None else snr
    f = np.linspace(0.0, band_hz, n_freq + 1)[1:]
    rho = spectral_routing(sys, C, 2 * np.pi * f, eps=eps)
    return float(np.mean(np.log1p(snr * rho)) / np.log1p(snr))


def routing_efficiency(sys, C, eps: float = 1e-9, band_hz: float = 100.0) -> float:
    """Finite-window routing efficiency: sensor bits / full-state bits (Langevin body)."""
    full = checks.langevin_bits(sys, np.eye(2 * sys.n), 0.0, band_hz=band_hz)
    return checks.langevin_bits(sys, np.atleast_2d(C), eps, band_hz=band_hz) / full


MOTOR = {"rigid": np.array([[1.0, 0.0]]), "sea": np.array([[0.0, 1.0, 0.0, 0.0]])}
JOINT = np.array([[1.0, 0.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------------------
# Checks used by the tests and the research note
# ---------------------------------------------------------------------------------------

def check_closed_forms(ks_list=(70.0, 1e3, 1e4, 1e6), f_hz=(1.0, 10.0, 50.0, 100.0)) -> dict:
    """State-space rho(w) against the closed forms, for both encoders."""
    w = 2 * np.pi * np.asarray(f_hz)
    worst = 0.0
    for ks in ks_list:
        s = series_elastic(ks, readout="energy")
        for sensor, C in (("motor", MOTOR["sea"]), ("joint", JOINT)):
            num = spectral_routing(s, C, w)
            worst = max(worst, float(np.max(np.abs(num - sea_routing_closed_form(ks, w, sensor)))))
    return dict(max_abs_err=worst)


def check_perfect_routing(f_hz=(1.0, 10.0, 50.0, 100.0)) -> dict:
    """No internal damping and no readout noise: every sensor routes everything."""
    w = 2 * np.pi * np.asarray(f_hz)
    r = proprioceptive(readout="energy")
    s = series_elastic(1e4, readout="energy")
    return dict(rigid_motor=spectral_routing(r, MOTOR["rigid"], w),
                rigid_velocity=spectral_routing(r, np.array([[0.0, 1.0]]), w),
                sea_full_state=spectral_routing(s, np.eye(4), w),
                rigid_internal_norm=float(np.abs(internal_damping(r)).max()))


def check_reparametrization(ks: float = 1e3, f_hz=(5.0, 40.0, 90.0)) -> dict:
    """rho is a property of the physical sensor, not of the coordinates used to describe it."""
    w = 2 * np.pi * np.asarray(f_hz)
    base = spectral_routing(checks.reparametrized_sea(ks, np.eye(2), "energy"), MOTOR["sea"], w)
    worst = 0.0
    for T in checks.REPARAMETRIZATIONS.values():
        sT = checks.reparametrized_sea(ks, T, "energy")
        calT = np.kron(np.eye(2), T)                      # x_q = calT x_new
        worst = max(worst, float(np.max(np.abs(spectral_routing(sT, MOTOR["sea"] @ calT, w) - base))))
    return dict(max_abs_err=worst)


def check_prediction(ks_list=(70.0, 1e3, 1e4, 1e6)) -> dict:
    """Finite-window efficiency against the infinite-window prediction."""
    rows = {}
    for ks in ks_list:
        s = series_elastic(ks, readout="energy")
        for sensor, C in (("motor", MOTOR["sea"]), ("joint", JOINT)):
            rows[(ks, sensor)] = dict(finite=routing_efficiency(s, C),
                                      predicted=predicted_efficiency(s, C))
    r = proprioceptive(readout="energy")
    rows[("rigid", "motor")] = dict(finite=routing_efficiency(r, MOTOR["rigid"]),
                                    predicted=predicted_efficiency(r, MOTOR["rigid"]))
    return rows


def check_low_snr_limit(ks: float = 1e3) -> dict:
    """At low SNR the efficiency tends to the band average of rho."""
    s = series_elastic(ks, readout="energy")
    f = np.linspace(0.0, 100.0, 2001)[1:]
    rho_mean = float(np.mean(spectral_routing(s, MOTOR["sea"], 2 * np.pi * f)))
    return dict(rho_mean=rho_mean,
                eta_low=predicted_efficiency(s, MOTOR["sea"], snr=1e-6),
                eta_default=predicted_efficiency(s, MOTOR["sea"]))


def check_series_damping(ks: float = 70.0, zetas=(0.1, 0.03, 0.01)) -> dict:
    """Less damping between contact and motor means more routed information."""
    out = {}
    for z in zetas:
        s = series_elastic(ks, zeta_series=z, readout="energy")
        full = checks.langevin_bits(s, np.eye(4), 0.0)
        out[z] = dict(full=full,
                      motor=checks.langevin_bits(s, MOTOR["sea"], 1e-9) / full)
    return out


def check_encoder_noise(ks_list=(70.0, 1e4)) -> dict:
    """Encoder resolution (SIGMA_THETA) lowers routing for both sensors."""
    out = {}
    for ks in ks_list:
        s = series_elastic(ks, readout="energy")
        for sensor, C in (("motor", MOTOR["sea"]), ("joint", JOINT)):
            out[(ks, sensor)] = dict(noiseless=routing_efficiency(s, C),
                                     encoder=routing_efficiency(s, C, eps=SIGMA_THETA))
    return out
