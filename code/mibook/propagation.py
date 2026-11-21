"""How information about the contact force propagates through the mechanism.

The setting is the Langevin body: the only noise is the body's own thermal noise, injected
through every damper with the intensity set by the fluctuation-dissipation theorem. A *part*
of the body is a set of its coordinates, selected by a matrix `P` acting on x_q = (q, qdot)
(for example the rotor's angle and velocity). The question is how much of the body's
information about the contact force that part holds, and what erodes it on the way.

The contact damper acts through the same Jacobian as the contact force, so its noise enters
exactly like the force. A part therefore holds the force as if it were seen through an
*effective contact damping* b_eff(w) >= b_c, and rho(w) = b_c / b_eff(w) is the spectral
propagation efficiency. Sensing (readout noise, where to put an encoder) is not modelled here.
"""
from __future__ import annotations

import numpy as np

from mechinfo.mechanics import ground_damping
from passive_info.information import FS, SIGMA_A, SIGMA_ENERGY
from passive_info.systems import (I_ROT, K_I, M_L, R, ZETA_CONTACT, ZETA_SERIES,
                                  proprioceptive, reference_m_e, series_elastic)

from mibook import checks

#: The contact damper b_i of the ground model (N s/m); identical for every design.
B_CONTACT = float(ground_damping(K_I, reference_m_e(), ZETA_CONTACT))

#: Parts of the two legs, as selectors on x_q. Each part is a coordinate and its velocity.
ROTOR = {"rigid": np.array([[1.0, 0.0], [0.0, 1.0]]),             # the rigid leg's one body
         "sea": np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])}
LINK = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])     # the SEA's joint-side link

#: Tiny diagonal added to the stacked noise covariance of a part, for numerical
#: conditioning only. It is many orders below the thermal noise and is not a model of anything.
JITTER = 1e-9


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


def effective_contact_damping(sys, P, omegas, sigma_T2: float | None = None,
                              b_contact: float = B_CONTACT) -> np.ndarray:
    """b_eff(w) = b_c + 1 / (2 sigma_T^2 q(w)), with q = h^* M^+ h.

    h(w) = P (iw - A)^{-1} B_F is the part's response to the contact force, and
    M(w) = 2 sigma_T^2 G D_int G^* is the thermal noise it receives from the internal dampers,
    with G = P (iw - A)^{-1} [0; H^{-1}]. When h lies outside the range of M the part holds
    the force free of internal noise, q is infinite, and b_eff = b_c.
    """
    sigma_T2 = SIGMA_ENERGY ** 2 if sigma_T2 is None else sigma_T2
    P = np.atleast_2d(P)
    n = sys.n
    Dint = internal_damping(sys, b_contact)
    Bxi = np.vstack([np.zeros((n, n)), np.linalg.inv(sys.H)])
    m = P.shape[0]
    out = []
    for w in np.atleast_1d(omegas):
        Rw = np.linalg.inv(1j * w * np.eye(2 * n) - sys.A)
        h = (P @ Rw @ sys.B_F).reshape(m)
        G = P @ Rw @ Bxi
        M = 2.0 * sigma_T2 * (G @ Dint @ G.conj().T)
        # Internal noise far below the contact damper's own noise at the part (12 orders)
        # changes b_eff by less than 1e-12 b_c and is set to zero, so round-off in a nearly
        # noise-free direction cannot decide the answer.
        lam, V = np.linalg.eigh(0.5 * (M + M.conj().T))
        floor = 1e-12 * max(2.0 * sigma_T2 * b_contact * float(np.real(h.conj() @ h)),
                            float(np.max(np.abs(lam))), 1e-300)
        keep = lam > floor
        c = V.conj().T @ h
        if np.linalg.norm(c[~keep]) > 1e-9 * max(np.linalg.norm(h), 1e-300):
            out.append(b_contact)
            continue
        q = float(np.sum(np.abs(c[keep]) ** 2 / lam[keep]))
        out.append(b_contact + (np.inf if q <= 0 else 1.0 / (2.0 * sigma_T2 * q)))
    return np.array(out)


def spectral_propagation(sys, P, omegas, b_contact: float = B_CONTACT, **kw) -> np.ndarray:
    """rho(w) = b_c / b_eff(w), in [0, 1]."""
    return b_contact / effective_contact_damping(sys, P, omegas, b_contact=b_contact, **kw)


def sea_propagation_closed_form(ks: float, omega, part: str,
                                zeta_series: float = ZETA_SERIES) -> np.ndarray:
    """Closed-form rho(w) for the rotor or the link of the clamped SEA.

    rotor: rho = 1 / (1 + (b_s/b_c) |k_i r^2 - m_j w^2 + i w b_c r^2|^2 / (r^2 |k_s + i b_s w|^2))
    link:  rho = 1 / (1 + (b_s/b_c) I^2 w^4 / (r^2 |k_s - I w^2 + i b_s w|^2))
    """
    w = np.asarray(omega, float)
    b_s = 2.0 * zeta_series * np.sqrt(ks * I_ROT)
    m_j = M_L * R ** 2
    c = ks + 1j * b_s * w
    if part == "rotor":
        z_g = K_I * R ** 2 - m_j * w ** 2 + 1j * w * B_CONTACT * R ** 2
        ratio = np.abs(z_g) ** 2 / (R ** 2 * np.abs(c) ** 2)
    elif part == "link":
        ratio = I_ROT ** 2 * w ** 4 / (R ** 2 * np.abs(c - I_ROT * w ** 2) ** 2)
    else:
        raise ValueError(part)
    return 1.0 / (1.0 + (b_s / B_CONTACT) * ratio)


def snr_per_band(sigma_T2: float | None = None, dt: float = 1.0 / FS,
                 sigma_w: float = SIGMA_A, b_contact: float = B_CONTACT) -> float:
    """S / (2 sigma_T^2 b_c): the whole body's SNR. The DCT prior has two-sided PSD sigma_w^2 dt."""
    sigma_T2 = SIGMA_ENERGY ** 2 if sigma_T2 is None else sigma_T2
    return sigma_w ** 2 * dt / (2.0 * sigma_T2 * b_contact)


def predicted_share(sys, P, band_hz: float = 100.0, snr: float | None = None,
                    n_freq: int = 2000, b_contact: float = B_CONTACT) -> float:
    """Infinite-window share: int log(1 + snr rho) df / (B log(1 + snr))."""
    snr = snr_per_band(b_contact=b_contact) if snr is None else snr
    f = np.linspace(0.0, band_hz, n_freq + 1)[1:]
    rho = spectral_propagation(sys, P, 2 * np.pi * f, b_contact=b_contact)
    return float(np.mean(np.log1p(snr * rho)) / np.log1p(snr))


def information_share(sys, P, band_hz: float = 100.0) -> float:
    """Finite-window share: information held by the part / information held by the body."""
    whole = checks.langevin_bits(sys, np.eye(2 * sys.n), 0.0, band_hz=band_hz)
    return checks.langevin_bits(sys, np.atleast_2d(P), JITTER, band_hz=band_hz) / whole


# ---------------------------------------------------------------------------------------
# Checks used by the tests and the research note
# ---------------------------------------------------------------------------------------

def check_closed_forms(ks_list=(70.0, 1e3, 1e4, 1e6), f_hz=(1.0, 10.0, 50.0, 100.0)) -> dict:
    """State-space rho(w) against the closed forms, for the rotor and the link.

    The closed forms are derived for a coordinate's trajectory, so the angle row is used;
    `check_angle_equals_part` shows that adding the velocity row leaves rho unchanged.
    """
    w = 2 * np.pi * np.asarray(f_hz)
    worst = 0.0
    for ks in ks_list:
        s = series_elastic(ks, readout="energy")
        for part, P in (("rotor", ROTOR["sea"][:1]), ("link", LINK[:1])):
            num = spectral_propagation(s, P, w)
            worst = max(worst, float(np.max(np.abs(num - sea_propagation_closed_form(ks, w, part)))))
    return dict(max_abs_err=worst)


def check_lossless_propagation(f_hz=(1.0, 10.0, 50.0, 100.0)) -> dict:
    """No internal damper: every part holds everything the body holds."""
    w = 2 * np.pi * np.asarray(f_hz)
    r = proprioceptive(readout="energy")
    s = series_elastic(1e4, readout="energy")
    return dict(rigid_angle=spectral_propagation(r, np.array([[1.0, 0.0]]), w),
                rigid_velocity=spectral_propagation(r, np.array([[0.0, 1.0]]), w),
                sea_whole_body=spectral_propagation(s, np.eye(4), w),
                rigid_internal_norm=float(np.abs(internal_damping(r)).max()))


def check_reparametrization(ks: float = 1e3, f_hz=(5.0, 40.0, 90.0)) -> dict:
    """rho belongs to the physical part, not to the coordinates used to describe it."""
    w = 2 * np.pi * np.asarray(f_hz)
    base = spectral_propagation(checks.reparametrized_sea(ks, np.eye(2), "energy"), ROTOR["sea"], w)
    worst = 0.0
    for T in checks.REPARAMETRIZATIONS.values():
        sT = checks.reparametrized_sea(ks, T, "energy")
        calT = np.kron(np.eye(2), T)                      # x_q = calT x_new
        worst = max(worst, float(np.max(np.abs(spectral_propagation(sT, ROTOR["sea"] @ calT, w) - base))))
    return dict(max_abs_err=worst)


def check_angle_equals_part(ks: float = 1e3, f_hz=(5.0, 40.0, 90.0)) -> dict:
    """A coordinate's angle alone and its angle with velocity give the same rho."""
    w = 2 * np.pi * np.asarray(f_hz)
    s = series_elastic(ks, readout="energy")
    return dict(angle=spectral_propagation(s, ROTOR["sea"][:1], w),
                part=spectral_propagation(s, ROTOR["sea"], w))


def check_prediction(ks_list=(70.0, 1e3, 1e4, 1e6)) -> dict:
    """Finite-window share against the infinite-window prediction."""
    rows = {}
    for ks in ks_list:
        s = series_elastic(ks, readout="energy")
        for part, P in (("rotor", ROTOR["sea"]), ("link", LINK)):
            rows[(ks, part)] = dict(finite=information_share(s, P),
                                    predicted=predicted_share(s, P))
    r = proprioceptive(readout="energy")
    rows[("rigid", "rotor")] = dict(finite=information_share(r, np.array([[1.0, 0.0]])),
                                    predicted=predicted_share(r, np.array([[1.0, 0.0]])))
    return rows


def check_low_snr_limit(ks: float = 1e3) -> dict:
    """At low SNR the share tends to the band average of rho."""
    s = series_elastic(ks, readout="energy")
    f = np.linspace(0.0, 100.0, 2001)[1:]
    rho_mean = float(np.mean(spectral_propagation(s, ROTOR["sea"], 2 * np.pi * f)))
    return dict(rho_mean=rho_mean,
                share_low=predicted_share(s, ROTOR["sea"], snr=1e-6),
                share_default=predicted_share(s, ROTOR["sea"]))


def check_series_damping(ks: float = 70.0, zetas=(0.1, 0.03, 0.01)) -> dict:
    """Less damping between the contact and the rotor: more information reaches the rotor."""
    out = {}
    for z in zetas:
        s = series_elastic(ks, zeta_series=z, readout="energy")
        whole = checks.langevin_bits(s, np.eye(4), 0.0)
        out[z] = dict(whole=whole,
                      rotor=checks.langevin_bits(s, ROTOR["sea"], JITTER) / whole)
    return out
