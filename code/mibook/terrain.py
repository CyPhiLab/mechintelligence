"""What the body holds about the height of the ground under the foot (thread T11).

The ground surface sits at an unknown height z(t). The contact spring k_c and damper b_c join
the foot to the surface, so the body is driven by the force k_c z + b_c zdot, through the same
channel as a contact force and with the same thermal noise. A height profile is expanded in
the same DCT modes as the force profiles elsewhere in the book; the force each mode exerts is
k_c phi_j + b_c phi_j', with phi_j' the mode's time derivative.
"""
from __future__ import annotations

import numpy as np

from mechinfo.mechanics import bits_from_gram, zoh
from passive_info.information import (FS, SIGMA_A, SIGMA_ENERGY, T_WINDOW, mode_basis,
                                      n_modes)
from passive_info.systems import K_I, proprioceptive, reference_m_e, series_elastic

from mibook import checks
from mibook import propagation as pg

#: Prior width per height mode (m): the height whose static force k_c z matches the force
#: prior SIGMA_A on the book's ground, so the two questions start on an equal footing.
SIGMA_Z = SIGMA_A / K_I


def height_modes(t_window: float = T_WINDOW, fs: float = FS, band_hz: float = 100.0):
    """DCT modes phi (n_t, d) and their time derivatives dphi, in 1/s."""
    n_t = int(round(t_window * fs)) + 1
    d = n_modes(t_window, band_hz)
    Phi = mode_basis(n_t, d)
    t = (np.arange(n_t) + 0.5) / n_t
    j = np.arange(d)
    norms = np.linalg.norm(np.cos(np.pi * np.outer(t, j)), axis=0)
    dPhi = -(np.pi * j / (n_t / fs)) * np.sin(np.pi * np.outer(t, j)) / norms
    return Phi, dPhi


def response_for_profiles(sys, C, F: np.ndarray, fs: float = FS) -> np.ndarray:
    """Response of C x to each column of sampled force profiles F (n_t, d), zero-order hold."""
    C = np.atleast_2d(C)
    Ad, Bd = zoh(sys.A, sys.B_F[:, :1], 1.0 / fs)
    n_t, d = F.shape
    X = np.zeros((sys.A.shape[0], d))
    out = [C @ X]
    for k in range(n_t - 1):
        X = Ad @ X + Bd * F[k][None, :]
        out.append(C @ X)
    return np.concatenate(out, axis=0)


def height_bits(sys, C, eps: float, k_c: float, b_c: float, band_hz: float = 100.0,
                sigma_z: float = SIGMA_Z) -> float:
    """Bits held by C x about the height profile, under the Langevin body's noise."""
    Phi, dPhi = height_modes(band_hz=band_hz)
    M = response_for_profiles(sys, C, k_c * Phi + b_c * dPhi)
    S = checks.langevin_noise_cov(sys, C, Phi.shape[0], eps)
    Z = np.linalg.solve(np.linalg.cholesky(S), M)
    return bits_from_gram(sigma_z ** 2 * (Z.T @ Z))


def leg(ks, k_c: float = K_I, b_c: float = pg.B_CONTACT):
    """The rigid leg (ks=None) or an SEA, on a ground with stiffness k_c and damping b_c."""
    zeta = b_c / (2.0 * np.sqrt(k_c * reference_m_e()))
    kw = dict(readout="energy", k_i=k_c, zeta_contact=zeta)
    return proprioceptive(**kw) if ks is None else series_elastic(ks, **kw)


def predicted_whole_bits(k_c: float, b_c: float, quantity: str, band_hz: float = 100.0,
                         fs: float = FS, t_window: float = T_WINDOW) -> float:
    """Whole-body bits mode by mode: sum_j 1/2 log2(1 + gamma_j).

    Each force mode has gamma = sigma_w^2 dt / (2 sigma_T^2 b_c). A height mode at frequency
    w_j = pi j / T exerts force k_c phi_j + b_c phi_j', of squared size k_c^2 + b_c^2 w_j^2,
    so gamma_j = sigma_z^2 dt (k_c^2 + b_c^2 w_j^2) / (2 sigma_T^2 b_c). Cross terms between
    the cosine and sine shapes, and the window's edges, are neglected.
    """
    dt = 1.0 / fs
    d = n_modes(t_window, band_hz)
    T = (int(round(t_window * fs)) + 1) * dt
    w = np.pi * np.arange(d) / T
    if quantity == "force":
        g = np.full(d, SIGMA_A ** 2 * dt / (2 * SIGMA_ENERGY ** 2 * b_c))
    else:
        g = SIGMA_Z ** 2 * dt * (k_c ** 2 + b_c ** 2 * w ** 2) / (2 * SIGMA_ENERGY ** 2 * b_c)
    return float(0.5 * np.sum(np.log2(1.0 + g)))


# ---------------------------------------------------------------------------------------
# Checks used by the tests and the research note
# ---------------------------------------------------------------------------------------

def check_designs(ks_list=(70.0, 1e3, 1e4, 1e6)) -> dict:
    """Whole-body bits about height and force for every design, and the rotor's share."""
    rows = {}
    for ks in (None,) + tuple(ks_list):
        s = leg(ks)
        n2 = 2 * s.n
        Cr = np.zeros((2, n2))
        Cr[0, s.n - 1], Cr[1, n2 - 1] = 1.0, 1.0           # the rotor: angle and velocity
        whole_z = height_bits(s, np.eye(n2), 0.0, K_I, pg.B_CONTACT)
        whole_f = checks.langevin_bits(s, np.eye(n2), 0.0)
        rows["rigid" if ks is None else ks] = dict(
            whole_z=whole_z, whole_f=whole_f,
            rotor_z=height_bits(s, Cr, pg.JITTER, K_I, pg.B_CONTACT) / whole_z,
            rotor_f=checks.langevin_bits(s, Cr, pg.JITTER) / whole_f)
    return rows


def check_contact_sweeps(k_factors=(0.25, 0.5, 1.0, 2.0, 4.0),
                         b_factors=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)) -> dict:
    """Whole-body bits on the rigid leg as the ground's stiffness or damping changes alone."""
    out = dict(k=[], b=[])
    for fk in k_factors:
        k_c = fk * K_I
        s = leg(None, k_c=k_c)
        n2 = 2 * s.n
        out["k"].append(dict(factor=fk,
                             z=height_bits(s, np.eye(n2), 0.0, k_c, pg.B_CONTACT),
                             f=checks.langevin_bits(s, np.eye(n2), 0.0),
                             z_pred=predicted_whole_bits(k_c, pg.B_CONTACT, "height"),
                             f_pred=predicted_whole_bits(k_c, pg.B_CONTACT, "force")))
    for fb in b_factors:
        b_c = fb * pg.B_CONTACT
        s = leg(None, b_c=b_c)
        n2 = 2 * s.n
        out["b"].append(dict(factor=fb,
                             z=height_bits(s, np.eye(n2), 0.0, K_I, b_c),
                             f=checks.langevin_bits(s, np.eye(n2), 0.0),
                             z_pred=predicted_whole_bits(K_I, b_c, "height"),
                             f_pred=predicted_whole_bits(K_I, b_c, "force")))
    return out
