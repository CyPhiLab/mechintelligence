"""Which properties of the environment have a finite information rate without an outside scale.

A part's position y = p^T q, driven by the Langevin body's white thermal force, has the spectrum
S(w) = 2 sigma_T^2 p^T Y D Y^* p with Y = (K - w^2 H + i w D)^{-1}. At high frequency
Y ~ -H^{-1} / w^2, so S(w) ~ c / w^4 with the leading coefficient

    c = 2 sigma_T^2 p^T H^{-1} D H^{-1} p.

By Whittle's formula the information rate about theta, resolved up to a band B, is
(1/4pi) int_{|w|<2 pi B} (d_theta log S)^2 dw. It converges as B grows when c does not depend on
theta, and grows like (d_theta log c)^2 B when it does.
"""
from __future__ import annotations

import numpy as np

from passive_info.systems import K_I, M_L, ZETA_SERIES, proprioceptive, series_elastic

from mibook import bath, checks
from mibook import propagation as pg
from mibook import terrain as tr

KS = 1e3                  #: the SEA used for the checks
DTHETA = 1e-3


def sea(theta=None, which: str | None = None, ks: float = KS):
    """The SEA with one property scaled by e^theta: 'mass' (link), 'b_c', 'b_s', 'k_c', 'k_s'."""
    scale = 1.0 if theta is None else float(np.exp(theta))
    m_l, b_c, zeta_s, k_c, k_s = M_L, pg.B_CONTACT, ZETA_SERIES, K_I, ks
    if which == "mass":
        m_l *= scale
    elif which == "b_c":
        b_c *= scale
    elif which == "b_s":
        zeta_s *= scale
    elif which == "k_c":
        k_c *= scale
    elif which == "k_s":
        k_s *= scale
        zeta_s /= np.sqrt(scale)            # keep b_s = 2 zeta_s sqrt(k_s I) fixed
    zeta_c = b_c / (2.0 * np.sqrt(k_c * tr.reference_m_e()))
    return series_elastic(k_s, m_l=m_l, zeta_series=zeta_s, k_i=k_c, zeta_contact=zeta_c,
                          readout="energy")


PARTS = {"link": 0, "rotor": 1}
PROPERTIES = ("mass", "b_c", "b_s", "k_c", "k_s")


def leading_coefficient(sys, row: int) -> float:
    """c / (2 sigma_T^2) = p^T H^{-1} D H^{-1} p for the part at coordinate `row`."""
    Hi = np.linalg.inv(sys.H)
    D = checks._damping(sys)
    p = np.zeros(sys.n)
    p[row] = 1.0
    return float(p @ Hi @ D @ Hi @ p)


def predicted_slope(which: str, part: str) -> float:
    """(d_theta log c)^2: the predicted information per second per hertz at high bands."""
    row = PARTS[part]
    cp = leading_coefficient(sea(DTHETA, which), row)
    cm = leading_coefficient(sea(-DTHETA, which), row)
    return float(((np.log(cp) - np.log(cm)) / (2 * DTHETA)) ** 2)


def rate(which: str, part: str, band_hz: float, fs: float = 64000.0, n_freq: int = 3000) -> float:
    """Fisher information per second about theta from the part's position, up to band_hz."""
    return bath.whittle_rate(lambda t: sea(t, which), 0.0, [PARTS[part]], fs, velocities=False,
                             white=True, band_hz=band_hz, n_freq=n_freq)


def check_criterion(bands=(800.0, 1600.0, 3200.0), n_freq: int = 3000) -> dict:
    """For every property and part: the measured growth per hertz between the top two bands,
    against the predicted (d_theta log c)^2."""
    rows = {}
    for which in PROPERTIES:
        for part in PARTS:
            r = [rate(which, part, b, n_freq=n_freq) for b in bands]
            rows[(which, part)] = dict(rates=np.array(r),
                                       slope=(r[-1] - r[-2]) / (bands[-1] - bands[-2]),
                                       predicted=predicted_slope(which, part))
    return dict(bands=np.asarray(bands), rows=rows)
