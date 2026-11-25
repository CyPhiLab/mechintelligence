"""Item 4 of the 2026-11-25 priorities: the checks owed under thread T11 before the
parameter notes go into a chapter.

1. The finite-window information, computed exactly (`linear.window_fisher`, no Whittle
   formula), against the long-window rate from the sampled spectrum
   (`parameters.whittle_rate_general`).
2. Growth with the band: the exact finite-window rate as the sampling rate rises, for a
   mass the part touches, a mass it does not touch, and the ground's stiffness.
3. Several coordinates and a chain: which masses a part made of one or more nodes of a chain
   learns without limit.
"""
from __future__ import annotations

import numpy as np

from mibook import parameters as pm
from mibook import propagation as pg
from passive_info.information import SIGMA_ENERGY
from passive_info.systems import K_I, M_L

from . import linear as L

EPS = pg.JITTER


def _adapt(s, params, build):
    n = s.n
    D = -s.H @ s.A[n:, n:]
    return L.LinearBody(H=s.H, D=0.5 * (D + D.T), K=s.K, inputs={}, sigma_T2=SIGMA_ENERGY ** 2,
                        params=params, build=build)


def leg(ks=None, m_l: float = M_L, k_c: float = K_I, b_c: float = pg.B_CONTACT):
    """The rigid leg (ks=None) or an SEA with link mass m_l on ground (k_c, b_c), as a
    LinearBody whose parameters m_l, k_c and b_c can be changed."""
    def build(m_l=m_l, k_c=k_c, b_c=b_c, ks=ks):
        if m_l != M_L:
            s = pm.leg_mass(ks, np.log(m_l / M_L))
            if k_c != K_I or b_c != pg.B_CONTACT:
                raise ValueError("change one parameter at a time")
        else:
            s = pm.leg_at(ks, np.log(k_c / K_I), b_c)
        return _adapt(s, dict(m_l=m_l, k_c=k_c, b_c=b_c, ks=ks), build)
    return build()


def positions(body, rows):
    P = np.zeros((len(rows), 2 * body.n))
    P[np.arange(len(rows)), list(rows)] = 1.0
    return P


def check_against_stacked(ks_list=(None, 1e3), t_window: float = 0.085, fs: float = 4000.0):
    """window_fisher about log k_c against the book's stacked-covariance computation."""
    rows = {}
    for ks in ks_list:
        b = leg(ks)
        for part in (("whole",) if ks is None else ("whole", "rotor")):
            C = positions(b, pm.part_rows(ks, part))
            n_t = int(round(t_window * fs)) + 1
            new = L.window_fisher(b, "k_c", C, n_t, fs, EPS)
            old = pm.fisher_thermal(ks, part, t_window, fs)
            rows[("rigid" if ks is None else ks, part)] = (new, old)
    return rows


def window_rates(ks=1e3, fs: float = 4000.0, t1: float = 0.1, t2: float = 0.4) -> dict:
    """Exact finite-window rate (slope between windows t1 and t2, per second) against the
    sampled-spectrum Whittle rate, for log m_l, log k_c and log b_c, at link and rotor."""
    b = leg(ks)
    out = {}
    builders = {"m_l": lambda t: pm.leg_mass(ks, t),
                "k_c": lambda t: pm.leg_at(ks, t),
                "b_c": lambda t: pm.leg_at(ks, 0.0, pg.B_CONTACT * np.exp(t))}
    for part in ("link", "rotor"):
        rows = pm.part_rows(ks, part)
        C = positions(b, rows)
        for param, bld in builders.items():
            n1, n2 = int(t1 * fs), int(t2 * fs)
            I1 = L.window_fisher(b, param, C, n1, fs, EPS)
            I2 = L.window_fisher(b, param, C, n2, fs, EPS)
            slope = (I2 - I1) / (t2 - t1)
            whittle = pm.whittle_rate_general(bld, rows, fs, 4000)
            out[(part, param)] = dict(window=I2, slope=slope, whittle=whittle)
    return out


def band_growth(ks=1e3, fs_list=(2000.0, 4000.0, 8000.0, 16000.0), t1: float = 0.1,
                t2: float = 0.3) -> dict:
    """Exact finite-window rate per second as the sampling rate rises."""
    b = leg(ks)
    out = {}
    for part in ("link", "rotor"):
        C = positions(b, pm.part_rows(ks, part))
        for param in ("m_l", "k_c"):
            r = []
            for fs in fs_list:
                n1, n2 = int(t1 * fs), int(t2 * fs)
                I1 = L.window_fisher(b, param, C, n1, fs, EPS, rel_step=1e-2)
                I2 = L.window_fisher(b, param, C, n2, fs, EPS, rel_step=1e-2)
                r.append((I2 - I1) / (t2 - t1))
            out[(part, param)] = np.array(r)
    return dict(fs=np.asarray(fs_list), rates=out)


def chain_body(m=(1.0, 0.5, 2.0), k=(4e3, 2e3, 3e3), b=(40.0, 30.0, 50.0), sigma_T2=1e-4):
    """A grounded chain: element 0 ties node 0 to the ground; element j ties j-1 to j.
    Parameters m0, m1, m2 (masses) can be changed. Damped enough that windows of a second
    reach the long-window rate (a lightly damped chain needs windows of many seconds, and
    its Whittle sum needs a very fine frequency grid)."""
    def build(**p):
        mm = [p[f"m{i}"] for i in range(len(m))]
        n = len(mm)
        K = np.zeros((n, n))
        D = np.zeros((n, n))
        for j in range(n):
            a = np.zeros(n)
            a[j] = 1.0
            if j > 0:
                a[j - 1] = -1.0
            K += k[j] * np.outer(a, a)
            D += b[j] * np.outer(a, a)
        return L.LinearBody(H=np.diag(mm), D=D, K=K, inputs={}, sigma_T2=sigma_T2,
                            params=p, build=build)
    return build(**{f"m{i}": mi for i, mi in enumerate(m)})


def chain_growth(fs_list=(1000.0, 2000.0, 4000.0, 8000.0), t1: float = 0.5, t2: float = 1.0,
                 parts=((1,), (1, 2), (0, 1, 2))) -> dict:
    """For parts made of one or more nodes of a 3-mass chain, the exact window rate about
    each mass as the sampling rate rises."""
    b = chain_body()
    out = {}
    for rows in parts:
        C = positions(b, rows)
        for i in range(b.n):
            r = []
            for fs in fs_list:
                n1, n2 = int(t1 * fs), int(t2 * fs)
                I1 = L.window_fisher(b, f"m{i}", C, n1, fs, 0.0, rel_step=1e-2)
                I2 = L.window_fisher(b, f"m{i}", C, n2, fs, 0.0, rel_step=1e-2)
                r.append((I2 - I1) / (t2 - t1))
            out[(rows, i)] = np.array(r)
    return dict(fs=np.asarray(fs_list), rates=out)


def growth_per_hz(fs, rates) -> float:
    """Slope of the rate against the band fs/2, from the last two sampling rates."""
    return float((rates[-1] - rates[-2]) / ((fs[-1] - fs[-2]) / 2))


def predicted_growth(body, param: str, rows, rel_step: float = 1e-6) -> float:
    """Growth per hertz of the band predicted for a part of one or more coordinates:
    tr[(c^-1 dc)^2], with c = P H^-1 D H^-1 P^T the leading coefficient of the part's
    high-frequency spectrum and d the derivative in log(param). For one coordinate this is
    (d log c)^2, the theorem of the finite-answers note."""
    P = positions(body, rows)[:, :body.n]

    def c_of(b):
        Hi = np.linalg.inv(b.H)
        return P @ Hi @ b.D @ Hi @ P.T

    th = body.params[param]
    cp = c_of(body.with_params(**{param: th * np.exp(rel_step)}))
    cm = c_of(body.with_params(**{param: th * np.exp(-rel_step)}))
    X = np.linalg.solve(c_of(body), (cp - cm) / (2 * rel_step))
    return float(np.trace(X @ X))
