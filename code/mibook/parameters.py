"""What the body holds about the ground's stiffness, a property of the environment that changes
the dynamics themselves (thread T11).

The unknown is theta = log k_c, with the ground damping b_c held fixed. The body is the Langevin
body at a fixed temperature sigma_T^2. Its thermal motion is a stationary Gaussian process whose
covariance depends on theta, so the motion itself carries information about the ground even when
nothing pushes on the body. The measure is the Fisher information about theta; a Gaussian
approximation turns it into bits.
"""
from __future__ import annotations

import numpy as np

from mechinfo.mechanics import energy_metric, zoh
from passive_info.information import FS, SIGMA_ENERGY, T_WINDOW
from passive_info.systems import K_I

from mibook import checks
from mibook import propagation as pg
from mibook import terrain as tr

#: Step in theta for the central differences, and a second one to check them.
DTHETA = 1e-3

#: Prior width on log k_c for the bits approximation: the ground stiffness known to within a
#: factor e before the window.
SIGMA_THETA = 1.0


def leg_at(ks, theta: float, b_c: float = pg.B_CONTACT):
    """The rigid leg (ks=None) or an SEA on ground of stiffness K_I e^theta, damping b_c."""
    return tr.leg(ks, k_c=K_I * np.exp(theta), b_c=b_c)


def positions(sys, rows) -> np.ndarray:
    """Selector for the positions of the coordinates in `rows`."""
    P = np.zeros((len(rows), 2 * sys.n))
    P[np.arange(len(rows)), list(rows)] = 1.0
    return P


def part_rows(ks, part: str):
    """Coordinates of a part: 'whole', 'rotor' or 'link' (the SEA's joint side)."""
    if ks is None:
        return [0]                                   # the rigid leg has one coordinate
    return {"whole": [0, 1], "rotor": [1], "link": [0]}[part]


def _cov(ks, theta, rows, n_t, eps, b_c=pg.B_CONTACT):
    s = leg_at(ks, theta, b_c)
    return checks.langevin_noise_cov(s, positions(s, rows), n_t, eps)


def fisher_thermal(ks, part: str = "whole", t_window: float = T_WINDOW, fs: float = FS,
                   dtheta: float = DTHETA, eps: float = pg.JITTER,
                   b_c: float = pg.B_CONTACT) -> float:
    """Fisher information about log k_c in the part's thermal motion over a window.

    For a zero-mean Gaussian observation with covariance Sigma(theta),
    I = 1/2 tr(Sigma^{-1} dSigma Sigma^{-1} dSigma); dSigma by central differences.
    """
    n_t = int(round(t_window * fs)) + 1
    rows = part_rows(ks, part)
    S0 = _cov(ks, 0.0, rows, n_t, eps, b_c)
    dS = (_cov(ks, dtheta, rows, n_t, eps, b_c) - _cov(ks, -dtheta, rows, n_t, eps, b_c)) / (2 * dtheta)
    X = np.linalg.solve(S0, dS)
    return 0.5 * float(np.trace(X @ X))


def fisher_push(ks, part: str = "whole", F0: float = 100.0, t_window: float = T_WINDOW,
                fs: float = FS, dtheta: float = DTHETA, eps: float = pg.JITTER) -> float:
    """Fisher information added by a known step force F0 at the foot: dmu^T Sigma^{-1} dmu."""
    n_t = int(round(t_window * fs)) + 1
    rows = part_rows(ks, part)

    def mean(theta):
        s = leg_at(ks, theta)
        F = np.full((n_t, 1), F0)
        return tr.response_for_profiles(s, positions(s, rows), F, fs).ravel()

    dmu = (mean(dtheta) - mean(-dtheta)) / (2 * dtheta)
    S0 = _cov(ks, 0.0, rows, n_t, eps)
    return float(dmu @ np.linalg.solve(S0, dmu))


def snapshot_fisher(ks, part: str = "whole") -> float:
    """Fisher information in one thermal snapshot of the part's position, from the Gibbs law.

    Equals 1/2 eps^2, with eps the share of the stored energy that sits in the ground spring when
    the part is pushed statically (see the note). Computed here from the covariance directly.
    """
    s = leg_at(ks, 0.0)
    rows = part_rows(ks, part)

    def cov(theta):
        st = leg_at(ks, theta)
        return SIGMA_ENERGY ** 2 * np.linalg.inv(st.K)[np.ix_(rows, rows)]

    S0 = cov(0.0)
    dS = (cov(DTHETA) - cov(-DTHETA)) / (2 * DTHETA)
    X = np.linalg.solve(S0, dS)
    return 0.5 * float(np.trace(X @ X))


def ground_energy_share(ks, part: str = "whole") -> float:
    """eps: push the part statically with a unit force (or the whole body at the foot); the
    fraction of the stored energy held by the ground spring."""
    s = leg_at(ks, 0.0)
    J = np.atleast_2d(s.J)
    Ki = np.linalg.inv(s.K)
    if part == "whole":
        return float(K_I * (J @ Ki @ J.T).item())
    p = np.zeros(s.n)
    p[part_rows(ks, part)[0]] = 1.0
    x = Ki @ p
    return float(K_I * (J @ x).item() ** 2 / (p @ x))


def whittle_rate(ks, part: str = "whole", fs: float = FS, n_freq: int = 4000,
                 dtheta: float = DTHETA, b_c: float = pg.B_CONTACT) -> float:
    """Fisher information per sample in the long-window limit (Whittle):
    1/(4 pi) int_{-pi}^{pi} tr[(S^{-1} dS)^2] dnu, with S the sampled process's spectral density."""
    rows = part_rows(ks, part)

    def spectrum(theta, nus):
        s = leg_at(ks, theta, b_c)
        Ad, _ = zoh(s.A, s.B_F[:, :1], 1.0 / fs)
        P = SIGMA_ENERGY ** 2 * np.linalg.inv(energy_metric(s.K, s.H))
        Qd = P - Ad @ P @ Ad.T
        C = positions(s, rows)
        out = []
        for nu in nus:
            R = np.linalg.inv(np.exp(1j * nu) * np.eye(2 * s.n) - Ad)
            out.append(C @ R @ Qd @ R.conj().T @ C.T)
        return out

    nus = np.linspace(-np.pi, np.pi, n_freq, endpoint=False) + np.pi / n_freq
    S0, Sp, Sm = spectrum(0.0, nus), spectrum(dtheta, nus), spectrum(-dtheta, nus)
    vals = []
    for a, b, c in zip(S0, Sp, Sm):
        X = np.linalg.solve(a, (b - c) / (2 * dtheta))
        vals.append(np.real(np.trace(X @ X)))
    return float(np.mean(vals) * 2 * np.pi / (4 * np.pi))


def bits(fisher: float, sigma_theta: float = SIGMA_THETA) -> float:
    """Gaussian approximation to the information about log k_c: 1/2 log2(1 + sigma^2 I)."""
    return 0.5 * float(np.log2(1.0 + sigma_theta ** 2 * fisher))


# ---------------------------------------------------------------------------------------
# Checks used by the tests and the research note
# ---------------------------------------------------------------------------------------

KS_LIST = (70.0, 3e2, 1e3, 3e3, 1e4, 1e5, 1e6)


def check_snapshot_law(ks_list=(70.0, 1e3, 1e4, 1e6)) -> dict:
    """One Gibbs snapshot: Fisher = 1/2 eps^2, eps = ground spring's share of the stored energy."""
    worst, rows = 0.0, {}
    for ks in (None,) + tuple(ks_list):
        for part in (("whole",) if ks is None else ("whole", "link", "rotor")):
            num, eps = snapshot_fisher(ks, part), ground_energy_share(ks, part)
            worst = max(worst, abs(num - 0.5 * eps ** 2))
            rows[("rigid" if ks is None else ks, part)] = dict(fisher=num, eps=eps)
    return dict(max_abs_err=worst, rows=rows)


def check_rate_law(b_factors=(0.5, 1.0, 2.0, 4.0), ks_list=(70.0, 1e3, 1e4)) -> dict:
    """Long-window Fisher rate of the whole body: k_c / (2 b_c) per second, for every design."""
    out = dict(b=[], designs={})
    for f in b_factors:
        b = f * pg.B_CONTACT
        out["b"].append(dict(factor=f, rate=whittle_rate(None, b_c=b, n_freq=2000) * FS,
                             predicted=K_I / (2 * b)))
    for ks in (None,) + tuple(ks_list):
        out["designs"]["rigid" if ks is None else ks] = whittle_rate(ks, n_freq=2000) * FS
    return out


def check_window(windows=(0.085, 0.17, 0.34)) -> dict:
    """Finite-window Fisher of the rigid leg grows at the predicted rate."""
    I = [fisher_thermal(None, t_window=T) for T in windows]
    T = [(round(t * FS) + 1) / FS for t in windows]
    slope = np.diff(I) / np.diff(T)
    return dict(windows=np.array(T), fisher=np.array(I), slopes=slope,
                predicted=K_I / (2 * pg.B_CONTACT), offset=I[0] - T[0] * K_I / (2 * pg.B_CONTACT))


def check_parts(ks_list=KS_LIST) -> dict:
    """Fisher about log k_c held by the whole body, the link and the rotor, thermal and with a
    known 100 N step, and the rotor's share next to its share of force information."""
    rows = {}
    whole_rigid = fisher_thermal(None)
    rows["rigid"] = dict(whole=whole_rigid, push_whole=fisher_push(None))
    for ks in ks_list:
        s = leg_at(ks, 0.0)
        whole = fisher_thermal(ks)
        rotor = fisher_thermal(ks, "rotor")
        rows[ks] = dict(whole=whole, link=fisher_thermal(ks, "link"), rotor=rotor,
                        rotor_share=rotor / whole,
                        rotor_force_share=pg.information_share(s, pg.ROTOR["sea"]),
                        push_whole=fisher_push(ks), push_rotor=fisher_push(ks, "rotor"))
    return rows


# ---------------------------------------------------------------------------------------
# Payload mass: the link carries an unknown mass, theta = log m_l
# ---------------------------------------------------------------------------------------

def leg_mass(ks, theta: float):
    """The rigid leg (ks=None) or an SEA whose link mass is M_L e^theta, on the book's ground."""
    from passive_info.systems import M_L, proprioceptive, series_elastic
    zeta = pg.B_CONTACT / (2.0 * np.sqrt(K_I * tr.reference_m_e()))
    kw = dict(readout="energy", k_i=K_I, zeta_contact=zeta, m_l=M_L * np.exp(theta))
    return proprioceptive(**kw) if ks is None else series_elastic(ks, **kw)


def mass_velocity_snapshot(ks, part: str = "whole") -> dict:
    """One snapshot of the part's velocity: Fisher about log m_l, numerically and as 1/2 eps^2,
    with eps the link's share of the kinetic energy when the part is given a unit impulse."""
    rows = part_rows(ks, part)

    def cov(theta):
        s = leg_mass(ks, theta)
        return SIGMA_ENERGY ** 2 * np.linalg.inv(s.H)[np.ix_(rows, rows)]

    S0 = cov(0.0)
    X = np.linalg.solve(S0, (cov(DTHETA) - cov(-DTHETA)) / (2 * DTHETA))
    s = leg_mass(ks, 0.0)
    Hl = (leg_mass(ks, DTHETA).H - leg_mass(ks, -DTHETA).H) / (2 * DTHETA)   # the link's inertia
    Hi = np.linalg.inv(s.H)
    if part == "whole":
        eps = float(np.trace(Hi @ Hl))           # rank one: the link's share of the inertia
    else:
        p = np.zeros(s.n)
        p[rows[0]] = 1.0
        v = Hi @ p
        eps = float(v @ Hl @ v / (p @ v))
    return dict(fisher=0.5 * float(np.trace(X @ X)), half_eps2=0.5 * eps ** 2, eps=eps)


def whittle_rate_general(build, rows, fs: float = FS, n_freq: int = 4000,
                         dtheta: float = DTHETA, velocities: bool = False) -> float:
    """Long-window Fisher information per second about theta for a system build(theta),
    observing the positions in `rows` (and their velocities if asked)."""
    def spectrum(theta, nus):
        s = build(theta)
        Ad, _ = zoh(s.A, s.B_F[:, :1], 1.0 / fs)
        P = SIGMA_ENERGY ** 2 * np.linalg.inv(energy_metric(s.K, s.H))
        Qd = P - Ad @ P @ Ad.T
        idx = list(rows) + ([s.n + r for r in rows] if velocities else [])
        C = np.eye(2 * s.n)[idx]
        out = []
        for nu in nus:
            R = np.linalg.inv(np.exp(1j * nu) * np.eye(2 * s.n) - Ad)
            out.append(C @ R @ Qd @ R.conj().T @ C.T)
        return out

    nus = np.linspace(-np.pi, np.pi, n_freq, endpoint=False) + np.pi / n_freq
    S0, Sp, Sm = spectrum(0.0, nus), spectrum(dtheta, nus), spectrum(-dtheta, nus)
    vals = []
    for a, b, c in zip(S0, Sp, Sm):
        X = np.linalg.solve(a, (b - c) / (2 * dtheta))
        vals.append(np.real(np.trace(X @ X)))
    return float(np.mean(vals) / 2 * fs)


def check_mass_snapshot(ks_list=(70.0, 1e3, 1e4)) -> dict:
    worst, rows = 0.0, {}
    for ks in (None,) + tuple(ks_list):
        for part in (("whole",) if ks is None else ("whole", "link", "rotor")):
            r = mass_velocity_snapshot(ks, part)
            worst = max(worst, abs(r["fisher"] - r["half_eps2"]))
            rows[("rigid" if ks is None else ks, part)] = r
    return dict(max_abs_err=worst, rows=rows)


def check_resolution(fs_list=(2000.0, 4000.0, 8000.0, 16000.0)) -> dict:
    """Fisher rate on the rigid leg as the sampling rate grows: payload mass (positions, and
    positions with velocities) and ground damping, against ground stiffness."""
    n_of = lambda fs: int(4000 * fs / 4000)       # keep the frequency grid as fine at every rate
    out = dict(fs=np.asarray(fs_list), mass_pos=[], mass_posvel=[], stiff_posvel=[],
               damp_posvel=[])
    damped = lambda t: leg_at(None, 0.0, pg.B_CONTACT * np.exp(t))
    for fs in fs_list:
        n = n_of(fs)
        out["damp_posvel"].append(whittle_rate_general(damped, [0], fs, n, velocities=True))
        out["mass_pos"].append(whittle_rate_general(lambda t: leg_mass(None, t), [0], fs, n))
        out["mass_posvel"].append(whittle_rate_general(lambda t: leg_mass(None, t), [0], fs, n,
                                                       velocities=True))
        out["stiff_posvel"].append(whittle_rate_general(lambda t: leg_at(None, t), [0], fs, n,
                                                        velocities=True))
    return {k: np.asarray(v) for k, v in out.items()}
