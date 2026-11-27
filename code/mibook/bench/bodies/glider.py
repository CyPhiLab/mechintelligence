# SPDX-License-Identifier: GPL-3.0-or-later
"""A gliding gull in the longitudinal plane, after Harvey and Inman (2022), "Gull dynamic
pitch stability is controlled by wing morphing", PNAS 119: e2204847119, with their SI
Appendix and their code (github.com/charvey23/AvianDynStab, commit 5f64416:
aero_functions.py, dyn_functions.py, run_script.py, analyze_data.R), ported here.

Model (PNAS Methods; SI Sect. 3-7). A rigid, symmetric gull gliding at trim, small
perturbations, quasi-steady aerodynamics. State X = (u, d_alpha, q, d_theta): speed
perturbation normalized by the trim speed U0, angle of attack (rad), pitch rate (rad/s) and
pitch angle (rad). With m~ = rho U0 S / (2 m) and I~ = rho U0^2 S c / (2 Iyy),

    A = [[-2 m~ CD,   m~ (CL - CD_a),     0,             -(g/U0) cos(g0)],
         [-2 m~ CL,   m~ (-CL_a - CD),    1 - m~ CL_q,   -(g/U0) sin(g0)],
         [0,          I~ Cm_a,            I~ Cm_q,        0],
         [0,          0,                  1,              0]].

Aerodynamic and inertial coefficients come from the fitted models in the authors' data
(figshare c.5895032, file 2022_02_07_coefficients_all.csv): CL(elbow, wrist, sweep,
dihedral, alpha), CD(elbow, wrist, alpha), Cm(elbow, wrist, sweep, dihedral, CL), the
pitch-rate derivatives CL_q and Cm_q, and Iyy. These were fitted to MachUpX lifting-line
results on the wing shapes of Harvey et al. (2021) and to AvInertia (Harvey et al. 2022,
Nature) for the same glaucous-winged gull. Mass 1.0154 kg, reference area S = 0.267782 m^2
(wings and body), reference chord c = 0.2861011 m (largest root chord), rho = 1.225 kg/m^3
(run_script.py). Angles of attack in the fits are in degrees.

Trim (aero_functions.trim_aero). The lift coefficient at trim makes Cm = 0; the angle of
attack gives that CL; speed and glide angle balance lift, drag and weight. A configuration
cannot trim if Cm at zero lift and dCm/dCL have the same sign. The original solves these
by numerical minimization; here they are solved in closed form (the CL polynomial is a
cubic in alpha, and its real root nearest zero is taken, which is the one the original's
minimization from alpha = 0 finds).

Inputs (environment channels):
- "streamwise_gust": a horizontal gust u_g / U0 along the flight path (dyn_functions.py;
  "should be opposite sign from A since an increase in gust speed will decrease the relative
  speed", after Nelson, Flight Stability and Automatic Control, ch. 6); the paper's discrete
  gust is a 1 - cos rise of 2% of U0 over 5 s (MIL-F-8785C).
- "vertical_gust": a gust w_g / U0 normal to the flight path, which changes the
  aerodynamic angle of attack by w_g / U0 (Etkin and Reid, Dynamics of Flight, ch. 7):
  the aerodynamic terms of the alpha column of A. The paper's transverse gust is the free
  response from a 2 degree offset in alpha.
Noise: the rigid glider has no internal damper, so no thermal noise follows from the model.
A white disturbance of intensity sigma_T2 on the three dynamic rows is declared as the
resolution, as an assumption (the open modeling question of thread T12).
"""
from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass

import numpy as np

from .. import register
from ..linear import StateSpaceBody

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "harvey2022"
RHO, G = 1.225, 9.81
MASS = 1.0154            # kg
S_MAX = 0.267782         # m^2
C_MAX = 0.2861011        # m
# The configuration grid of run_script.py.
SWEEPS = (-20, -15, -10, -5, 0, 5, 10)
DIHEDRALS = (10, 15, 20)
ELBOWS = tuple(range(86, 166, 2))
WRISTS = tuple(range(106, 180, 2))


def load_coefficients(path=None) -> dict[str, dict[str, float]]:
    """Rows of 2022_02_07_coefficients_all.csv keyed by y.model (Iyy, CL, CD, Cm, dCLdq,
    dCmdq)."""
    path = DATA / "2022_02_07_coefficients_all.csv" if path is None else path
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["y.model"]: {k: float(v) for k, v in r.items() if k not in ("", "y.model")}
            for r in rows}


def load_hull(path=None) -> np.ndarray:
    path = DATA / "gull_joint_hull.csv" if path is None else path
    return np.loadtxt(path, delimiter=",", skiprows=1)


def in_hull(elbow, wrist, hull=None) -> np.ndarray:
    """Inside the alpha hull of the gull's measured joint angles (analyze_data.R,
    cut_trueshape)."""
    from matplotlib.path import Path
    hull = load_hull() if hull is None else hull
    pts = np.stack(np.broadcast_arrays(np.asarray(elbow, float), np.asarray(wrist, float)), -1)
    return Path(hull).contains_points(pts.reshape(-1, 2)).reshape(pts.shape[:-1])


# ---------------------------------------------------------------------------------------
# Coefficient models (aero_functions.py, dyn_functions.get_Iyy); angles in degrees
# ---------------------------------------------------------------------------------------

class Coefficients:
    def __init__(self, coef=None):
        self.c = load_coefficients() if coef is None else coef

    def Iyy(self, e, w, s, d):
        k = self.c["Iyy"]
        return (k["intercept"] + k["elbow"] * e + k["manus"] * w + k["sweep"] * s
                + k["dihedral"] * d + k["elbow2"] * e ** 2 + k["manus2"] * w ** 2
                + k["sweep2"] * s ** 2 + k["dihedral2"] * d ** 2 + k["elbowmanus"] * e * w
                + k["elbowsweep"] * e * s + k["manussweep"] * w * s + k["elbowdihedral"] * e * d
                + k["manusdihedral"] * w * d + k["sweepdihedral"] * s * d
                + k["elbowmanussweep"] * e * w * s + k["elbowmanusdihedral"] * e * w * d
                + k["elbowsweepdihedral"] * e * s * d + k["manussweepdihedral"] * w * s * d
                + k["elbowmanussweepdihedral"] * e * w * s * d)

    def CL_poly(self, e, w, s, d):
        """Coefficients (c0, c1, c2, c3) of CL = c0 + c1 a + c2 a^2 + c3 a^3, a in degrees."""
        k = self.c["CL"]
        c0 = (k["intercept"] + k["elbow"] * e + k["manus"] * w + k["sweep"] * s
              + k["dihedral"] * d + k["elbow2"] * e ** 2 + k["manus2"] * w ** 2
              + k["manus3"] * w ** 3 + k["elbowmanus"] * e * w)
        c1 = (k["alpha"] + k["elbowalpha"] * e + k["manusalpha"] * w + k["sweepalpha"] * s
              + k["dihedralalpha"] * d + k["elbowmanusalpha"] * e * w)
        return c0, c1, k["alpha2"] + 0 * e, k["alpha3"] + 0 * e

    def CL(self, e, w, s, d, a):
        c0, c1, c2, c3 = self.CL_poly(e, w, s, d)
        return c0 + c1 * a + c2 * a ** 2 + c3 * a ** 3

    def CL_alpha(self, e, w, s, d, a):
        """Per radian."""
        c0, c1, c2, c3 = self.CL_poly(e, w, s, d)
        return (c1 + 2 * c2 * a + 3 * c3 * a ** 2) / (np.pi / 180)

    def CD(self, e, w, a):
        k = self.c["CD"]
        return (k["intercept"] + k["elbow"] * e + k["manus"] * w + k["alpha"] * a
                + k["alpha2"] * a ** 2 + k["elbowalpha"] * e * a + k["manusalpha"] * w * a
                + k["elbowalpha2"] * e * a ** 2 + k["manusalpha2"] * w * a ** 2)

    def CD_alpha(self, e, w, a):
        k = self.c["CD"]
        return (k["alpha"] + 2 * k["alpha2"] * a + k["elbowalpha"] * e + k["manusalpha"] * w
                + 2 * k["elbowalpha2"] * e * a + 2 * k["manusalpha2"] * w * a) / (np.pi / 180)

    def Cm(self, e, w, s, d, CL):
        k = self.c["Cm"]
        return (k["intercept"] + k["elbow"] * e + k["manus"] * w + k["sweep"] * s
                + k["dihedral"] * d + k["CL"] * CL + k["elbow2"] * e ** 2 + k["elbow3"] * e ** 3
                + k["manus2"] * w ** 2 + k["elbowmanus"] * e * w + k["elbowsweep"] * e * s
                + k["manussweep"] * w * s + k["elbowCL"] * e * CL + k["manusCL"] * w * CL
                + k["sweepCL"] * s * CL + k["dihedralCL"] * d * CL
                + k["elbowmanussweep"] * e * w * s + k["elbowmanusCL"] * e * w * CL
                + k["elbowsweepCL"] * e * s * CL + k["manussweepCL"] * w * s * CL
                + k["elbowmanussweepCL"] * e * w * s * CL)

    def dCm_dCL(self, e, w, s, d):
        k = self.c["Cm"]
        return (k["CL"] + k["elbowCL"] * e + k["manusCL"] * w + k["sweepCL"] * s
                + k["dihedralCL"] * d + k["elbowmanusCL"] * e * w + k["elbowsweepCL"] * e * s
                + k["manussweepCL"] * w * s + k["elbowmanussweepCL"] * e * w * s)

    def CL_q(self, e, w, s, d):
        k = self.c["dCLdq"]
        return (k["intercept"] + k["elbow"] * e + k["manus"] * w + k["sweep"] * s
                + k["dihedral"] * d + k["elbowmanus"] * e * w + k["elbowsweep"] * e * s
                + k["manussweep"] * w * s + k["elbowmanussweep"] * e * w * s)

    def Cm_q(self, e, w, s, d, CL_q):
        k = self.c["dCmdq"]
        return (k["intercept"] + k["elbow"] * e + k["manus"] * w + k["sweep"] * s
                + k["dihedral"] * d + k["CL_q"] * CL_q)


# ---------------------------------------------------------------------------------------
# Trim and linear model
# ---------------------------------------------------------------------------------------

@dataclass
class Trim:
    ok: np.ndarray           # trim exists
    alpha: np.ndarray        # deg
    CL: np.ndarray
    CD: np.ndarray
    U0: np.ndarray           # m/s
    gamma: np.ndarray        # rad, flight-path angle (negative: descending)


def trim(coef: Coefficients, e, w, s, d, mass: float = MASS, S: float = S_MAX) -> Trim:
    e, w, s, d = (np.asarray(x, float) for x in np.broadcast_arrays(e, w, s, d))
    Cm0 = coef.Cm(e, w, s, d, 0.0)
    cmcl = coef.dCm_dCL(e, w, s, d)
    CLt = -Cm0 / cmcl
    ok = ~(((Cm0 < 0) & (cmcl < 0)) | ((Cm0 > 0) & (cmcl > 0)))
    c0, c1, c2, c3 = coef.CL_poly(e, w, s, d)
    alpha = np.full(e.shape, np.nan)
    for idx in np.ndindex(e.shape):
        r = np.roots([c3[idx], c2[idx], c1[idx], c0[idx] - CLt[idx]])
        r = r[np.abs(r.imag) < 1e-9].real
        if len(r):
            alpha[idx] = r[np.argmin(np.abs(r))]
    CL = coef.CL(e, w, s, d, alpha)
    CD = coef.CD(e, w, alpha)
    ok &= np.isfinite(alpha) & (CL > 0) & (CD > 0)
    W = mass * G
    gamma = -np.arctan2(CD, CL)                 # tan|gamma| = D / L
    U0 = np.sqrt(2 * W * np.cos(gamma) / (RHO * S * np.where(CL > 0, CL, np.nan)))
    return Trim(ok, alpha, CL, CD, U0, gamma)


def system_matrix(coef: Coefficients, e, w, s, d, tr: Trim, mass: float = MASS,
                  S: float = S_MAX, c: float = C_MAX):
    """A (..., 4, 4) and the streamwise and vertical gust input columns (dyn_functions.py)."""
    a = tr.alpha
    CL_a = coef.CL_alpha(e, w, s, d, a)
    CD_a = coef.CD_alpha(e, w, a)
    Cm_a = coef.dCm_dCL(e, w, s, d) * CL_a
    CL_q = coef.CL_q(e, w, s, d)
    Cm_q = coef.Cm_q(e, w, s, d, CL_q)
    Iyy = coef.Iyy(e, w, s, d)
    U0, g0, CL, CD = tr.U0, tr.gamma, tr.CL, tr.CD
    mt = RHO * U0 * S / (2 * mass)
    It = RHO * U0 ** 2 * S * c / (2 * Iyy)
    wt = G / U0
    z = np.zeros_like(U0)
    A = np.stack([
        np.stack([-2 * mt * CD, mt * (CL - CD_a), z, -wt * np.cos(g0)], -1),
        np.stack([-2 * mt * CL, mt * (-CL_a - CD), 1 - mt * CL_q, -wt * np.sin(g0)], -1),
        np.stack([z, It * Cm_a, It * Cm_q, z], -1),
        np.stack([z, z, z + 1, z], -1)], -2)
    B_u = np.stack([2 * mt * CD, 2 * mt * CL, z, z], -1)
    B_w = np.stack([mt * (CL - CD_a), mt * (-CL_a - CD), It * Cm_a, z], -1)
    derivs = dict(CL_alpha=CL_a, CD_alpha=CD_a, Cm_alpha=Cm_a, CL_q=CL_q, Cm_q=Cm_q, Iyy=Iyy)
    return A, B_u, B_w, derivs


def modes(A: np.ndarray) -> dict:
    """Short-period and phugoid eigenvalues: of the two oscillatory pairs, the faster is the
    short period. Natural frequency |lambda| and damping ratio -Re/|lambda| (PNAS Methods).
    For configurations without two oscillatory pairs the entries are NaN."""
    lam = np.linalg.eigvals(A)
    osc = np.abs(lam.imag) > 1e-12
    n_osc = osc.sum(-1)
    mag = np.abs(lam)
    big = np.where(osc, mag, -1).max(-1)
    small = np.where(osc, mag, np.inf).min(-1)
    pick = lambda m: np.take_along_axis(lam, np.argmax(osc & np.isclose(mag, m[..., None]), -1)[..., None], -1)[..., 0]
    two = n_osc == 4
    sp, ph = pick(big), pick(small)
    sp = np.where(two, sp, np.nan)
    ph = np.where(two, ph, np.nan)
    return dict(eig=lam, sp=sp, ph=ph, omega_sp=np.abs(sp), zeta_sp=-sp.real / np.abs(sp),
                omega_ph=np.abs(ph), zeta_ph=-ph.real / np.abs(ph))


def sweep_grid(coef: Coefficients | None = None, sweeps=SWEEPS, dihedrals=DIHEDRALS,
               elbows=ELBOWS, wrists=WRISTS) -> dict:
    """All configurations of run_script.py, flattened, with trim, derivatives and modes."""
    coef = Coefficients() if coef is None else coef
    S_, D_, E_, W_ = (x.ravel() for x in np.meshgrid(sweeps, dihedrals, elbows, wrists,
                                                       indexing="ij"))
    tr = trim(coef, E_, W_, S_, D_)
    A, B_u, B_w, der = system_matrix(coef, E_, W_, S_, D_, tr)
    good = tr.ok
    md = modes(np.where(good[:, None, None], A, np.eye(4)))
    return dict(sweep=S_, dihedral=D_, elbow=E_, wrist=W_, trim=tr, A=A, B_u=B_u, B_w=B_w,
                derivs=der, modes=md, ok=good, static_stable=der["Cm_alpha"] / der["CL_alpha"] < 0,
                in_hull=in_hull(E_, W_))


def paper_selection(g: dict) -> np.ndarray:
    """The configurations the paper analyses (analyze_data.R): inside the joint hull, trim
    angle of attack below 5 deg, glide angle shallower than 45 deg, dihedral 20 deg, forward
    or zero sweep (sweep <= 0 in the code's sign)."""
    tr = g["trim"]
    return (g["ok"] & g["in_hull"] & (tr.alpha < 5) & (tr.gamma > -np.radians(45))
            & (g["dihedral"] == 20) & (g["sweep"] <= 0))


# ---------------------------------------------------------------------------------------
# Benchmark body
# ---------------------------------------------------------------------------------------

@register("gull_glider", "4-fluid",
          "Gliding gull, longitudinal dynamics after Harvey and Inman 2022 (PNAS), with "
          "wing morphing (elbow, wrist, shoulder sweep and dihedral); inputs: streamwise and "
          "vertical gusts.")
def gull_glider(elbow: float = 130.0, wrist: float = 126.0, sweep: float = -15.0,
                dihedral: float = 20.0, sigma_T2: float = 1e-6) -> StateSpaceBody:
    """Linear model about the trimmed glide for one wing configuration. The default is the
    paper's Fig. 6 configuration (elbow 130 deg, 15 deg forward sweep) at a stable wrist
    angle."""
    coef = Coefficients()
    tr = trim(coef, elbow, wrist, sweep, dihedral)
    if not tr.ok:
        raise ValueError("this wing configuration cannot trim")
    A, B_u, B_w, der = system_matrix(coef, elbow, wrist, sweep, dihedral, tr)
    GG = sigma_T2 * np.diag([1.0, 1.0, 1.0, 0.0])
    sel = lambda *i: np.eye(4)[list(i)]
    params = dict(elbow=elbow, wrist=wrist, sweep=sweep, dihedral=dihedral,
                  U0=float(tr.U0), gamma0=float(tr.gamma), alpha0=float(tr.alpha),
                  CL=float(tr.CL), CD=float(tr.CD), sigma_T2=sigma_T2,
                  **{k: float(v) for k, v in der.items()})
    return StateSpaceBody(A_=A, Bs={"streamwise_gust": B_u[:, None], "vertical_gust": B_w[:, None]},
                          GG_=GG, parts={"speed": sel(0), "angle_of_attack": sel(1),
                                         "pitch_rate": sel(2), "pitch": sel(3)},
                          positions=sel(0, 1, 2), sigma_T2=sigma_T2,
                          name=f"gull glider e={elbow:g} w={wrist:g}", params=params,
                          build=gull_glider)
