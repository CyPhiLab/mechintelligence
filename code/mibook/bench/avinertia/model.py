# SPDX-License-Identifier: GPL-3.0-or-later
#
# A Python translation of AvInertia (version 0.0.2), Copyright (C) Christina Harvey,
# Vikram B. Baliga and Jasmin C. M. Wong, distributed under the GNU General Public License
# version 3: https://github.com/charvey23/AvInertia. This file is a derivative work and is
# distributed under the same licence. Each function names the R function it translates.
"""Inertia of a bird as a composite of simple solids, after AvInertia (Harvey et al. 2022).

Reference: C. Harvey, V. B. Baliga, J. C. M. Wong, D. L. Altshuler and D. J. Inman, "Birds
can transition between stable and unstable states via wing morphing", Nature 603, 648-653
(2022), and its Supplementary Information, which documents each approximation.

Frame: the vehicle reference point (VRP) at the clavicle on the centre line, x forward,
y to the right wing, z down. SI units throughout. The body (head, neck, torso with legs,
tail) is fixed for a specimen. The wing depends on ten landmark points, so every function
that takes points accepts a leading batch axis: many wing shapes are evaluated at once.

The translation keeps the R code's choices, including two that look unusual: the solid-cone
tensor of the head is about the cone's base but is treated as about its centre of gravity
(`massprop_head`), and a triangular plate's centroid is divided by the unsigned area
(`calc_inertia_platetri`). Changing either would change the published numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DENSITY = {"Muscle": 1100.0, "Bone": 2060.0, "Skin": 1060.0, "Cortex": 1150.0,
           "Medullary": 80.0}
"""Material densities (kg/m^3) used for the paper (AnalysisFunctions/run_script.R)."""


@dataclass
class Mass:
    """Moment of inertia tensor I (about the VRP unless stated), centre of gravity CG and
    mass m. I has shape (..., 3, 3) and CG shape (..., 3)."""
    I: np.ndarray
    CG: np.ndarray
    m: float | np.ndarray

    def __add__(self, other: "Mass") -> "Mass":
        m = self.m + other.m
        m_ = np.asarray(m)[..., None]
        CG = (np.asarray(self.m)[..., None] * self.CG + np.asarray(other.m)[..., None] * other.CG) / m_
        return Mass(self.I + other.I, CG, m)


# ---------------------------------------------------------------------------------------
# Mathematical functions (R/mathematicalfunctions.R)
# ---------------------------------------------------------------------------------------

def _cosd(a):
    return np.cos(np.deg2rad(a))


def _sind(a):
    return np.sin(np.deg2rad(a))


def rotx(angle: float) -> np.ndarray:
    """Rotation about x by `angle` degrees, counterclockwise positive (rotx)."""
    c, s = _cosd(angle), _sind(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def calc_rot(z_vector, x_vector) -> np.ndarray:
    """Rotation from the VRP frame to an object frame whose z and x axes are given in VRP
    coordinates; rows are the object's x, y, z axes (calc_rot)."""
    uz = _unit(np.asarray(z_vector, float))
    ux = _unit(np.asarray(x_vector, float))
    uy = _unit(np.cross(uz, ux))
    return np.stack([ux, uy, uz], axis=-2)


def parallelaxis(I, offset_vec, m, cg_a: str) -> np.ndarray:
    """Parallel axis theorem (parallelaxis). With cg_a == "CG", I is about the centre of
    gravity and the result is about the point A, where offset_vec points from the CG to A;
    with "A" the shift is removed instead."""
    sign = 1.0 if cg_a == "CG" else -1.0
    d = np.asarray(offset_vec, float)
    dd = np.sum(d * d, axis=-1)[..., None, None]
    outer = d[..., :, None] * d[..., None, :]
    m = np.asarray(m, float)[..., None, None]
    return I + sign * m * (dd * np.eye(3) - outer)


def _rot_tensor(R, I):
    """Express a tensor given in the object frame in the VRP frame: R^T I R."""
    return np.swapaxes(R, -1, -2) @ I @ R


def _rot_vec(R, v):
    return np.einsum("...ji,...j->...i", R, v)       # R^T v


def _apply(R, v):
    return np.einsum("...ij,...j->...i", R, v)       # R v


# ---------------------------------------------------------------------------------------
# Tensors of simple solids (R/inertiafunctions.R); reference: DTIC AD0274936
# ---------------------------------------------------------------------------------------

def calc_inertia_cylsolid(r, h, m):
    return m * np.diag([(3 * r ** 2 + h ** 2) / 12, (3 * r ** 2 + h ** 2) / 12, 0.5 * r ** 2])


def calc_inertia_cylhollow(r_out, r_in, h, m):
    tr = r_out ** 2 + r_in ** 2
    return m * np.diag([(3 * tr + h ** 2) / 12, (3 * tr + h ** 2) / 12, 0.5 * tr])


def calc_inertia_ellipse(a, b, c, m):
    return 0.2 * m * np.diag([b ** 2 + c ** 2, a ** 2 + c ** 2, a ** 2 + b ** 2])


def calc_inertia_conesolid(r, h, m):
    """Solid circular cone, about the centre of its base."""
    return 0.1 * m * np.diag([1.5 * r ** 2 + h ** 2, 1.5 * r ** 2 + h ** 2, 3 * r ** 2])


def calc_inertia_ellcone(A, B, l, m):
    """Solid elliptical cone, about the centre of its wide base."""
    x2, y2, z2 = 0.15 * A ** 2, 0.15 * B ** 2, 0.1 * l ** 2
    return m * np.diag([y2 + z2, x2 + z2, x2 + y2])


def calc_inertia_ellcyl(a, b, l, m):
    return m * np.diag([b ** 2 / 4 + l ** 2 / 12, a ** 2 / 4 + l ** 2 / 12, (a ** 2 + b ** 2) / 4])


def calc_inertia_pyrasolid(w, h, m):
    """Solid square pyramid, about the centre of its base."""
    return 0.05 * m * np.diag([w ** 2 + 2 * h ** 2, w ** 2 + 2 * h ** 2, 2 * w ** 2])


def calc_inertia_platerect(w, h, m):
    return m / 12 * np.diag([w ** 2 + h ** 2, h ** 2, w ** 2])


def calc_inertia_platetri(pts, A, rho, t):
    """Flat triangular plate whose three points (..., 3, 3) share a z coordinate
    (calc_inertia_platetri; DTIC ADA183444 eqs 2.16-2.20). Returns (I about the plate's CG,
    CG in the frame of the points)."""
    P = np.concatenate([pts, pts[..., :1, :]], axis=-2)
    x, y = P[..., 0], P[..., 1]
    cr = x[..., :3] * y[..., 1:] - x[..., 1:] * y[..., :3]
    A_ = np.asarray(A)[..., None]
    x1 = np.sum(cr * (x[..., :3] + x[..., 1:]), axis=-1) / (6 * A_[..., 0])
    y1 = np.sum(cr * (y[..., :3] + y[..., 1:]), axis=-1) / (6 * A_[..., 0])
    CG = np.stack([x1, y1, P[..., 0, 2]], axis=-1)
    Q = P - CG[..., None, :]
    x, y = Q[..., 0], Q[..., 1]
    cr = x[..., :3] * y[..., 1:] - x[..., 1:] * y[..., :3]
    x2 = np.sum(cr * (x[..., :3] ** 2 + x[..., :3] * x[..., 1:] + x[..., 1:] ** 2), axis=-1)
    y2 = np.sum(cr * (y[..., :3] ** 2 + y[..., :3] * y[..., 1:] + y[..., 1:] ** 2), axis=-1)
    ixy = 0.5 * np.sum(cr * (2 * x[..., :3] * y[..., :3] + x[..., :3] * y[..., 1:]
                             + y[..., :3] * x[..., 1:] + 2 * x[..., 1:] * y[..., 1:]), axis=-1)
    I = np.zeros(x2.shape + (3, 3))
    I[..., 0, 0] = y2
    I[..., 1, 1] = x2
    I[..., 2, 2] = x2 + y2
    I[..., 0, 1] = I[..., 1, 0] = -ixy
    k = (rho * np.asarray(t) / 12.0)[..., None, None]
    return k * I, CG


# ---------------------------------------------------------------------------------------
# Anatomical components (R/anatomicalmassprop.R)
# ---------------------------------------------------------------------------------------

_ONES = np.ones(3) / np.sqrt(3.0)
_NEG_Y = np.array([0.0, -1.0, 0.0])


def _along(start, end, temp):
    z = np.asarray(end, float) - np.asarray(start, float)
    return calc_rot(z, np.cross(z, temp))


def massprop_bones(m, l, r_out, r_in, rho, start, end) -> Mass:
    """A bone as a hollow cylinder closed by two solid end caps sized to match the bone's
    volume m/rho (massprop_bones)."""
    vol = m / rho
    r_cap = r_out
    t_cap = 0.5 * (-(l * (r_out ** 2 - r_in ** 2)) / r_in ** 2 + vol / (np.pi * r_in ** 2))
    m_cap = rho * np.pi * t_cap * r_cap ** 2
    m_cyl = m - 2 * m_cap
    l_cyl = l - 2 * t_cap
    R = _along(start, end, _ONES)
    I_cyl = calc_inertia_cylhollow(r_out, r_in, l_cyl, m_cyl)
    I_cap = calc_inertia_cylsolid(r_cap, t_cap, m_cap)
    off = _apply(R, start)
    o_cyl = np.array([0, 0, 0.5 * l]) + off
    o_c1 = np.array([0, 0, 0.5 * t_cap]) + off
    o_c2 = np.array([0, 0, l - 0.5 * t_cap]) + off
    I = (parallelaxis(I_cyl, -o_cyl, m_cyl, "CG") + parallelaxis(I_cap, -o_c1, m_cap, "CG")
         + parallelaxis(I_cap, -o_c2, m_cap, "CG"))
    return Mass(_rot_tensor(R, I), _rot_vec(R, o_cyl), m)


def massprop_muscles(m, rho, l, start, end) -> Mass:
    """Muscles as a solid cylinder along the bone (massprop_muscles)."""
    r = np.sqrt(m / (rho * np.pi * l))
    R = _along(start, end, _ONES)
    off = _apply(R, start)
    o = np.array([0, 0, 0.5 * l]) + off
    I = parallelaxis(calc_inertia_cylsolid(r, l, m), -o, m, "CG")
    return Mass(_rot_tensor(R, I), _rot_vec(R, o), m)


def massprop_skin(m, rho, pts) -> Mass:
    """Skin, coverts or tertiaries as a flat triangular plate through three points
    (..., 3, 3) (massprop_skin)."""
    pts = np.asarray(pts, float)
    cr = np.cross(pts[..., 2, :] - pts[..., 0, :], pts[..., 2, :] - pts[..., 1, :])
    A = 0.5 * np.linalg.norm(cr, axis=-1)
    t = (m / rho) / A
    R = calc_rot(cr, pts[..., 2, :] - pts[..., 0, :])
    adj = np.einsum("...ij,...kj->...ki", R, pts)
    I_s, CG_s = calc_inertia_platetri(adj, A, rho, t)
    I = parallelaxis(I_s, -CG_s, m, "CG")
    return Mass(_rot_tensor(R, I), _rot_vec(R, CG_s), m)


def massprop_pm(m, pt) -> Mass:
    """A point mass (massprop_pm)."""
    pt = np.asarray(pt, float)
    return Mass(parallelaxis(np.zeros(pt.shape[:-1] + (3, 3)), -pt, m, "CG"), pt, m)


def massprop_feathers(m_f, l_c, l_r_cor, w_cal, r_b, d_b, rho_cor, rho_med, w_vp, w_vd,
                      angle) -> Mass:
    """A flight feather in its own frame: hollow-cylinder calamus, square-pyramid rachis
    (cortex shell around a medullary core) and two rectangular vanes of barbs
    (massprop_feathers). Returns I about the feather's CG and the CG measured from the start
    of the calamus, in the frame whose z axis runs from the start to the tip of the
    feather."""
    r_cor = 0.5 * w_cal
    n_barb = l_r_cor / (d_b + 2 * r_b)
    m_vp = rho_cor * n_barb * w_vp * np.pi * r_b ** 2
    m_vd = rho_cor * n_barb * w_vd * np.pi * r_b ** 2
    m_rc = m_f - m_vp - m_vd
    r_med = np.sqrt((m_rc - rho_cor * r_cor ** 2 * (np.pi * l_c + 4 / 3 * l_r_cor))
                    / (4 / 3 * l_r_cor * (rho_med - rho_cor) - l_c * np.pi * rho_cor))
    l_r_med = l_r_cor
    if not (r_med <= r_cor and m_rc >= 0):
        raise ValueError("feather medullary radius larger than the cortex radius")
    m_c = rho_cor * np.pi * l_c * (r_cor ** 2 - r_med ** 2)
    m_r_cor = 4 / 3 * rho_cor * (l_r_cor * r_cor ** 2 - l_r_med * r_med ** 2)
    m_r_med = 4 / 3 * rho_med * r_med ** 2 * l_r_med
    mass_outer = 4 / 3 * rho_cor * r_cor ** 2 * l_r_cor
    mass_inner = 4 / 3 * rho_cor * r_med ** 2 * l_r_med
    # calamus
    I_cCG = calc_inertia_cylhollow(r_cor, r_med, l_c, m_c)
    CG_c1 = np.array([0, 0, 0.5 * l_c])
    I_c1 = parallelaxis(I_cCG, -CG_c1, m_c, "CG")
    # rachis
    I_r_med = calc_inertia_pyrasolid(2 * r_med, l_r_med, m_r_med)
    I_r_cor = (calc_inertia_pyrasolid(2 * r_cor, l_r_cor, mass_outer)
               - calc_inertia_pyrasolid(2 * r_med, l_r_med, mass_inner))
    pyr = np.array([0, 0, 0.25])
    CG_r_cor1 = (mass_outer * pyr * l_r_cor - mass_inner * pyr * l_r_med) / m_r_cor
    CG_r_med1 = pyr * l_r_med
    # vanes
    I_vd1 = calc_inertia_platerect(w_vd, l_r_cor, m_vd)
    CG_vd1 = np.array([0, -0.5 * w_vd, 0.5 * l_r_cor])
    I_vp1 = calc_inertia_platerect(w_vp, l_r_cor, m_vp)
    CG_vp1 = np.array([0, 0.5 * w_vp, 0.5 * l_r_cor])
    a = np.rad2deg(np.arctan(r_cor / l_r_cor))
    rot_vd, rot_vp = rotx(-a), rotx(a)
    I_vd2 = rot_vd @ I_vd1 @ rot_vd.T
    I_vp2 = rot_vp @ I_vp1 @ rot_vp.T
    CG_vd3 = rot_vd @ CG_vd1 - np.array([0, r_cor, 0])
    CG_vp3 = rot_vp @ CG_vp1 + np.array([0, r_cor, 0])
    I_vd3 = parallelaxis(I_vd2, -CG_vd3, m_vd, "CG")
    I_vp3 = parallelaxis(I_vp2, -CG_vp3, m_vp, "CG")
    # rachis and vanes together, rotated into the calamus frame
    m_r = m_r_cor + m_r_med + m_vd + m_vp
    I_vr1 = I_vp3 + I_vd3 + I_r_med + I_r_cor
    CG_vr1 = (CG_r_cor1 * m_r_cor + CG_r_med1 * m_r_med + CG_vd3 * m_vd + CG_vp3 * m_vp) / m_r
    Ra = rotx(angle)
    I_vr2 = Ra @ I_vr1 @ Ra.T
    CG_vr2 = Ra @ CG_vr1
    I_vrCG = parallelaxis(I_vr2, CG_vr2, m_r, "A")
    CG_vr3 = CG_vr2 + np.array([0, 0, l_c])
    I_vr3 = parallelaxis(I_vrCG, -CG_vr3, m_r, "CG")
    I_1 = I_c1 + I_vr3
    CG_1 = (CG_c1 * m_c + CG_vr3 * m_r) / m_f
    full = rotx(np.rad2deg(np.arctan(l_r_cor * abs(_sind(angle))
                                     / (l_c + l_r_cor * abs(_cosd(angle))))))
    I_2 = full @ I_1 @ full.T
    CG_2 = full @ CG_1
    return Mass(parallelaxis(I_2, CG_2, m_f, "A"), CG_2, m_f)


def structural2VRP_feat(feat: Mass, start, end, normal) -> Mass:
    """Place a feather (I about its CG, CG from its start) with its start, tip and dorsal
    normal given in the VRP frame (structural2VRP_feat)."""
    R = calc_rot(np.asarray(end) - np.asarray(start), normal)
    CG_3 = feat.CG + _apply(R, start)
    I_3 = parallelaxis(feat.I, -CG_3, feat.m, "CG")
    return Mass(_rot_tensor(R, I_3), _rot_vec(R, CG_3), feat.m)


def orient_feather(no_pri, no_sec, Pt1, Pt2, Pt3, Pt4, Pt8, Pt9, Pt10, Pt11, Pt12):
    """Start, tip and normal of every flight feather from the landmarks (orient_feather).
    Points have shape (..., 3); outputs have shape (..., no_pri + no_sec, 3), primaries
    P1..Pn first, then secondaries S1..Sn."""
    starts, ends = [], []
    k = 1
    for i in range(1, no_pri + 1):
        if i < no_pri - 2:
            starts.append(Pt3 + i / 8 * (Pt4 - Pt3))
            ends.append(Pt10 + i / 7 * (Pt9 - Pt10))
        else:
            starts.append(Pt4)
            ends.append(Pt9 + k / (no_pri - 7) * (Pt8 - Pt9))
            k += 1
    sec_vec = Pt11 - Pt10
    tt = (Pt12[..., 1] - Pt10[..., 1]) / sec_vec[..., 1]
    sec_end_a = np.stack([tt * sec_vec[..., 0] + Pt10[..., 0], Pt12[..., 1],
                          tt * sec_vec[..., 2] + Pt10[..., 2]], axis=-1)
    sec_end = np.where((Pt11[..., 1] < 0)[..., None], sec_end_a, Pt11)
    for i in range(1, no_sec + 1):
        starts.append(Pt3 + (i - 1) / (no_sec - 1) * (Pt2 - Pt3))
        ends.append(Pt10 + (i - 1) / (no_sec - 1) * (sec_end - Pt10))
    starts = np.stack(np.broadcast_arrays(*starts), axis=-2)
    ends = np.stack(np.broadcast_arrays(*ends), axis=-2)
    n_pri = np.cross((Pt3 - Pt4)[..., None, :], ends[..., :no_pri, :] - Pt4[..., None, :])
    n_sec = np.cross((Pt2 - Pt3)[..., None, :], ends[..., no_pri:, :] - Pt3[..., None, :])
    return starts, ends, np.concatenate([n_pri, n_sec], axis=-2)


def massprop_neck(m, r, l, start, end) -> Mass:
    R = _along(start, end, _NEG_Y)
    o = np.array([0, 0, 0.5 * l]) + _apply(R, start)
    I = parallelaxis(calc_inertia_cylsolid(r, l, m), -o, m, "CG")
    return Mass(_rot_tensor(R, I), 0.5 * (np.asarray(start) + np.asarray(end)), m)


def massprop_head(m, r, l, start, end) -> Mass:
    """Head (with the neck when it is retracted) as a solid cone pointing forward
    (massprop_head). As in the R code, the cone's base-centred tensor is shifted as if it
    were about the CG."""
    R = _along(start, end, _NEG_Y)
    o = np.array([0, 0, 0.25 * l]) + _apply(R, start)
    I = parallelaxis(calc_inertia_conesolid(r, l, m), -o, m, "CG")
    return Mass(_rot_tensor(R, I), _rot_vec(R, o), m)


def massprop_tail(m, l, w, start, end) -> Mass:
    """Tail as a flat rectangular plate (massprop_tail)."""
    R = _along(start, end, _NEG_Y)
    o = _apply(R, start) + np.array([0, 0, 0.5 * l])
    I = parallelaxis(calc_inertia_platerect(w, l, m), -o, m, "CG")
    return Mass(_rot_tensor(R, I), _rot_vec(R, o), m)


def _density_fit(m_body, A, v_ell, CG_x, CG_ell, rho_avg):
    """The torso densities of option 3 in massprop_torso. The R code minimizes
    density_optimizer with pracma::lsqnonlin; that objective is quadratic in the front
    density (rho = |x| + 200 >= 200) apart from a constant penalty, so its minimizer is found
    here in closed form."""
    Ainv = np.linalg.inv(A)
    b0 = Ainv @ np.array([m_body, m_body * CG_x])
    b1 = Ainv @ np.array([-v_ell, -v_ell * CG_ell])
    # err(rho) = (rho - ra)^2 + (b0[0] + rho b1[0] - ra)^2 + (b0[1] + rho b1[1] - ra)^2
    rho = (rho_avg - b1[0] * (b0[0] - rho_avg) - b1[1] * (b0[1] - rho_avg)) / (1 + b1 @ b1)
    rho = max(rho, 200.0)
    out = b0 + rho * b1
    return rho, out[0], out[1]


def massprop_torso(m_true, m_legs, w_max, h_max, l_bmax, w_leg, l_leg, l_tot, CG_true_x,
                   CG_true_z, start, end) -> Mass:
    """Torso as a hemi-ellipsoid, a partial elliptical cone and a back section, with the
    legs as point masses; densities chosen to put the CG where it was measured
    (massprop_torso). Returns the Mass and, as attribute `option`, which back section and
    density rule was used (1, 2 or 3)."""
    R = _along(start, end, _NEG_Y)
    h_leg = w_leg * h_max / w_max
    l_par = l_leg - l_bmax
    l_full = l_par / (w_max - w_leg) * w_max
    l_end = l_tot - l_leg
    CG_leg_r = np.array([0.5 * h_leg, 0.5 * w_leg, l_leg])
    CG_leg_l = np.array([0.5 * h_leg, -0.5 * w_leg, l_leg])
    leg_r = massprop_pm(0.5 * m_legs, CG_leg_r)
    leg_l = massprop_pm(0.5 * m_legs, CG_leg_l)
    m_body = m_true - m_legs
    CG_body_z = (CG_true_z * m_true - CG_leg_l[0] * 0.5 * m_legs - CG_leg_r[0] * 0.5 * m_legs) / m_body
    CG_body_x = (CG_true_x * m_true - CG_leg_l[2] * 0.5 * m_legs - CG_leg_r[2] * 0.5 * m_legs) / m_body
    v_ell = np.pi * w_max * h_max * l_bmax / 6
    CG_ell = 5 / 8 * l_bmax
    v_full = np.pi * w_max * h_max * l_full / 12
    v_cut = np.pi * w_leg * h_leg * (l_full - l_par) / 12
    v_par = v_full - v_cut
    CG_full = l_bmax + 0.25 * l_full
    CG_cut = l_leg + 0.25 * (l_full - l_par)
    CG_par = (v_full * CG_full - v_cut * CG_cut) / v_par
    v_end_1 = np.pi * w_leg * h_leg * l_end / 4
    CG_end_1 = l_leg + 0.5 * l_end
    rho1 = m_body / (v_ell + v_par + v_end_1)
    est1 = rho1 * (v_ell * CG_ell + v_par * CG_par + v_end_1 * CG_end_1) / m_body
    if abs(est1 - CG_body_x) > 0.05 * l_tot:
        v_end_2 = np.pi * w_leg * h_leg * l_end / 12
        CG_end_2 = l_leg + 0.25 * l_end
        rho2 = m_body / (v_ell + v_par + v_end_2)
        est2 = rho2 * (v_ell * CG_ell + v_par * CG_par + v_end_2 * CG_end_2) / m_body
        if abs(est2 - CG_body_x) > 0.05 * l_tot:
            A = np.array([[v_par, v_end_1], [v_par * CG_par, v_end_1 * CG_end_1]])
            rho_ell, rho_par, rho_back = _density_fit(m_body, A, v_ell, CG_body_x, CG_ell, rho2)
            if rho_par < 0 or rho_back < 0:
                rho_ell, rho_par, rho_back = _density_fit(m_body, A, v_ell,
                                                          CG_body_x + 0.05 * l_tot, CG_ell, rho2)
            option, v_end, CG_end = 3, v_end_1, CG_end_1
        else:
            option, v_end, CG_end = 2, v_end_2, CG_end_2
            rho_ell = rho_par = rho_back = rho2
    else:
        option, v_end, CG_end = 1, v_end_1, CG_end_1
        rho_ell = rho_par = rho_back = rho1
    m_ell, m_par, m_end = rho_ell * v_ell, rho_par * v_par, rho_back * v_end
    if min(m_ell, m_par, m_end) < 0:
        raise ValueError("a torso section has negative mass")
    # front: hemi-ellipsoid
    I_ell1 = calc_inertia_ellipse(h_max / 2, w_max / 2, l_bmax, m_ell)
    I_ellCG = parallelaxis(I_ell1, np.array([0, 0, -3 / 8 * l_bmax]), m_ell, "A")
    CG_ell2 = np.array([CG_body_z, 0, 5 / 8 * l_bmax])
    I_ell = parallelaxis(I_ellCG, -CG_ell2, m_ell, "CG")
    # middle: partial elliptical cone
    CG_full1 = np.array([0, 0, 0.25 * l_full])
    CG_cut1 = np.array([0, 0, l_par + 0.25 * (l_full - l_par)])
    CG_par1 = (v_full * CG_full1 - v_cut * CG_cut1) / v_par
    m_cut, m_full = rho_par * v_cut, rho_par * v_full
    I_full = calc_inertia_ellcone(0.5 * h_max, 0.5 * w_max, l_full, m_full)
    I_cut = calc_inertia_ellcone(0.5 * h_leg, 0.5 * w_leg, l_full - l_par, m_cut)
    I_cut2 = parallelaxis(I_cut, -np.array([0, 0, 0.25 * (l_full - l_par)]), m_cut, "A")
    I_cut3 = parallelaxis(I_cut2, CG_cut1, m_cut, "CG")
    I_parCG = parallelaxis(I_full - I_cut3, CG_par1, m_par, "A")
    CG_par2 = CG_par1 + np.array([CG_body_z, 0, l_bmax])
    I_par = parallelaxis(I_parCG, -CG_par2, m_par, "CG")
    # back
    if option == 2:
        CG_end1 = np.array([0, 0, 0.25 * l_end])
        I_end1 = calc_inertia_ellcone(0.5 * h_leg, 0.5 * w_leg, l_end, m_end)
        I_endCG = parallelaxis(I_end1, CG_end1, m_end, "A")
        CG_end2 = CG_end1 + np.array([CG_body_z, 0, l_leg])
    else:
        CG_end2 = np.array([0, 0, 0.5 * l_end]) + np.array([CG_body_z, 0, l_leg])
        I_endCG = calc_inertia_ellcyl(0.5 * h_leg, 0.5 * w_leg, l_end, m_end)
    I_end = parallelaxis(I_endCG, -CG_end2, m_end, "CG")
    I_t = I_ell + I_par + I_end + leg_r.I + leg_l.I
    CG_t = (m_ell * CG_ell2 + m_par * CG_par2 + m_end * CG_end2
            + 0.5 * m_legs * CG_leg_l + 0.5 * m_legs * CG_leg_r) / m_true
    out = Mass(_rot_tensor(R, I_t), _rot_vec(R, CG_t), m_true)
    out.option = option
    return out


# ---------------------------------------------------------------------------------------
# Whole-bird assembly (R/mainrunfunctions.R)
# ---------------------------------------------------------------------------------------

@dataclass
class Specimen:
    """Everything AvInertia needs about one specimen apart from the wing shape, in SI units:
    `bird` holds the body and wing measurements under the R column names, `bones` maps bone
    name to (mass, length, outer radius, inner radius), `feathers` lists dicts with keys
    feather, m_f, l_cal, l_vane, w_cal, w_vp, w_vd, vane_angle (alula: mass only)."""
    species: str
    bird_id: str
    bird: dict
    bones: dict
    feathers: list
    density: dict = field(default_factory=lambda: dict(DENSITY))


def massprop_restbody(sp: Specimen) -> dict[str, Mass]:
    """Head, neck (if extended), torso with legs, and tail (massprop_restbody)."""
    b = sp.bird
    out = {}
    neck_start = np.zeros(3)
    if b["extend_neck"]:
        neck_end = np.array([b["neck_length"], 0, 0])
        head_end = neck_end + np.array([b["head_length"], 0, 0])
        out["head"] = massprop_head(b["head_mass"], 0.5 * b["head_height"], b["head_length"],
                                    neck_end, head_end)
        out["neck"] = massprop_neck(b["neck_mass"], 0.5 * b["neck_width"], b["neck_length"],
                                    neck_start, neck_end)
    else:
        head_end = np.array([b["head_length"], 0, 0])
        out["head"] = massprop_head(b["head_mass"] + b["neck_mass"], 0.5 * b["head_height"],
                                    b["head_length"], neck_start, head_end)
    tail_start = np.array([-(b["torsotail_length"] - b["tail_length"]), 0, 0])
    tail_end = np.array([-b["torsotail_length"], 0, 0])
    m_legs = b["right_leg_mass"] + b["left_leg_mass"]
    tail = massprop_tail(b["tail_mass"], b["tail_length"], b["tail_width"], tail_start, tail_end)
    m_torso = b["torsotail_mass"] - b["tail_mass"]
    l_torso = b["torsotail_length"] - b["tail_length"]
    CGx = (b["torsotail_mass"] * b["x_loc_TorsotailCoG"] - b["tail_mass"] * tail.CG[0]) / m_torso
    CGz = (b["torsotail_mass"] * b["z_loc_TorsotailCoG"] - b["tail_mass"] * tail.CG[2]) / m_torso
    out["torso"] = massprop_torso(m_torso, m_legs, b["body_width_max"], b["body_height_max"],
                                  b["x_loc_of_body_max"], b["body_width_at_leg_insert"],
                                  b["x_loc_leg_insertion"], l_torso, abs(CGx), CGz,
                                  neck_start, tail_start)
    out["tail"] = tail
    return out


def compute_feat_inertia(sp: Specimen) -> tuple[list[Mass], list[Mass]]:
    """Each primary and secondary in its own frame (compute_feat_inertia)."""
    b, rho = sp.bird, sp.density
    by_name = {f["feather"]: f for f in sp.feathers}
    pri = sorted((n for n in by_name if "P" in n), key=lambda n: int(n[1:]))
    sec = sorted((n for n in by_name if "S" in n), key=lambda n: int(n[1:]))

    def one(name):
        f = by_name[name]
        return massprop_feathers(f["m_f"], f["l_cal"], f["l_vane"], f["w_cal"], b["barb_radius"],
                                 b["barb_distance"], rho["Cortex"], rho["Medullary"], f["w_vp"],
                                 f["w_vd"], f["vane_angle"])
    if pri != [f"P{i}" for i in range(1, len(pri) + 1)] or \
            sec != [f"S{i}" for i in range(1, len(sec) + 1)]:
        raise ValueError("feathers must be numbered P1..Pn and S1..Sn without gaps")
    return [one(n) for n in pri], [one(n) for n in sec]


POINTS = ("Pt1", "Pt2", "Pt3", "Pt4", "Pt6", "Pt8", "Pt9", "Pt10", "Pt11", "Pt12")
"""Order of the ten landmarks of `clean_pts`: shoulder, elbow, wrist, end of the
carpometacarpus, leading edge in front of the wrist, tip of the most distal primary, tip of
the last primary modelled on the end of the carpometacarpus, tip of S1, tip of the last
secondary at the wing root, leading edge of the wing root."""


def massprop_birdwing(sp: Specimen, pts, feather_inertia=None) -> dict[str, Mass]:
    """One (right) wing for one or many shapes: pts has shape (..., 10, 3) in the order of
    POINTS (massprop_birdwing). Returns bones, muscles, feathers, skin and the whole wing."""
    b, rho = sp.bird, sp.density
    pts = np.asarray(pts, float)
    P = {n: pts[..., i, :] for i, n in enumerate(POINTS)}
    if feather_inertia is None:
        feather_inertia = compute_feat_inertia(sp)
    pri, sec = feather_inertia
    # bones
    bo = sp.bones
    hum = massprop_bones(*bo["Humerus"], rho["Bone"], P["Pt1"], P["Pt2"])
    uln = massprop_bones(*bo["Ulna"], rho["Bone"], P["Pt2"], P["Pt3"])
    rad = massprop_bones(*bo["Radius"], rho["Bone"], P["Pt2"], P["Pt3"])
    car = massprop_bones(*bo["Carpometacarpus"], rho["Bone"], P["Pt3"], P["Pt4"])
    wrist = massprop_pm(bo["Ulnare"][0] + bo["Radiale"][0], P["Pt3"])
    bones = hum + uln + rad + car + wrist
    # muscles
    brach = massprop_muscles(b["brachial_muscle_mass"], rho["Muscle"], bo["Humerus"][1],
                             P["Pt1"], P["Pt2"])
    abrach = massprop_muscles(b["antebrachial_muscle_mass"], rho["Muscle"], bo["Ulna"][1],
                              P["Pt2"], P["Pt3"])
    manus = massprop_muscles(b["manus_muscle_mass"], rho["Muscle"], bo["Carpometacarpus"][1],
                             P["Pt3"], P["Pt4"])
    muscles = brach + abrach + manus
    # flight feathers
    starts, ends, normals = orient_feather(len(pri), len(sec), P["Pt1"], P["Pt2"], P["Pt3"],
                                           P["Pt4"], P["Pt8"], P["Pt9"], P["Pt10"], P["Pt11"],
                                           P["Pt12"])
    feathers = None
    for i, f in enumerate(list(pri) + list(sec)):
        placed = structural2VRP_feat(f, starts[..., i, :], ends[..., i, :], normals[..., i, :])
        feathers = placed if feathers is None else feathers + placed
    m_alula = next(f["m_f"] for f in sp.feathers if f["feather"] == "alula")
    feathers = feathers + massprop_pm(m_alula, P["Pt6"])
    # tertiaries: two triangles between the wing root and the elbow
    edge = np.stack([P["Pt11"][..., 0], np.minimum(P["Pt12"][..., 1], P["Pt11"][..., 1]),
                     P["Pt12"][..., 2]], axis=-1)
    tert = 0.5 * b["tertiary_mass"]
    feathers = (feathers
                + massprop_skin(tert, rho["Cortex"], np.stack([P["Pt12"], edge, P["Pt2"]], -2))
                + massprop_skin(tert, rho["Cortex"], np.stack([P["Pt11"], P["Pt2"], edge], -2)))
    # skin and coverts take the rest of the measured wing mass
    m_skin = b["wing_mass"] - feathers.m - bones.m - muscles.m
    skin = massprop_skin(m_skin, rho["Skin"], np.stack([P["Pt12"], P["Pt6"], P["Pt2"]], -2))
    wing = bones + muscles + skin + feathers
    return dict(bones=bones, muscles=muscles, feathers=feathers, skin=skin, wing=wing)


@dataclass
class Bird:
    """Inertial properties of the whole bird for one or many wing shapes (symmetric wings).
    I_vrp is about the VRP; I is about the bird's CG (the paper's reported tensor)."""
    m: float
    CG: np.ndarray
    I_vrp: np.ndarray
    I: np.ndarray
    wing: Mass
    rest: dict


def combine_inertialprop(rest: dict[str, Mass], wing: Mass) -> Bird:
    """Body plus two mirror-image wings (combine_inertialprop with symmetric = TRUE: the
    left wing is the right wing reflected, so the xy and yz products cancel and the CG lies on
    the centre line)."""
    m = sum(c.m for c in rest.values()) + 2 * wing.m
    I = np.zeros(np.shape(wing.I))
    I_rest = sum(c.I for c in rest.values())
    for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 2)]:
        I[..., i, j] = I_rest[i, j] + 2 * wing.I[..., i, j]
    I[..., 2, 0] = I[..., 0, 2]
    CG = np.zeros(np.shape(wing.CG))
    for k in (0, 2):
        CG[..., k] = (sum(c.m * c.CG[k] for c in rest.values()) + 2 * wing.m * wing.CG[..., k]) / m
    I_cg = parallelaxis(I, -CG, m, "A")
    return Bird(m=m, CG=CG, I_vrp=I, I=I_cg, wing=wing, rest=rest)


def bird_inertia(sp: Specimen, pts) -> Bird:
    """The whole pipeline for one specimen and wing shapes pts (..., 10, 3)."""
    return combine_inertialprop(massprop_restbody(sp), massprop_birdwing(sp, pts)["wing"])


def shift_origin(mass: Mass, new_origin) -> np.ndarray:
    """Tensor of a component about another point, from its tensor about the VRP
    (shift_Iorigin in AnalysisFunctions/calc_MOIparams.R)."""
    I_cg = parallelaxis(mass.I, -mass.CG, mass.m, "A")
    return parallelaxis(I_cg, mass.CG - np.asarray(new_origin), mass.m, "CG")
