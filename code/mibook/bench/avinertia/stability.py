# SPDX-License-Identifier: GPL-3.0-or-later
#
# Translated from AvInertia's analysis scripts AnalysisFunctions/process_outputdata.R and
# AnalysisFunctions/calc_np_loc_functions.R (https://github.com/charvey23/AvInertia, GPL-3);
# see model.py for the licence note.
"""Neutral point, static margin and pitch agility of Harvey et al. (2022), Eqs (3)-(8).

The neutral point of the wing-body is estimated from the quarter chord of the wing's
standard mean chord (Eq. 7), non-dimensionalized by the specimen's largest root chord and
raised to the power 0.8 fitted to gull wing-body aerodynamics (Eq. 6). The static margin
(Eq. 8) is the distance from that neutral point to the centre of gravity, positive when the
centre of gravity is ahead (stable). Distances along x are measured from the humeral head.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import WingShapes
from .model import Bird

N_STRIPS = 1000
NP_EXPONENT = 0.8


def _lin(x1, x2, y1, y2, y):
    """calc_lin_edge."""
    with np.errstate(divide="ignore", invalid="ignore"):   # vertical segments are never
        m = (x1 - x2) / (y1 - y2)                          # selected by the conditions
        return m * y + (x1 - m * y1)


def _edges(peri, n_rows, y, coord):
    """Leading and trailing edge coordinate at span stations y (calc_LE_xz, calc_TE_xz).
    peri: (N, 8, 3) perimeter points Pt12, Pt6, Pt7, Pt8, Pt9, Pt10, Pt11 and, when n_rows
    is 8, the root trailing edge; y: (N, S). Branches are applied in the R code's order, so
    a later true condition overwrites an earlier one."""
    py = peri[..., 1][..., None]                    # (N, 8, 1)
    pc = peri[..., coord][..., None]
    d = np.argmax(peri[..., :7, 1], axis=-1)[:, None] + 1   # which.max, 1-based
    d = np.where(n_rows[:, None] == 8, np.argmax(peri[..., 1], axis=-1)[:, None] + 1, d)
    row = lambda i: (py[:, i - 1], pc[:, i - 1])
    LE = np.full(y.shape, np.nan)
    for i in range(1, 6):                           # segments Pt12-Pt6 ... Pt9-Pt10
        (ya, xa), (yb, xb) = row(i), row(i + 1)
        cond = (y <= yb) & (d >= i)
        if i > 1:
            cond &= y > ya
        LE = np.where(cond, _lin(xa, xb, ya, yb, y), LE)
    TE = np.full(y.shape, np.nan)
    eight = (n_rows == 8)[:, None]
    (y7, x7), (y8, x8), (y6, x6) = row(7), row(8), row(6)
    c8a = eight & (y <= y7) & (d <= 7)
    TE = np.where(c8a, _lin(x8, x7, y8, y7, y), TE)
    c8b = eight & (y <= y6) & (y > y7) & (d <= 6)
    TE = np.where(c8b, _lin(x7, x6, y7, y6, y), TE)
    c7 = ~eight & (y <= y6) & (d <= 6)
    TE = np.where(c7, _lin(x7, x6, y7, y6, y), TE)
    for i in (5, 4, 3, 2):                          # Pt10-Pt9 ... Pt7-Pt6
        (ya, xa), (yb, xb) = row(i + 1), row(i)
        cond = (y <= yb) & (y > ya) & (d <= i)
        TE = np.where(cond, _lin(xa, xb, ya, yb, y), TE)
    return LE, TE


def standard_mean_c4(w: WingShapes, chunk: int = 512) -> np.ndarray:
    """x of the quarter chord of the standard mean chord, origin at the VRP (find_c4_x with
    "standard mean", Eq. 7), for every wing shape, from 1,000 spanwise strips."""
    P = w.pts
    peri = np.stack([P["pt12"], P["pt6"], P["pt7"], P["pt8"], P["pt9"], P["pt10"], P["pt11"],
                     np.stack([P["pt11"][:, 0], P["pt12"][:, 1], P["pt12"][:, 2]], -1)], axis=1)
    n_rows = np.where(P["pt11"][:, 1] > P["pt12"][:, 1], 8, 7)
    out = np.empty(len(peri))
    for s in range(0, len(peri), chunk):
        pe, nr = peri[s:s + chunk], n_rows[s:s + chunk]
        ys = np.where(nr[:, None] == 8, pe[..., 1], np.concatenate(
            [pe[..., :7, 1], pe[..., :1, 1]], axis=1))
        y = np.linspace(ys.min(1), ys.max(1), N_STRIPS, axis=-1)
        LEx, TEx = _edges(pe, nr, y, 0)
        LEz, TEz = _edges(pe, nr, y, 2)
        chord = np.sqrt((LEx - TEx) ** 2 + (LEz - TEz) ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = np.cos(np.arctan((TEz - LEz) / (TEx - LEx)))
        c4 = np.where(TEx > LEx, LEx + 0.25 * chord * cos, LEx - 0.25 * chord * cos)
        c, x = chord[:, :-1], c4[:, :-1]            # the R loop skips the last station
        out[s:s + chunk] = np.sum(c * x, axis=1) / np.sum(c, axis=1)
    return out


def root_chord(w: WingShapes) -> np.ndarray:
    """Root chord in the x-z plane between Pt12 and Pt11 (process_outputdata.R)."""
    a, b = w.pts["pt12"], w.pts["pt11"]
    return np.sqrt((a[:, 0] - b[:, 0]) ** 2 + (a[:, 2] - b[:, 2]) ** 2)


def wing_area(w: WingShapes) -> np.ndarray:
    """Single-wing area S, with the triangle proximal to the last secondary added when the
    wing root extends beyond it (process_outputdata.R)."""
    P = w.pts
    corner = np.stack([P["pt11"][:, 0], P["pt12"][:, 1], P["pt12"][:, 2]], -1)
    A8 = 0.5 * np.linalg.norm(np.cross(corner - P["pt2"], corner - P["pt11"]), axis=-1)
    return w.S + np.where(P["pt11"][:, 1] > P["pt12"][:, 1], A8, 0.0)


@dataclass
class Stability:
    c_root_max: float
    x_np: np.ndarray        # neutral point, from the humeral head (m; negative is aft)
    x_cg: np.ndarray        # centre of gravity, from the humeral head (m)
    sm: np.ndarray          # static margin (m), positive = stable
    sm_nd: np.ndarray       # static margin / c_root_max
    S_max: float
    q_dot: np.ndarray       # pitch agility metric of Eq. (3), up to a constant


def stability(bird: Bird, w: WingShapes) -> Stability:
    """Static margin and agility for every wing shape of one specimen
    (process_outputdata.R, 'Compute manoeuvring metrics')."""
    c_root_max = float(np.max(root_chord(w)))
    x_hum = w.pts["pt1"][:, 0]
    nd = (standard_mean_c4(w) - x_hum) / c_root_max
    x_np = -np.abs(nd) ** NP_EXPONENT * c_root_max
    x_np = np.where(nd > 0, -x_np, x_np)
    x_cg = bird.CG[:, 0] - x_hum
    sm = x_cg - x_np
    S_max = float(np.max(wing_area(w)))
    q_dot = -sm * S_max * bird.m ** 0.24 / bird.I[:, 1, 1]
    return Stability(c_root_max, x_np, x_cg, sm, sm / c_root_max, S_max, q_dot)
