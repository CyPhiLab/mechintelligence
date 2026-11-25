"""Item 3 of the 2026-11-25 priorities: holding, acting and paying on one body (thread T12).

The question is open: across stiffness and frequency, where does compliance help or hurt
perception, action and cost? Nothing here assumes a trade-off. Every quantity is a
frequency response of a linear body, so each is computed on a grid of designs and
frequencies.

Leg (clamped, foot on a ground of stiffness k_c and damping b_c):
- perception: information density (bits per second per hertz) that the rotor's motion
  holds about the ground force, against the body's own thermal noise, for a force of
  two-sided spectral density `level`;
- action: ground force per unit motor torque, |F_c / tau|;
- cost: motor heat per unit squared ground force, proportional to |tau / F_c|^2 for a
  motor whose losses grow with the square of its torque (the inverse square of action);
- work: mechanical power the motor delivers per unit squared ground force,
  (1/2) Re(tau conj(theta_m')) / |F_c|^2 (all of it is dissipated in the body and ground);
- shock: rotor acceleration per unit ground force, |omega^2 theta_m / f| (what a hard
  contact does to the gearbox).

Bird (shoulder stiffness k_s):
- perception: information density that the shoulder's deflection holds about the gust;
- withstanding: body acceleration per unit gust velocity (lower is better);
- action: body acceleration per unit shoulder actuator force (a stroke between wing and
  body).
"""
from __future__ import annotations

import numpy as np

from . import make

K_C, B_C = 1e6, 600.0          # the book's ground, N/m and N s/m (passive_info K_I, B_CONTACT)


def _Z(body, w):
    return body.K - w ** 2 * body.H + 1j * w * body.D


def leg_maps(ks_grid=None, f_grid=None, level: float = 1.0) -> dict:
    ks_grid = np.logspace(1.5, 6.0, 37) if ks_grid is None else np.asarray(ks_grid)
    f_grid = np.logspace(0.0, 3.0, 61) if f_grid is None else np.asarray(f_grid)
    designs = [("rigid", make("rigid"))] + [(ks, make("sea", ks=ks)) for ks in ks_grid]
    out = {k: np.empty((len(designs), len(f_grid)))
           for k in ("percep", "action", "cost", "work", "shock")}
    for i, (_, b) in enumerate(designs):
        n = b.n
        J = np.atleast_2d(b.inputs["force"])[0]          # generalized force per unit ground force
        e_m = np.atleast_2d(b.inputs["motor"])[0]
        Jfoot = np.zeros(n)
        Jfoot[0] = -0.1                                  # foot displacement y = J q (r = 0.1 m)
        for j, f in enumerate(f_grid):
            w = 2 * np.pi * f
            Z = _Z(b, w)
            # Rotor response to the ground force, and the thermal noise at the rotor.
            x_f = np.linalg.solve(Z, J.astype(complex))
            Y = np.linalg.inv(Z)
            Sn = 2 * b.sigma_T2 * np.real(Y[n - 1] @ b.D @ Y[n - 1].conj())
            snr = level * abs(x_f[n - 1]) ** 2 / Sn
            out["percep"][i, j] = np.log2(1 + snr)
            # Ground force produced by a motor torque.
            x_m = np.linalg.solve(Z, e_m.astype(complex))
            Fc = (K_C + 1j * w * B_C) * (Jfoot @ x_m)
            out["action"][i, j] = abs(Fc)
            out["cost"][i, j] = 1.0 / abs(Fc) ** 2
            v_m = 1j * w * x_m[n - 1]
            out["work"][i, j] = 0.5 * np.real(np.conj(v_m)) / abs(Fc) ** 2
            out["shock"][i, j] = w ** 2 * abs(x_f[n - 1])
    return dict(f=f_grid, ks=ks_grid, labels=[d[0] for d in designs], **out)


def bird_maps(ks_grid=None, f_grid=None, level: float = 1.0) -> dict:
    ks_grid = np.logspace(0.0, 4.0, 33) if ks_grid is None else np.asarray(ks_grid)
    f_grid = np.logspace(-1.0, 2.0, 61) if f_grid is None else np.asarray(f_grid)
    out = {k: np.empty((len(ks_grid), len(f_grid))) for k in ("percep", "gust", "action")}
    for i, ks in enumerate(ks_grid):
        b = make("bird", k_s=ks)
        A = b.A
        d = A.shape[0]
        Bg = b.B("gust")
        m_b, m_w = b.params["m_b"], b.params["m_w"]
        Bu = np.zeros((d, 1))
        Bu[1, 0], Bu[2, 0] = -1.0 / m_b, 1.0 / m_w        # stroke: + on wings, - on body
        GG = b.GG
        a_row = b._accel_row
        for j, f in enumerate(f_grid):
            w = 2 * np.pi * f
            R = np.linalg.inv(1j * w * np.eye(d) - A)
            xg = R @ Bg[:, 0]
            xu = R @ Bu[:, 0]
            Sn = np.real(R[0] @ GG @ R[0].conj())
            out["percep"][i, j] = np.log2(1 + level * abs(xg[0]) ** 2 / Sn)
            out["gust"][i, j] = abs(a_row @ xg)
            out["action"][i, j] = abs(a_row @ xu)
    return dict(f=f_grid, ks=ks_grid, **out)


def helps(maps: dict, key: str, better: str = "higher", ref: int = -1,
          margin: float = 0.1) -> np.ndarray:
    """Where a design beats the stiffest one (row `ref`) by more than `margin` at each
    frequency. For the leg the stiffest design is the rigid leg (row 0); for the bird the
    last row."""
    X = maps[key]
    r = X[ref]
    return (X > r * (1 + margin)) if better == "higher" else (X < r * (1 - margin))


def hurts(maps: dict, key: str, better: str = "higher", ref: int = -1,
          margin: float = 0.1) -> np.ndarray:
    return helps(maps, key, "lower" if better == "higher" else "higher", ref, margin)


def summary() -> dict:
    leg = leg_maps()
    bird = bird_maps()
    L = {}
    for key, better in LEG_SENSE.items():
        L[key] = (float(helps(leg, key, better, ref=0)[1:].mean()),
                  float(hurts(leg, key, better, ref=0)[1:].mean()))
    B = {}
    for key, better in BIRD_SENSE.items():
        B[key] = (float(helps(bird, key, better, ref=-1)[:-1].mean()),
                  float(hurts(bird, key, better, ref=-1)[:-1].mean()))
    return dict(leg=leg, bird=bird, leg_helped_hurt=L, bird_helped_hurt=B)


LEG_SENSE = {"percep": "higher", "action": "higher", "cost": "lower", "work": "lower",
             "shock": "lower"}
BIRD_SENSE = {"percep": "higher", "gust": "lower", "action": "higher"}


def sea_resonance_hz(ks: float, i_rot: float = 0.01) -> float:
    """The rotor on the series spring with the link held: sqrt(ks / I_rot) / 2 pi."""
    return float(np.sqrt(ks / i_rot) / (2 * np.pi))
