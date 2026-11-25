"""The book's running example: a leg on the ground, rigid or series-elastic."""
from __future__ import annotations

import numpy as np

from passive_info.information import SIGMA_ENERGY
from passive_info.systems import proprioceptive, series_elastic

from .. import register
from ..linear import LinearBody


def _from_passive(s, name, params, build) -> LinearBody:
    n = s.n
    D = -s.H @ s.A[n:, n:]
    JT = s.H @ s.B_F[n:, :1]
    parts = {"rotor": _sel(n, n - 1), "foot": _sel(n, 0)}
    if n > 1:
        parts["link"] = _sel(n, 0)
    motor = np.zeros((1, n))
    motor[0, n - 1] = 1.0                      # torque on the rotor coordinate
    return LinearBody(H=s.H, D=0.5 * (D + D.T), K=s.K, inputs={"force": JT.T, "motor": motor},
                      parts=parts, sigma_T2=SIGMA_ENERGY ** 2, name=name,
                      params=params, build=build)


def _sel(n: int, i: int) -> np.ndarray:
    C = np.zeros((1, 2 * n))
    C[0, i] = 1.0
    return C


@register("rigid", "1-linear", "Rigid (proprioceptive) leg on a stiff, damped ground.")
def rigid() -> LinearBody:
    return _from_passive(proprioceptive(readout="energy"), "rigid", {}, None)


@register("sea", "1-linear", "Series-elastic leg; parameter ks (N m/rad).")
def sea(ks: float = 1e4) -> LinearBody:
    return _from_passive(series_elastic(ks, readout="energy"), f"SEA {ks:g}", {"ks": ks}, sea)
