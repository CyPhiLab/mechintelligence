"""An inverted pendulum held up by a torsional spring: the simplest body that can be
stable or unstable, chosen by one parameter (thread T12, instability).

Small angles: m l^2 theta'' + b theta' + (k - m g l) theta = tau + l f. The net stiffness
k - m g l plays the role of a bird's static margin: positive, the pendulum returns upright;
negative, it falls over. Inputs: a horizontal push f on the mass (the environment) and a
motor torque tau at the pivot (the actuator). With k < m g l there is no Gibbs law, because
the energy metric is not positive definite.
"""
from __future__ import annotations

import numpy as np

from .. import register
from ..linear import LinearBody

G = 9.81


@register("pendulum", "1-linear",
          "Inverted pendulum on a torsional spring; stable or unstable by the spring k.")
def pendulum(k: float = 15.0, m: float = 1.0, l: float = 1.0, b: float = 0.5,
             sigma_T2: float = 1e-4) -> LinearBody:
    H = np.array([[m * l * l]])
    K = np.array([[k - m * G * l]])
    D = np.array([[b]])
    params = dict(k=k, m=m, l=l, b=b, sigma_T2=sigma_T2)
    return LinearBody(H=H, D=D, K=K, inputs={"force": np.array([[l]]), "motor": np.array([[1.0]])},
                      parts={"angle": np.array([[1.0, 0.0]])}, sigma_T2=sigma_T2,
                      name=f"pendulum k={k:g}", params=params, build=pendulum)
