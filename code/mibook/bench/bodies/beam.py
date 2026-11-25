"""A whisker-like beam: a slender cantilever touched somewhere along its length.

Euler-Bernoulli finite elements (cubic Hermite) on a beam of length L. The base sits in a
follicle, modelled as a rotational spring k_f with damper b_f, so the base angle is a
coordinate of its own: the "follicle" part, which plays the role of the rotor in the legs.
Material damping is stiffness-proportional, D_beam = beta K_beam.

Environment channels: a transverse contact force ("force") at distance `contact` from the
base, entering through the Hermite shape functions there, so the contact point is a
continuous property whose Fisher information can be computed; and the force itself.

Default numbers are a 10 cm, 1 mm wire with a first mode near 40 Hz. The declared resolution
sigma_T2 defaults to 1e-9 J so that a 1 mN force is resolved at a moderate signal-to-noise
ratio; physical thermal noise (about 4e-21 J) would give far more bits.
"""
from __future__ import annotations

import numpy as np

from .. import register
from ..linear import LinearBody


def _hermite(xi: float, h: float) -> np.ndarray:
    """Cubic Hermite shape functions at local coordinate xi in [0, 1], element length h."""
    return np.array([1 - 3 * xi ** 2 + 2 * xi ** 3, h * (xi - 2 * xi ** 2 + xi ** 3),
                     3 * xi ** 2 - 2 * xi ** 3, h * (-xi ** 2 + xi ** 3)])


@register("whisker", "1-linear",
          "Cantilever beam in a compliant follicle, touched at a point along its length.")
def whisker(n_el: int = 12, length: float = 0.1, diameter: float = 1e-3, E: float = 5e10,
            rho: float = 6500.0, k_f: float = 0.05, b_f: float = 2e-5, beta: float = 1e-5,
            contact: float = 0.07, sigma_T2: float = 1e-9) -> LinearBody:
    h = length / n_el
    Iz = np.pi * diameter ** 4 / 64.0
    Ar = np.pi * diameter ** 2 / 4.0
    EI, rA = E * Iz, rho * Ar
    ke = EI / h ** 3 * np.array([[12, 6 * h, -12, 6 * h], [6 * h, 4 * h * h, -6 * h, 2 * h * h],
                                 [-12, -6 * h, 12, -6 * h], [6 * h, 2 * h * h, -6 * h, 4 * h * h]])
    me = rA * h / 420.0 * np.array([[156, 22 * h, 54, -13 * h], [22 * h, 4 * h * h, 13 * h, -3 * h * h],
                                    [54, 13 * h, 156, -22 * h], [-13 * h, -3 * h * h, -22 * h, 4 * h * h]])
    nd = 2 * (n_el + 1)
    Kf = np.zeros((nd, nd))
    Mf = np.zeros((nd, nd))
    for e in range(n_el):
        idx = slice(2 * e, 2 * e + 4)
        Kf[idx, idx] += ke
        Mf[idx, idx] += me
    # Base: no transverse displacement (dof 0 removed); the base angle (dof 1) is held by
    # the follicle spring.
    keep = np.arange(1, nd)
    K = Kf[np.ix_(keep, keep)].copy()
    H = Mf[np.ix_(keep, keep)].copy()
    D = beta * K.copy()
    K[0, 0] += k_f
    D[0, 0] += b_f
    # Contact force: consistent load vector of the shape functions at `contact`.
    e = min(int(contact // h), n_el - 1)
    xi = (contact - e * h) / h
    load = np.zeros(nd)
    load[2 * e:2 * e + 4] = _hermite(xi, h)
    J = load[keep][None, :]
    n = H.shape[0]

    def sel(i):
        C = np.zeros((1, 2 * n))
        C[0, i] = 1.0
        return C

    parts = {"follicle": sel(0), "tip": sel(n - 2)}       # base angle; tip displacement
    params = dict(n_el=n_el, length=length, diameter=diameter, E=E, rho=rho, k_f=k_f, b_f=b_f,
                  beta=beta, contact=contact, sigma_T2=sigma_T2)
    return LinearBody(H=H, D=D, K=K, inputs={"force": J}, parts=parts, sigma_T2=sigma_T2,
                      name=f"whisker contact={contact:g}", params=params, build=whisker)
