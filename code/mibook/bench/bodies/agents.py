"""Tethered coupled agents: a linear network of agents, each tethered to its own set point
and coupled to its neighbours by springs and dampers, sensing a common environment.

This is not a swarm: it has none of the collective dynamics of swarms (no self-propulsion,
no aggregation, no synchronization). For a swarm see `swarmalators`.


Agents move in the plane. Each is tethered to its own set point (spring k_0, drag b_0) and
coupled to the agents within range r_c by springs k_c with dampers b_c. The coupling can be
read as physical links or as the virtual springs of a consensus controller; the mathematics
is the same. Environment channels: a force common to every agent ("common", along x) and a
force on agent 0 alone ("local0"). The question it is built for: what does one member hold
about the common environment, against the whole network?
"""
from __future__ import annotations

import numpy as np

from .. import register
from ..linear import LinearBody


@register("agents", "5-collective",
          "Agents in the plane, tethered and coupled to neighbours within range; "
          "inputs: a common force and a force on one agent.")
def agents(n_agents: int = 12, seed: int = 0, r_c: float = 0.45, k_0: float = 1.0,
          b_0: float = 0.5, k_c: float = 5.0, b_c: float = 0.5, mass: float = 1.0,
          sigma_T2: float = 1e-4) -> LinearBody:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, (n_agents, 2))
    n = 2 * n_agents
    Lap = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if np.linalg.norm(pts[i] - pts[j]) < r_c:
                Lap[i, i] += 1
                Lap[j, j] += 1
                Lap[i, j] -= 1
                Lap[j, i] -= 1
    I2 = np.eye(2)
    K = k_0 * np.eye(n) + k_c * np.kron(Lap, I2)
    D = b_0 * np.eye(n) + b_c * np.kron(Lap, I2)
    H = mass * np.eye(n)
    common = np.zeros((1, n))
    common[0, 0::2] = 1.0
    local0 = np.zeros((1, n))
    local0[0, 0] = 1.0
    parts = {}
    for i in range(n_agents):
        C = np.zeros((2, 2 * n))
        C[0, 2 * i] = C[1, 2 * i + 1] = 1.0
        parts[f"agent{i}"] = C
    params = dict(n_agents=n_agents, seed=seed, r_c=r_c, k_0=k_0, b_0=b_0, k_c=k_c, b_c=b_c,
                  mass=mass, sigma_T2=sigma_T2)
    body = LinearBody(H=H, D=D, K=K, inputs={"common": common, "local0": local0},
                      parts=parts, sigma_T2=sigma_T2, name=f"agents n={n_agents}",
                      params=params, build=agents)
    object.__setattr__(body, "degrees", np.diag(Lap).copy())
    return body
