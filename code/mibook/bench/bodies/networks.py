"""Random mass-spring-damper networks: the standard body of morphological computation
and physical reservoir computing (Hauser et al. 2011; Nakajima 2020).

Nodes sit at random points in the unit square, joined by the edges of a Delaunay
triangulation. Each edge is a linear spring with a damper in parallel. A few nodes are
anchored to the environment by springs and dampers. Near rest the network is linear: an edge
resists only stretch along its own direction, so it adds k_e u_e u_e^T to the stiffness
between its two nodes, and the same pattern scaled by b_e to the damping.

Environment channels: an input force on chosen nodes ("force", along x), and the anchor
stiffness `k_anchor` as a property of what the network is attached to.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from .. import register
from ..linear import LinearBody


def _edges(points: np.ndarray) -> list[tuple[int, int]]:
    tri = Delaunay(points)
    E = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            i, j = sorted((int(s[a]), int(s[b])))
            E.add((i, j))
    return sorted(E)


@register("network", "1-linear",
          "Random 2D mass-spring-damper network (Delaunay edges), anchored at a few nodes.")
def network(n_nodes: int = 20, seed: int = 0, k_edge: float = 100.0, b_edge: float = 0.2,
            mass: float = 0.01, n_anchor: int = 3, k_anchor: float = 100.0,
            b_anchor: float = 0.2, n_input: int = 1, sigma_T2: float = 1e-4) -> LinearBody:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, (n_nodes, 2))
    edges = _edges(pts)
    kk = k_edge * rng.uniform(0.5, 1.5, len(edges))
    bb = b_edge * rng.uniform(0.5, 1.5, len(edges))
    n = 2 * n_nodes
    K = np.zeros((n, n))
    D = np.zeros((n, n))
    for (i, j), k, b in zip(edges, kk, bb):
        u = pts[j] - pts[i]
        u /= np.linalg.norm(u)
        a = np.zeros(n)
        a[2 * i:2 * i + 2] = -u
        a[2 * j:2 * j + 2] = u
        K += k * np.outer(a, a)
        D += b * np.outer(a, a)
    # Anchor the lowest nodes (smallest y) to the environment in both directions.
    anchors = np.argsort(pts[:, 1])[:n_anchor]
    for i in anchors:
        for c in (0, 1):
            K[2 * i + c, 2 * i + c] += k_anchor
            D[2 * i + c, 2 * i + c] += b_anchor
    H = mass * np.eye(n)
    # Input on the highest nodes, along x.
    inputs_at = np.argsort(-pts[:, 1])[:n_input]
    J = np.zeros((1, n))
    for i in inputs_at:
        J[0, 2 * i] = 1.0
    parts = {}
    for i in range(n_nodes):
        C = np.zeros((2, 2 * n))
        C[0, 2 * i] = C[1, 2 * i + 1] = 1.0
        parts[f"node{i}"] = C
    parts["anchors"] = np.vstack([parts[f"node{i}"] for i in anchors])
    params = dict(n_nodes=n_nodes, seed=seed, k_edge=k_edge, b_edge=b_edge, mass=mass,
                  n_anchor=n_anchor, k_anchor=k_anchor, b_anchor=b_anchor, n_input=n_input,
                  sigma_T2=sigma_T2)
    body = LinearBody(H=H, D=D, K=K, inputs={"force": J}, parts=parts, sigma_T2=sigma_T2,
                      name=f"network n={n_nodes} seed={seed}", params=params, build=network)
    return body


def node_distance(body: LinearBody, i: int, j: int) -> float:
    """Euclidean distance between two nodes' rest positions (re-derived from the seed)."""
    rng = np.random.default_rng(body.params["seed"])
    pts = rng.uniform(0.0, 1.0, (body.params["n_nodes"], 2))
    return float(np.linalg.norm(pts[i] - pts[j]))
