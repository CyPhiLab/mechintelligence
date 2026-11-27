"""Random recurrent networks of nonlinear springs and masses, after Hauser, Ijspeert,
Fuchslin, Pfeifer and Maass (2011), "Towards a theoretical foundation for morphological
computation with compliant bodies", Biol. Cybern. 105: 355-370, Section 4.

Construction (their Sect. 4.1.1): N nodes placed uniformly at random in a rectangle of the
plane, joined by the edges of a Delaunay triangulation. Each edge is a nonlinear spring
(their Eq. 4) with tension p(x1) + q(x2), p(x) = k3 x^3 + k1 x, q(x) = d3 x^3 + d1 x, where
x1 = l - l0 is the stretch and x2 its rate. Rest lengths are the distances at the start, so
the network starts at rest. The leftmost and the rightmost node are fixed. A fraction of the
nodes receive the input as a horizontal force w_in u, with w_in drawn once. All masses are 1
(their Sect. 4.1.2). Spring parameters for the robot-arm and multitasking networks: k1, d1
log-uniform on [1, 100], k3, d3 uniform on [100, 200]; 20% input nodes; w_in uniform on
[-1, 1] (their Sect. 4.2). The readout is a weighted sum of the spring lengths, fitted by
the pseudoinverse (their Sect. 4.1.3).

Simulation: fourth-order Runge-Kutta at 1 ms, as in the paper. One difference: the paper
estimates x2 by a backward difference of x1 over one step; here x2 is the exact rate
e . (v_j - v_i). The two agree as the step goes to zero.

Not given in the main text, and assumed here until the paper's supplementary material is
read: the size of the rectangle the nodes are placed in (`width`, `height`), and the
scaling of the input signal (`input_gain`). They set how far into the cubic range the
springs are driven, so the task errors depend on them.

The tasks of Sect. 4.3 are in `hauser_tasks`: the Volterra operator of their Eq. 2 and the
two discrete systems of their Eqs. 7 and 8 (from Atiya and Parlos 2000), driven by the
product of three sines of Sect. 3.

`hauser_network` registers the body for the sampling engine (`MechSim`): the linear damping
d1 at the rest geometry goes into the damping matrix, which also sets the thermal noise; the
rest of the spring and damper forces go into the conservative and velocity-dependent forces.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay

from .. import register
from ..sampling import MechSim

DT = 1e-3                          # Hauser et al.: 1 ms time step
FREQS = (2.11, 3.73, 4.33)         # Hz, the input of their Sect. 3


def _edges(points: np.ndarray) -> np.ndarray:
    tri = Delaunay(points)
    E = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            i, j = sorted((int(s[a]), int(s[b])))
            E.add((i, j))
    return np.array(sorted(E))


@dataclass
class HauserNet:
    """One network. Positions of free nodes are the state; fixed nodes stay put."""
    pos0: np.ndarray            # (N, 2) initial node positions
    edges: np.ndarray           # (L, 2) node indices
    l0: np.ndarray              # (L,) rest lengths
    k1: np.ndarray
    k3: np.ndarray
    d1: np.ndarray
    d3: np.ndarray
    fixed: np.ndarray           # (N,) bool
    w_in: np.ndarray            # (N,) input weight, 0 for non-input nodes
    mass: float = 1.0

    @property
    def n_nodes(self) -> int:
        return len(self.pos0)

    @property
    def n_springs(self) -> int:
        return len(self.edges)

    def homogeneous(self) -> "HauserNet":
        """The same network with every spring given the mean parameters (their Sect. 4.2
        control for diversity)."""
        f = lambda a: np.full_like(a, a.mean())
        return HauserNet(self.pos0, self.edges, self.l0, f(self.k1), f(self.k3), f(self.d1),
                         f(self.d3), self.fixed, self.w_in, self.mass)

    def lengths(self, P: np.ndarray) -> np.ndarray:
        """Spring lengths for positions P (..., N, 2)."""
        i, j = self.edges.T
        return np.linalg.norm(P[..., j, :] - P[..., i, :], axis=-1)

    def accel(self, P: np.ndarray, V: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Accelerations (B, N, 2) for positions and velocities (B, N, 2) and input (B,)."""
        i, j = self.edges.T
        d = P[:, j] - P[:, i]
        l = np.linalg.norm(d, axis=-1)
        e = d / l[..., None]
        x1 = l - self.l0
        x2 = np.einsum("bld,bld->bl", e, V[:, j] - V[:, i])
        T = self.k3 * x1 ** 3 + self.k1 * x1 + self.d3 * x2 ** 3 + self.d1 * x2
        F = np.zeros_like(P)
        np.add.at(F, (slice(None), i), T[..., None] * e)
        np.add.at(F, (slice(None), j), -T[..., None] * e)
        F[..., 0] += u[:, None] * self.w_in[None, :]
        F[:, self.fixed] = 0.0
        return F / self.mass


def build(n_nodes: int = 30, seed: int = 0, width: float = 1.0, height: float = 1.0,
          k1_range=(1.0, 100.0), d1_range=(1.0, 100.0), k3_range=(100.0, 200.0),
          d3_range=(100.0, 200.0), input_fraction: float = 0.2, w_in_range=(-1.0, 1.0),
          rng=None) -> HauserNet:
    """Draw one network with the construction of Hauser et al. (2011), Sect. 4.1-4.2.
    k1 and d1 are log-uniform, k3 and d3 uniform. `width` and `height` are assumed."""
    rng = np.random.default_rng(seed) if rng is None else rng
    pts = rng.uniform((0.0, 0.0), (width, height), (n_nodes, 2))
    edges = _edges(pts)
    L = len(edges)
    logu = lambda r: np.exp(rng.uniform(np.log(r[0]), np.log(r[1]), L))
    k1, d1 = logu(k1_range), logu(d1_range)
    k3, d3 = rng.uniform(*k3_range, L), rng.uniform(*d3_range, L)
    fixed = np.zeros(n_nodes, bool)
    fixed[[np.argmin(pts[:, 0]), np.argmax(pts[:, 0])]] = True
    n_in = max(1, int(round(input_fraction * n_nodes)))
    w_in = np.zeros(n_nodes)
    w_in[rng.choice(n_nodes, n_in, replace=False)] = rng.uniform(*w_in_range, n_in)
    l0 = np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=-1)
    return HauserNet(pts, edges, l0, k1, k3, d1, d3, fixed, w_in)


def simulate(nets: list[HauserNet] | HauserNet, u: np.ndarray, dt: float = DT) -> np.ndarray:
    """Spring lengths (n_steps, L) under input u (n_steps,), RK4 from rest, as in the paper.
    A list of networks with the same graph (e.g. a network and its homogeneous copy) is
    run as one batch and returns (n_steps, B, L)."""
    single = isinstance(nets, HauserNet)
    nets = [nets] if single else list(nets)
    net = nets[0]
    B = len(nets)
    stack = lambda name: np.stack([getattr(n_, name) for n_ in nets])
    k1, k3, d1, d3 = (stack(a)[:, None, :] for a in ("k1", "k3", "d1", "d3"))
    w_in = stack("w_in")
    i, j = net.edges.T
    fixed = net.fixed

    def acc(P, V, uk):
        d = P[:, j] - P[:, i]
        l = np.linalg.norm(d, axis=-1)
        e = d / l[..., None]
        x1 = l - net.l0
        x2 = np.einsum("bld,bld->bl", e, V[:, j] - V[:, i])
        T = (k3[:, 0] * x1 ** 3 + k1[:, 0] * x1 + d3[:, 0] * x2 ** 3 + d1[:, 0] * x2)
        F = np.zeros_like(P)
        np.add.at(F, (slice(None), i), T[..., None] * e)
        np.add.at(F, (slice(None), j), -T[..., None] * e)
        F[..., 0] += uk * w_in
        F[:, fixed] = 0.0
        return F / net.mass

    P = np.repeat(net.pos0[None], B, axis=0)
    V = np.zeros_like(P)
    out = np.empty((len(u), B, net.n_springs))
    for k in range(len(u)):
        out[k] = np.linalg.norm(P[:, j] - P[:, i], axis=-1)
        u0 = u[k]
        u1 = u[k + 1] if k + 1 < len(u) else u[k]
        um = 0.5 * (u0 + u1)
        a1 = acc(P, V, u0)
        a2 = acc(P + 0.5 * dt * V, V + 0.5 * dt * a1, um)
        a3 = acc(P + 0.5 * dt * (V + 0.5 * dt * a1), V + 0.5 * dt * a2, um)
        a4 = acc(P + dt * (V + 0.5 * dt * a2), V + dt * a3, u1)
        P = P + dt * V + dt * dt / 6.0 * (a1 + a2 + a3)
        V = V + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
    return out[:, 0] if single else out


# ---------------------------------------------------------------------------------------
# Tasks of Hauser et al. (2011)
# ---------------------------------------------------------------------------------------

def input_signal(t: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """u(t) = sin(2 pi f1 t) sin(2 pi f2 t) sin(2 pi f3 t), f = 2.11, 3.73, 4.33 Hz (Sect. 3).
    `gain` is assumed (see the module docstring)."""
    f1, f2, f3 = FREQS
    return gain * np.sin(2 * np.pi * f1 * t) * np.sin(2 * np.pi * f2 * t) * np.sin(2 * np.pi * f3 * t)


def volterra(u: np.ndarray, dt: float = DT, mu: float = 0.1, sigma: float = 0.05,
             span: float = 0.2) -> np.ndarray:
    """Their Eq. 2: y(t) = int int h2(t1, t2) u(t - t1) u(t - t2) dt1 dt2 with the Gaussian
    kernel h2 = exp(-[(t1 - mu)^2 + (t2 - mu)^2] / (2 sigma^2)) on [0, 0.2] s. The kernel is
    separable, so y = (int g(t1) u(t - t1) dt1)^2. The paper prints the exponent without its
    minus sign; the kernel in their Fig. 2 is a bump, so the minus is meant. Whether the
    discretized sum carries dt^2 is not stated; the integral form is used here."""
    tau = np.arange(int(round(span / dt)) + 1) * dt
    g = np.exp(-(tau - mu) ** 2 / (2 * sigma ** 2))
    z = np.convolve(u, g)[:len(u)] * dt
    return z ** 2


def second_order(u: np.ndarray) -> np.ndarray:
    """Their Eq. 7: y[k+1] = 0.4 y[k] + 0.4 y[k] y[k-1] + 0.6 u[k]^3 + 0.1."""
    y = np.zeros(len(u))
    for k in range(1, len(u) - 1):
        y[k + 1] = 0.4 * y[k] + 0.4 * y[k] * y[k - 1] + 0.6 * u[k] ** 3 + 0.1
    return y


def tenth_order(u: np.ndarray) -> np.ndarray:
    """Their Eq. 8: y[k+1] = 0.3 y[k] + 0.05 y[k] sum_{i=0}^{9} y[k-i] + 1.5 u[k-9] u[k] + 0.1."""
    y = np.zeros(len(u))
    for k in range(9, len(u) - 1):
        y[k + 1] = 0.3 * y[k] + 0.05 * y[k] * y[k - 9:k + 1].sum() + 1.5 * u[k - 9] * u[k] + 0.1
    return y


TASKS = dict(volterra=volterra, second_order=second_order, tenth_order=tenth_order)


def fit_readout(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Output weights w* = X^+ Y (their Sect. 4.1.3)."""
    return np.linalg.pinv(X) @ Y


def multitask(net: HauserNet, input_gain: float = 1.0, washout: float = 50.0,
              learn: float = 45.0, test: float = 5.0, dt: float = DT, lengths=None,
              tasks=tuple(TASKS)) -> dict:
    """Sect. 4.3: one network, one input, three linear readouts. Returns the test mean
    squared error of the network readout and of linear regression on the raw input
    (y = w1 u + w2), for each task. With the input unscaled, the systems of Eqs. 7 and 8
    diverge (a slow input near 1 leaves Eq. 7 without a fixed point), so the input scale of
    the paper is needed before they can be run; see the module docstring."""
    n = int(round((washout + learn + test) / dt))
    t = np.arange(n) * dt
    u = input_signal(t, input_gain)
    if lengths is None:
        lengths = simulate(net, u, dt)
    a, b = int(round(washout / dt)), int(round((washout + learn) / dt))
    raw = np.stack([u, np.ones(n)], 1)
    out = {}
    for name in tasks:
        f = TASKS[name]
        y = f(u, dt) if name == "volterra" else f(u)
        w = fit_readout(lengths[a:b], y[a:b])
        w_lr = fit_readout(raw[a:b], y[a:b])
        out[name] = dict(mse=float(np.mean((lengths[b:] @ w - y[b:]) ** 2)),
                         mse_lr=float(np.mean((raw[b:] @ w_lr - y[b:]) ** 2)),
                         var=float(np.var(y[b:])))
    return out


@register("hauser_network", "2-nonlinear",
          "Random recurrent network of nonlinear (cubic) springs and masses after Hauser "
          "et al. 2011; input: horizontal force on 20% of nodes; readout: spring lengths.")
def hauser_network(n_nodes: int = 30, seed: int = 0, sigma_T2: float = 1e-4,
                   **kw) -> MechSim:
    net = build(n_nodes, seed, **kw)
    free = np.flatnonzero(~net.fixed)
    nf = len(free)
    idx = {int(n_): r for r, n_ in enumerate(free)}
    # Linear damping d1 at the rest geometry: thermal noise and the OU part of the step.
    D = np.zeros((2 * nf, 2 * nf))
    for (i, j), d1 in zip(net.edges, net.d1):
        e = net.pos0[j] - net.pos0[i]
        e /= np.linalg.norm(e)
        a = np.zeros(2 * nf)
        if i in idx:
            a[2 * idx[i]:2 * idx[i] + 2] = -e
        if j in idx:
            a[2 * idx[j]:2 * idx[j] + 2] = e
        D += d1 * np.outer(a, a)
    H = net.mass * np.eye(2 * nf)

    def full(q):
        P = np.repeat(net.pos0[None], q.shape[0], axis=0)
        P[:, free] = q.reshape(-1, nf, 2)
        return P

    def force(q, p):
        P = full(q)
        return net.accel(P, np.zeros_like(P), np.zeros(q.shape[0]))[:, free].reshape(
            q.shape[0], -1) * net.mass

    def velocity_force(q, v, p):
        P = full(q)
        V = np.zeros_like(P)
        V[:, free] = v.reshape(-1, nf, 2)
        F = net.accel(P, V, np.zeros(q.shape[0]))[:, free].reshape(q.shape[0], -1) * net.mass
        return F - force(q, p) + v @ D.T          # the d1 part at rest is in D already

    Jrow = np.zeros((1, 2 * nf))
    for n_ in free:
        Jrow[0, 2 * idx[int(n_)]] = net.w_in[n_]
    sim = MechSim(H=H, D=D, force=force,
                  J=lambda q: np.broadcast_to(Jrow, (q.shape[0],) + Jrow.shape),
                  sigma_T2=sigma_T2, params=dict(n_nodes=n_nodes, seed=seed, **kw),
                  name=f"hauser network N={n_nodes}", q0=net.pos0[free].ravel(),
                  velocity_force=velocity_force)
    sim.net = net
    return sim
