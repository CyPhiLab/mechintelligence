"""Random recurrent networks of nonlinear springs and masses, after Hauser, Ijspeert,
Fuchslin, Pfeifer and Maass (2011), "Towards a theoretical foundation for morphological
computation with compliant bodies", Biol. Cybern. 105: 355-370, Section 4, and the authors'
Matlab toolbox (github.com/helmuthauser/Matlab_Morphological_Computation_Simulation,
commit 1e7a17f), which this module ports.

Construction (paper Sect. 4.1.1; toolbox init_ms_sys_net.m): N nodes placed uniformly at
random in [0, 10] x [0, 10] (toolbox init_ms_sys_data.m), joined by the edges of a Delaunay
triangulation. Each edge is a nonlinear spring (paper Eq. 4) with tension
p(x1) + q(x2), p(x) = k3 x^3 + k1 x, q(x) = d3 x^3 + d1 x, x1 = l - l0 the stretch and x2
its rate. Rest lengths are the starting distances, so the network starts at rest. The
leftmost and the rightmost node are fixed. ceil(0.2 N) randomly chosen nodes receive the
input as a horizontal force w_in u, w_in uniform on [-1, 1] (zero on fixed nodes). All
masses are 1. The readout is a weighted sum of the spring lengths, fitted by least squares.

Parameter ranges. The paper's robot-arm and multitasking network (Sect. 4.2): k1, d1
log-uniform on [1, 100]; k3, d3 uniform on [100, 200] (`PAPER`). The toolbox's task
scripts use other ranges, all log-uniform (`VOLTERRA`, `NARMA`).

Time stepping (toolbox simulate_ms_sys.m): at each 1 ms step the spring forces are computed
from the current positions, with x2 the backward difference of the length over one step,
and each mass is then advanced under that force held over the step. The paper describes this
as fourth-order Runge-Kutta; with the force held, RK4 reduces to the exact constant-force
update (`scheme="held"`, the default). The toolbox's default is the forward Euler update
(`scheme="euler"`).

Tasks (toolbox data_Volterra/make_Volterra_data.m and data_NARMA/make_NARMA_data.m):
- Volterra operator, paper Eq. 2. Kernel h2 = dt exp(-[(t1 - 0.1)^2 + (t2 - 0.1)^2] /
  (2 0.05^2)) sampled at 200 points of linspace(0, 0.2, 200), summed (no dt^2). Input
  x = sin(2 pi 2.11 t) sin(2 pi 3.73 t) sin(2 pi 4.33 t); the network is driven by 0.2 x.
- NARMA system, paper Eq. 8 (Atiya and Parlos 2000) with u = 0.1 x; both input and target
  are normalized to zero mean and unit standard deviation, and the network is driven by
  the normalized input.

`hauser_network` registers the body for the sampling engine (`MechSim`): the linear damping
d1 at the rest geometry goes into the damping matrix, which also sets the thermal noise; the
rest of the spring and damper forces go into the conservative and velocity-dependent forces.
There the stretch rate is the exact rate e . (v_j - v_i), the limit of the backward
difference.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import Delaunay

from .. import register
from ..sampling import MechSim

DT = 1e-3                          # 1 ms time step
FREQS = (2.11, 3.73, 4.33)         # Hz

# Parameter ranges: (k1, k3, d1, d3) ranges and whether k3, d3 are log-uniform.
PAPER = dict(k1=(1.0, 100.0), k3=(100.0, 200.0), d1=(1.0, 100.0), d3=(100.0, 200.0),
             cubic_log=False)                                # paper Sect. 4.2
VOLTERRA = dict(k1=(1.0, 200.0), k3=(1.0, 100.0), d1=(1.0, 200.0), d3=(1.0, 100.0),
                cubic_log=True)                              # toolbox learning_Volterra.m
NARMA = dict(k1=(10.0, 100.0), k3=(1.0, 100.0), d1=(10.0, 100.0), d3=(1.0, 100.0),
             cubic_log=True)                                 # toolbox learning_NARMA.m


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
        """The same network with every spring given the mean parameters (the control for
        diversity in paper Sect. 4.2)."""
        f = lambda a: np.full_like(a, a.mean())
        return replace(self, k1=f(self.k1), k3=f(self.k3), d1=f(self.d1), d3=f(self.d3))

    def lengths(self, P: np.ndarray) -> np.ndarray:
        i, j = self.edges.T
        return np.linalg.norm(P[..., j, :] - P[..., i, :], axis=-1)

    def forces(self, P: np.ndarray, rate: np.ndarray) -> np.ndarray:
        """Spring and damper forces on the nodes (B, N, 2), for positions (B, N, 2) and
        stretch rates (B, L)."""
        i, j = self.edges.T
        d = P[:, j] - P[:, i]
        l = np.linalg.norm(d, axis=-1)
        e = d / l[..., None]
        x1 = l - self.l0
        T = self.k3 * x1 ** 3 + self.k1 * x1 + self.d3 * rate ** 3 + self.d1 * rate
        F = np.zeros_like(P)
        np.add.at(F, (slice(None), i), T[..., None] * e)
        np.add.at(F, (slice(None), j), -T[..., None] * e)
        F[:, self.fixed] = 0.0
        return F


def build(n_nodes: int = 30, seed: int = 0, ranges: dict = PAPER, area=(10.0, 10.0),
          input_fraction: float = 0.2, w_in_range=(-1.0, 1.0), rng=None) -> HauserNet:
    """Draw one network (paper Sect. 4.1; toolbox init_ms_sys_net.m)."""
    rng = np.random.default_rng(seed) if rng is None else rng
    pts = rng.uniform((0.0, 0.0), area, (n_nodes, 2))
    edges = _edges(pts)
    L = len(edges)
    logu = lambda r: 10 ** rng.uniform(np.log10(r[0]), np.log10(r[1]), L)
    cub = logu if ranges["cubic_log"] else (lambda r: rng.uniform(*r, L))
    k1, k3, d1, d3 = logu(ranges["k1"]), cub(ranges["k3"]), logu(ranges["d1"]), cub(ranges["d3"])
    fixed = np.zeros(n_nodes, bool)
    fixed[[np.argmin(pts[:, 0]), np.argmax(pts[:, 0])]] = True
    n_in = int(np.ceil(input_fraction * n_nodes))
    w_in = np.zeros(n_nodes)
    w_in[rng.permutation(n_nodes)[:n_in]] = rng.uniform(*w_in_range, n_in)
    w_in[fixed] = 0.0
    l0 = np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=-1)
    return HauserNet(pts, edges, l0, k1, k3, d1, d3, fixed, w_in)


@dataclass
class State:
    P: np.ndarray               # (B, N, 2)
    V: np.ndarray
    l_old: np.ndarray           # (B, L), lengths at the previous step


def rest(nets: list[HauserNet]) -> State:
    P = np.stack([n_.pos0 for n_ in nets])
    return State(P, np.zeros_like(P), np.stack([n_.l0 for n_ in nets]))


def simulate(nets, u: np.ndarray, state: State | None = None, dt: float = DT,
             scheme: str = "held"):
    """Run networks (one, or a list sharing a graph) under input u (n_steps,). Returns the
    spring lengths at each step, (n_steps, L) or (n_steps, B, L), and the final state."""
    single = isinstance(nets, HauserNet)
    nets = [nets] if single else list(nets)
    net = nets[0]
    st = rest(nets) if state is None else State(state.P.copy(), state.V.copy(), state.l_old.copy())
    stack = lambda a: np.stack([getattr(n_, a) for n_ in nets])[:, None, :]
    k1, k3, d1, d3 = (stack(a)[:, 0] for a in ("k1", "k3", "d1", "d3"))
    w_in = np.stack([n_.w_in for n_ in nets])
    i, j = net.edges.T
    fixed = net.fixed
    P, V, l_old = st.P, st.V, st.l_old
    out = np.empty((len(u), len(nets), net.n_springs))
    for k in range(len(u)):
        d = P[:, j] - P[:, i]
        l = np.sqrt((d ** 2).sum(-1))
        out[k] = l
        e = d / l[..., None]
        x1 = l - net.l0
        x2 = (l - l_old) / dt
        T = k3 * x1 ** 3 + k1 * x1 + d3 * x2 ** 3 + d1 * x2
        l_old = l
        F = np.zeros_like(P)
        np.add.at(F, (slice(None), i), T[..., None] * e)
        np.add.at(F, (slice(None), j), -T[..., None] * e)
        F[..., 0] += u[k] * w_in
        F[:, fixed] = 0.0
        a = F / net.mass
        if scheme == "held":
            P = P + dt * V + 0.5 * dt * dt * a
        else:
            P = P + dt * V
        V = V + dt * a
        V[:, fixed] = 0.0
    st = State(P, V, l_old)
    return (out[:, 0] if single else out), st


# ---------------------------------------------------------------------------------------
# Task data, as the toolbox generates it
# ---------------------------------------------------------------------------------------

def three_sines(t: np.ndarray) -> np.ndarray:
    f1, f2, f3 = FREQS
    return np.sin(2 * np.pi * f1 * t) * np.sin(2 * np.pi * f2 * t) * np.sin(2 * np.pi * f3 * t)


def mapstd(x: np.ndarray) -> np.ndarray:
    """Matlab's mapstd: zero mean, unit standard deviation (N - 1 normalization)."""
    return (x - x.mean()) / x.std(ddof=1)


def volterra_data(seconds: float = 500.0, dt: float = DT) -> dict:
    """make_Volterra_data.m. Returns u (the network input, 0.2 x), y and the normalized yn,
    with the 200 leading zeros the script prepends."""
    n_h = int(round(0.2 / dt))
    tau = np.linspace(0.0, 0.2, n_h)
    g = np.exp(-(tau - 0.1) ** 2 / (2 * 0.05 ** 2))          # h2 = dt * g g^T (separable)
    t = np.linspace(0.0, seconds, int(round(seconds / dt)))
    x = np.concatenate([np.zeros(n_h), three_sines(t)])
    z = np.convolve(x, g)[:len(x)]                             # z[i] = sum_m g[m] x[i - m]
    y = dt * z ** 2
    y[:n_h] = 0.0                                              # the script's loop starts at n_h + 1
    return dict(u=0.2 * x, y=y, yn=mapstd(y))


def narma_data(seconds: float = 500.0, dt: float = DT, cutoff: int = 1000) -> dict:
    """make_NARMA_data.m: paper Eq. 8 driven by u = 0.1 x; normalized input un and target yn."""
    n = int(round(seconds / dt))
    t = np.linspace(0.0, seconds, n)
    u = 0.1 * three_sines(t)
    u[:10] = 0.0
    y = np.zeros(n)
    for k in range(10, n):
        y[k] = (0.3 * y[k - 1] + 0.05 * y[k - 1] * y[k - 10:k].sum()
                + 1.5 * u[k - 10] * u[k - 1] + 0.1)
    return dict(un=mapstd(u[cutoff - 1:]), yn=mapstd(y[cutoff - 1:]))


def learn_and_test(nets, U, Y, U_test, Y_test, washout: int, normalize_output: bool,
                   scheme: str = "held") -> list[dict]:
    """learning_Volterra.m / learning_NARMA.m: simulate on the learning input, fit output
    weights to the lengths after the washout (least squares, no bias), continue from the
    final state on the test input, and report the test mean squared error. With
    `normalize_output` the output and target are both normalized first (Volterra script).
    Also returns the error of the linear model y = a u + b on the raw input
    (learn_linear_model.m), fitted on the same samples."""
    single = isinstance(nets, HauserNet)
    nets = [nets] if single else list(nets)
    L, st = simulate(nets, U, scheme=scheme)
    L_test, _ = simulate(nets, U_test, st, scheme=scheme)
    raw = np.stack([U[washout:], np.ones(len(U) - washout)], 1)
    w_lr = np.linalg.lstsq(raw, Y[washout:], rcond=None)[0]
    lr_out = np.stack([U_test, np.ones(len(U_test))], 1) @ w_lr
    res = []
    for b in range(len(nets)):
        X = L[washout:, b]
        w = np.linalg.lstsq(X, Y[washout:], rcond=None)[0]
        o = L_test[:, b] @ w
        tgt = Y_test
        if normalize_output:
            o, tgt, lr = mapstd(o), mapstd(Y_test), mapstd(lr_out)
        else:
            lr = lr_out
        res.append(dict(mse=float(np.mean((tgt - o) ** 2)), mse_lr=float(np.mean((tgt - lr) ** 2))))
    return res[0] if single else res


def volterra_task(net, data=None, start=20001, n_learn=200001, n_test=15000,
                  washout=60000, **kw):
    """The Volterra run of learning_Volterra.m (Matlab indices start:len with start = 20000,
    len = 220000, washout 60000, 15000 test steps); lengths can be shortened for tests."""
    d = volterra_data() if data is None else data
    s = start - 1
    U, Y = d["u"][s:s + n_learn], d["y"][s:s + n_learn]
    U_t, Y_t = d["u"][s + n_learn:s + n_learn + n_test], d["yn"][s + n_learn:s + n_learn + n_test]
    return learn_and_test(net, U, Y, U_t, Y_t, washout, normalize_output=True, **kw)


def narma_task(net, data=None, start=50000, n_learn=210001, n_test=15000,
               washout=60000, **kw):
    """The NARMA run of learning_NARMA.m (start = 50000, len = 260000, washout 60000,
    15000 test steps)."""
    d = narma_data() if data is None else data
    s = start - 1
    U, Y = d["un"][s:s + n_learn], d["yn"][s:s + n_learn]
    U_t, Y_t = d["un"][s + n_learn:s + n_learn + n_test], d["yn"][s + n_learn:s + n_learn + n_test]
    return learn_and_test(net, U, Y, U_t, Y_t, washout, normalize_output=False, **kw)


@register("hauser_network", "2-nonlinear",
          "Random recurrent network of nonlinear (cubic) springs and masses after Hauser "
          "et al. 2011; input: horizontal force on 20% of nodes; readout: spring lengths.")
def hauser_network(n_nodes: int = 30, seed: int = 0, sigma_T2: float = 1e-4,
                   **kw) -> MechSim:
    net = build(n_nodes, seed, **kw)
    free = np.flatnonzero(~net.fixed)
    nf = len(free)
    idx = {int(n_): r for r, n_ in enumerate(free)}
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
    ei, ej = net.edges.T

    def full(q, v=None):
        P = np.repeat(net.pos0[None], q.shape[0], axis=0)
        P[:, free] = q.reshape(-1, nf, 2)
        V = np.zeros_like(P)
        if v is not None:
            V[:, free] = v.reshape(-1, nf, 2)
        return P, V

    def rate(P, V):
        d = P[:, ej] - P[:, ei]
        e = d / np.linalg.norm(d, axis=-1)[..., None]
        return np.einsum("bld,bld->bl", e, V[:, ej] - V[:, ei])

    def force(q, p):
        P, _ = full(q)
        return net.forces(P, np.zeros((q.shape[0], net.n_springs)))[:, free].reshape(q.shape[0], -1)

    def velocity_force(q, v, p):
        P, V = full(q, v)
        F = net.forces(P, rate(P, V))[:, free].reshape(q.shape[0], -1)
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
