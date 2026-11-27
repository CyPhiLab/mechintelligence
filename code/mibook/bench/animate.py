"""Short video demos of every benchmark, for sanity checks (thread T12).

    uv run python -m mibook.bench.animate OUTDIR [name ...]

Each demo runs a real simulation of the benchmark with a simple environment input and draws
it. Where the motion is too small to see, displacements are magnified for display only, and
the factor is printed on the frame. Slow motion is printed too. Videos are H.264 MP4.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402

from . import make  # noqa: E402
from . import linear as L  # noqa: E402
from .sampling import gibbs_start  # noqa: E402

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FPS, N_FRAMES = 30, 150


def _write(fig, update, path, n_frames=N_FRAMES):
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(str(path), writer=FFMpegWriter(
        fps=FPS, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]), dpi=110)
    plt.close(fig)
    return path


def _label(ax, text, loc="upper left"):
    x, ha = (0.02, "left") if "left" in loc else (0.98, "right")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8, color=INK2)


def _mag(target, peak):
    """Display magnification: never below 1 (displays never shrink motion)."""
    return max(1.0, target / (peak + 1e-30))


def _fmt(A):
    return f"x{A:.2g}" if A < 100 else f"x{A:.0f}"


def _smooth_noise(rng, n, width, scale=1.0):
    k = np.exp(-0.5 * (np.arange(-3 * width, 3 * width + 1) / width) ** 2)
    x = np.convolve(rng.standard_normal(n + len(k)), k / np.sqrt((k ** 2).sum()), mode="same")
    return scale * x[:n]


def _trace_panel(ax, t, series, labels, colors):
    lines = []
    for y, lab, c in zip(series, labels, colors):
        y = np.asarray(y)
        yn = y / (np.max(np.abs(y)) + 1e-30)
        ax.plot(t, yn, color=c, lw=1.0, label=lab)
    cursor = ax.axvline(t[0], color=INK, lw=0.8)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.tick_params(labelsize=7)
    return cursor


# ---------------------------------------------------------------------------------------
# Linear bodies
# ---------------------------------------------------------------------------------------

def legs(path):
    """Rigid leg and SEA (k_s = 300) side by side, under the same random ground force."""
    rng = np.random.default_rng(0)
    fs, T = 4000.0, 0.3
    n = int(T * fs)
    f = _smooth_noise(rng, n, 25, 100.0) + 150.0 * (np.arange(n) > n // 3)
    t = np.arange(n) / fs
    bodies = [("rigid", make("rigid")), ("SEA $k_s=300$", make("sea", ks=300.0))]
    runs = []
    for _, b in bodies:
        out = L.simulate_linear(b, n, 1 / fs, 1, u=f[:, None, None], channel="force",
                                parts=("whole",), rng=rng, stationary=False)["whole"][:, 0]
        runs.append(out)
    A = _mag(0.6, max(np.abs(r[:, :b.n]).max() for r, (_, b) in zip(runs, bodies)))
    fig = plt.figure(figsize=(7.2, 4.0))
    axs = [fig.add_axes([0.03 + 0.33 * i, 0.32, 0.3, 0.62]) for i in range(2)]
    tr = fig.add_axes([0.70, 0.32, 0.28, 0.62])
    cursor = _trace_panel(tr, t, [f, runs[0][:, 0], runs[1][:, 1]],
                          ["ground force", "rigid rotor", "SEA rotor"], [INK2, C1, C2])
    art = []
    for ax, (name, b) in zip(axs, bodies):
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.6, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name, fontsize=9)
        ax.axhline(-1.35, color=INK2, lw=1)
        leg, = ax.plot([], [], color=INK, lw=3)
        rotor = plt.Circle((0.0, 0.45), 0.3, fill=False, color=C2, lw=1.5)
        ax.add_patch(rotor)
        mark, = ax.plot([], [], color=C2, lw=2)
        arrow, = ax.plot([], [], color=C1, lw=2)
        art.append((leg, mark, arrow))
    _label(fig.axes[0], f"angles {_fmt(A)}, slow motion x{N_FRAMES / FPS / T:.0f}", "lower left")

    def update(k):
        i = int(k * (n - 1) / (N_FRAMES - 1))
        for (leg, mark, arrow), r, (_, b) in zip(art, runs, bodies):
            th_j = A * r[i, 0]
            th_m = A * r[i, b.n - 1]
            foot = (1.2 * np.sin(th_j), -1.2 * np.cos(th_j))
            leg.set_data([0, foot[0]], [0, foot[1]])
            mark.set_data([0, 0.3 * np.sin(th_m)], [0.45, 0.45 + 0.3 * np.cos(th_m)])
            arrow.set_data([foot[0], foot[0]], [foot[1] - 0.15, foot[1] - 0.15 - 0.3 * f[i] / 250])
        cursor.set_xdata([t[i], t[i]])
    return _write(fig, update, path)


def network(path):
    from scipy.spatial import Delaunay  # noqa: F401
    from .bodies.networks import _edges
    b = make("network")
    p = b.params
    pts = np.random.default_rng(p["seed"]).uniform(0, 1, (p["n_nodes"], 2))
    edges = _edges(pts)
    rng = np.random.default_rng(1)
    fs, T = 200.0, 6.0
    n = int(T * fs)
    t = np.arange(n) / fs
    f = 0.8 * np.sin(2 * np.pi * 1.5 * t) + _smooth_noise(rng, n, 8, 0.4)
    x = L.simulate_linear(b, n, 1 / fs, 1, u=f[:, None, None], channel="force",
                          parts=("whole",), rng=rng)["whole"][:, 0, :b.n].reshape(n, -1, 2)
    A = _mag(0.08, np.abs(x).max())
    anchors = np.argsort(pts[:, 1])[:p["n_anchor"]]
    top = np.argsort(-pts[:, 1])[:p["n_input"]]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    lines = [ax.plot([], [], color=INK2, lw=0.8)[0] for _ in edges]
    nodes = ax.scatter(pts[:, 0], pts[:, 1], s=25, color=C1, zorder=3)
    ax.scatter(pts[anchors, 0], pts[anchors, 1] - 0.04, marker="^", s=40, color=INK2)
    arrow, = ax.plot([], [], color=C2, lw=2)
    _label(ax, f"random spring network: force on the top node; displacements {_fmt(A)}")

    def update(k):
        i = int(k * (n - 1) / (N_FRAMES - 1))
        P = pts + A * x[i]
        for ln, (a, c) in zip(lines, edges):
            ln.set_data([P[a, 0], P[c, 0]], [P[a, 1], P[c, 1]])
        nodes.set_offsets(P)
        j = top[0]
        arrow.set_data([P[j, 0], P[j, 0] + 0.15 * f[i]], [P[j, 1], P[j, 1]])
    return _write(fig, update, path)


def whisker(path):
    b = make("whisker")
    p = b.params
    rng = np.random.default_rng(2)
    fs, T = 4000.0, 0.25
    n = int(T * fs)
    t = np.arange(n) / fs
    f = np.zeros(n)
    for t0 in (0.02, 0.11, 0.17):                     # three taps
        m = (t > t0) & (t < t0 + 0.004)
        f[m] = 5e-3 * np.sin(np.pi * (t[m] - t0) / 0.004)
    q = L.simulate_linear(b, n, 1 / fs, 1, u=f[:, None, None], channel="force",
                          parts=("whole",), rng=rng)["whole"][:, 0, :b.n]
    xs = np.linspace(0, p["length"], p["n_el"] + 1)
    w = np.hstack([np.zeros((n, 1)), q[:, 1::2]])     # transverse displacement at each node
    A = _mag(0.02, np.abs(w).max())
    fig, (ax, tr) = plt.subplots(2, 1, figsize=(6.4, 4.2), gridspec_kw=dict(height_ratios=[2, 1]))
    ax.set_xlim(-0.01, 0.11)
    ax.set_ylim(-0.03, 0.03)
    ax.axis("off")
    beam, = ax.plot([], [], color=INK, lw=2)
    ax.plot([-0.005, 0], [0, 0], color=C2, lw=6)
    contact, = ax.plot([], [], color=C1, lw=2)
    _label(ax, f"whisker tapped at 7 cm; deflection {_fmt(A)}, slow motion x{N_FRAMES / FPS / T:.0f}")
    cursor = _trace_panel(tr, t, [f, q[:, 0]], ["contact force", "follicle angle"], [C1, C2])

    def update(k):
        i = int(k * (n - 1) / (N_FRAMES - 1))
        beam.set_data(xs, A * w[i])
        j = np.argmin(abs(xs - p["contact"]))
        y = A * w[i, j]
        contact.set_data([p["contact"], p["contact"]], [y + 0.004 + 3 * f[i], y + 0.004])
        cursor.set_xdata([t[i], t[i]])
    return _write(fig, update, path)


def agents(path):
    b = make("agents")
    p = b.params
    pts = np.random.default_rng(p["seed"]).uniform(0, 1, (p["n_agents"], 2))
    links = [(i, j) for i in range(len(pts)) for j in range(i + 1, len(pts))
             if np.linalg.norm(pts[i] - pts[j]) < p["r_c"]]
    rng = np.random.default_rng(3)
    fs, T = 20.0, 30.0
    n = int(T * fs)
    t = np.arange(n) / fs
    f = 0.1 * np.sign(np.sin(2 * np.pi * t / 10.0))
    x = L.simulate_linear(b, n, 1 / fs, 1, u=f[:, None, None], channel="common",
                          parts=("whole",), rng=rng)["whole"][:, 0, :b.n].reshape(n, -1, 2)
    A = _mag(0.15, np.abs(x).max())
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.1, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    lines = [ax.plot([], [], color=INK2, lw=0.8)[0] for _ in links]
    dots = ax.scatter(pts[:, 0], pts[:, 1], s=40, color=C1, zorder=3)
    arrow, = ax.plot([], [], color=C2, lw=2.5)
    _label(ax, f"tethered agents under a common push (orange); displacements {_fmt(A)}, sped up x{T / (N_FRAMES / FPS):.0f}")

    def update(k):
        i = int(k * (n - 1) / (N_FRAMES - 1))
        P = pts + A * x[i]
        for ln, (a, c) in zip(lines, links):
            ln.set_data([P[a, 0], P[c, 0]], [P[a, 1], P[c, 1]])
        dots.set_offsets(P)
        arrow.set_data([0.5, 0.5 + 1.5 * f[i]], [1.08, 1.08])
    return _write(fig, update, path)


# ---------------------------------------------------------------------------------------
# Nonlinear and contact bodies
# ---------------------------------------------------------------------------------------

def soft_arm(path):
    """The soft arm as a MuJoCo link chain: a tip push sets it swaying, then a curling couple
    ramps up and the arm wraps an object; side view, links drawn at their radius."""
    from .bodies.softarm import soft_arm as make_arm
    arm = make_arm(grasp=True)
    T = 4.0
    push = lambda t: np.array([[1.5e-2 if t < 0.05 else 0.0, 0.0]])
    curl = lambda t: -0.04 * np.clip((t - 1.5) / 1.5, 0.0, 1.0)
    out = arm.simulate(T, 1, tip_force=push, curl=curl, rng=np.random.default_rng(14))
    steps = out["q"].shape[1]
    idx = np.linspace(0, steps - 1, N_FRAMES).astype(int)
    pts = arm.centreline(out["q"][0, idx])
    p = arm.params
    s_mid = np.concatenate([[0.0], arm.s_joint, [p["length"]]])
    r = p["r_base"] + (p["r_tip"] - p["r_base"]) * s_mid / p["length"]
    fig, ax = plt.subplots(figsize=(4.4, 5.0))
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.02, 0.56)
    ax.set_aspect("equal")
    ax.axis("off")
    c = p["cylinder"]
    ax.add_patch(plt.Circle((c["x"], c["z"]), c["r"], color=C4, alpha=0.6))
    ax.plot([-0.03, 0.03], [0, 0], color=INK2, lw=4)
    # Line widths in points that match the arm's diameter at this axis scale.
    pt_per_m = fig.get_size_inches()[1] * 72 / 0.58 * 0.8
    segs = [ax.plot([], [], color=C1, lw=max(1.0, 2 * r[i] * pt_per_m),
                    solid_capstyle="round")[0] for i in range(len(r) - 1)]
    _label(ax, f"soft arm, MuJoCo link chain ({arm.n} links): a tip push,\n"
               f"then a curling couple wraps the object; real scale, slowed x{N_FRAMES / FPS / T:.1f}")

    def update(k):
        P = pts[k]
        for i, ln in enumerate(segs):
            ln.set_data(P[i:i + 2, 0], P[i:i + 2, 1])
    return _write(fig, update, path)


def bistable_chain(path):
    from .bodies.nonlinear import mesostate
    ch = make("bistable_chain")
    rng = np.random.default_rng(6)
    dt = 0.01
    Fc = ch.critical_forces
    prof = np.concatenate([np.linspace(0, 1.25 * Fc.max(), 3000), np.linspace(1.25 * Fc.max(), 0, 1500),
                           np.zeros(500)])
    n = len(prof)
    every = n // N_FRAMES
    cache = ch._ou(dt)
    q = np.tile(ch.q0, (1, 1))
    v = np.zeros((1, ch.n))
    Q, S = [], []
    for i, F in enumerate(prof):
        q, v, _ = ch.step(q, v, np.array([[F]]), rng, dt, cache)
        if i % every == 0:
            Q.append(q[0].copy())
            S.append(mesostate(ch, q)[0].copy())
    fig, (ax, tr) = plt.subplots(2, 1, figsize=(7.0, 3.6), gridspec_kw=dict(height_ratios=[1.3, 1]))
    ax.set_xlim(-2.0, 15.0)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    segs = [ax.plot([], [], lw=6, solid_capstyle="butt")[0] for _ in range(ch.n)]
    dots, = ax.plot([], [], "o", color=INK, ms=6)
    ax.plot([-1.5, -1.5], [-0.5, 0.5], color=INK2, lw=3)
    _label(ax, "bistable chain: short blue = left well, long orange = switched; "
               "ramp up, ramp down, wait")
    tt = np.arange(n) * dt
    cursor = _trace_panel(tr, tt, [prof], ["drive force"], [INK2])
    for F in Fc:
        tr.axhline(F / prof.max(), color=C2, lw=0.5, ls=":")

    def update(k):
        k = min(k, len(Q) - 1)
        # Display each element with length 1.5 + s (s = -1 short, +1 long), so wells read as
        # short and long segments from the wall at -1.5.
        s_el = np.diff(np.concatenate([[0.0], Q[k]]))
        x = -1.5 + np.concatenate([[0.0], np.cumsum(1.5 + s_el)])
        for j, sg in enumerate(segs):
            sg.set_data([x[j], x[j + 1]], [0, 0])
            sg.set_color(C2 if S[k][j] else C1)
        dots.set_data(x, np.zeros_like(x))
        cursor.set_xdata([tt[k * every]] * 2)
    return _write(fig, update, path)


def hopper(path):
    """The reduced hopper of Haeufle et al. with the muscle-fibre leg (Ghazi-Zahedi et al.
    2016); the bar shows the leg force."""
    from .bodies.contact import apexes
    h = make("hopper", actuator="musfib")
    T = 2.5
    o = h.simulate(T=T)
    n = len(o["t"])
    every = n // N_FRAMES
    y, F = o["y"][::every, 0], o["F"][::every, 0]
    ap = apexes(o["y"][:, 0], o["t"], t_min=0.5)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.2, 4.2), gridspec_kw=dict(width_ratios=[1, 0.35]))
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.05, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.axhline(0, color=INK2, lw=2)
    ax.axhline(h.l0, color=GRID, lw=0.8, ls=":")
    mass = plt.Circle((0, 1.0), 0.08, color=C1)
    ax.add_patch(mass)
    leg, = ax.plot([], [], color=INK, lw=2)
    bar = bx.bar([0], [0], color=C2)[0]
    bx.set_ylim(0, 3.5)
    bx.set_xticks([])
    bx.set_ylabel("leg force (kN)", fontsize=8)
    _label(ax, f"80 kg hopper, muscle-fibre leg; apex {ap.mean():.3f} m (published 1.070); "
               "real speed, slowed x2", "upper left")

    def update(k):
        k = min(k, len(y) - 1)
        mass.center = (0, y[k])
        foot = 0.0 if y[k] <= h.l0 else y[k] - h.l0
        leg.set_data([0, 0], [foot, y[k] - 0.08])
        bar.set_height(F[k] / 1e3)
    return _write(fig, update, path)


def gripper(path):
    gp = make("gripper")
    rng = np.random.default_rng(8)
    dt, T = 1e-4, 1.0
    n = int(T / dt)
    every = n // N_FRAMES
    tt = np.arange(n) * dt
    d = np.zeros(n)
    for t0 in (0.2, 0.5, 0.75):
        m = (tt > t0) & (tt < t0 + 0.02)
        d[m] = 1.5 * np.sin(np.pi * (tt[m] - t0) / 0.02)
    out, _ = gp.simulate(n, dt, 1, u=d[:, None, None], rng=rng, parts=("whole",),
                         q_init=gp.q0[None], v_init=np.zeros((1, 3)))
    x = out["whole"][::every, 0, :3]
    dx = x - gp.q0
    A = _mag(0.01, np.abs(dx).max())
    r_o = gp.params["r_o"]
    fig, (ax, tr) = plt.subplots(2, 1, figsize=(6.0, 3.8), gridspec_kw=dict(height_ratios=[1.2, 1]))
    ax.set_xlim(-0.07, 0.07)
    ax.set_ylim(-0.03, 0.03)
    ax.set_aspect("equal")
    ax.axis("off")
    obj = plt.Rectangle((-r_o, -0.015), 2 * r_o, 0.03, color=C4, alpha=0.6)
    padL = plt.Rectangle((0, -0.02), 0.006, 0.04, color=C1)
    padR = plt.Rectangle((0, -0.02), 0.006, 0.04, color=C1)
    for p_ in (obj, padL, padR):
        ax.add_patch(p_)
    _label(ax, f"gripper squeezing an object, taps on the object; motion {_fmt(A)}, slowed x5")
    cursor = _trace_panel(tr, tt, [d, out["whole"][:, 0, 1] - gp.q0[1]], ["disturbance", "object"], [C2, C1])

    def update(k):
        k = min(k, len(x) - 1)
        xl, xo, xr = gp.q0 + A * dx[k]
        obj.set_x(xo - r_o)
        padL.set_x(xl - 0.006)
        padR.set_x(xr)
        cursor.set_xdata([tt[k * every]] * 2)
    return _write(fig, update, path)


def walker(path):
    wk = make("walker", gamma=0.1)
    p = wk.params
    rng = np.random.default_rng(9)
    dt, T = 1e-3, 10.0
    n = int(T / dt)
    every = n // N_FRAMES
    q = np.zeros((1, 1))
    v = np.full((1, 1), 1.0)
    cache = {"dt": dt, "steps": np.zeros(1)}
    TH, ST = [], []
    for i in range(n):
        q, v, _ = wk.step(q, v, None, rng, dt, cache)
        if i % every == 0:
            TH.append(q[0, 0])
            ST.append(cache["steps"][0])
    a, g = p["alpha"], p["gamma"]
    step = 2 * np.sin(a)
    n_spokes = int(round(np.pi / a))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.set_aspect("equal")
    ax.axis("off")
    span = step * (max(ST) + 3)
    xs = np.array([-1, span + 1])
    ax.plot(xs * np.cos(g), -xs * np.sin(g), color=INK2, lw=1.5)
    spokes = [ax.plot([], [], color=INK, lw=2)[0] for _ in range(n_spokes)]
    hub, = ax.plot([], [], "o", color=C1, ms=8)
    _label(ax, f"rimless wheel on a {g:.2f} rad slope; real scale and speed")

    def update(k):
        k = min(k, len(TH) - 1)
        s = ST[k] * step                              # stance foot along the slope
        foot = np.array([s * np.cos(g), -s * np.sin(g)])
        ang = TH[k] - g                               # stance spoke angle from vertical
        c = foot + np.array([np.sin(ang), np.cos(ang)])
        for j, sp in enumerate(spokes):
            phi = ang + np.pi + 2 * a * j
            sp.set_data([c[0], c[0] + np.sin(phi)], [c[1], c[1] + np.cos(phi)])
        hub.set_data([c[0]], [c[1]])
        ax.set_xlim(c[0] - 3, c[0] + 3)
        ax.set_ylim(c[1] - 2, c[1] + 1.5)
    return _write(fig, update, path)


def many_legged(path):
    """Compliant-legged and stiff-legged versions of the same 12-legged robot on the same
    rough terrain. Legs are drawn from hip to the simulated foot position."""
    from .bodies.contact import crawl
    T, dt = 4.0, 2e-4
    every = int(T / dt) // N_FRAMES
    runs = []
    for kl in (150.0, 3e4):
        robot = make("many_legged", n_legs=12, rugosity=0.015, k_leg=kl)
        r = crawl(robot, 1, T, dt=dt, rng=np.random.default_rng(10), record_every=every)
        runs.append((kl, robot, r))
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 4.4))
    arts = []
    for ax, (kl, robot, r) in zip(axs, runs):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("compliant legs" if kl < 1e3 else "stiff legs", fontsize=9)
        terrain, = ax.plot([], [], color=INK2, lw=1)
        body, = ax.plot([], [], color=C1, lw=5, solid_capstyle="round")
        legs = [ax.plot([], [], lw=1.5)[0] for _ in range(robot.params["n_legs"])]
        arts.append((ax, terrain, body, legs))
    fig.text(0.5, 0.02, "same rough terrain (sd 1.5 cm); green = stance foot touching, red = stance "
             "foot missing, grey = swing; real scale", ha="center", fontsize=7.5, color=INK2)

    def update(k):
        for (ax, terrain, body, legs), (kl, robot, r) in zip(arts, runs):
            fr = r["frames"][min(k, len(r["frames"]) - 1)]
            q = fr["q"][0]
            h, res = r["terrain"]
            xs = (np.arange(h.shape[1]) - h.shape[1] // 2) * res
            L = robot.params["body_len"]
            win = (xs > q[0] - L) & (xs < q[0] + L)
            terrain.set_data(xs[win], h[0, win])
            c, s_ = np.cos(q[2]), np.sin(q[2])
            hx = q[0] + robot.hips * c
            hz = q[1] + robot.hips * s_
            body.set_data([hx[0], hx[-1]], [hz[0], hz[-1]])
            for j, lg in enumerate(legs):
                fx, fz = fr["pos"][0, j]
                if fr["touch"][0, j]:          # a touching foot sits on the surface (leg compressed)
                    gi = int(np.clip(fx / res + h.shape[1] // 2, 0, h.shape[1] - 1))
                    fz = max(fz, h[0, gi])
                lg.set_data([hx[j], fx], [hz[j], fz])
                touch, st = fr["touch"][0, j], fr["stance"][j]
                lg.set_color(C3 if touch else (C2 if st else "#b9b8b4"))
            ax.set_xlim(q[0] - 0.75 * L, q[0] + 0.75 * L)
            ax.set_ylim(-0.06, 0.16)
    return _write(fig, update, path)


def swarmalators(path):
    """The five published states of swarmalators (O'Keeffe et al. 2017), from random starts;
    colour is phase."""
    from .bodies.swarmalators import STATES, Swarmalators
    dt, T = 0.05, 120.0
    n = int(T / dt)
    every = n // N_FRAMES
    runs = []
    for name, (J, K) in STATES.items():
        sw = Swarmalators(N=150, J=J, K=K, seed=1)
        _, _, frames = sw.simulate(n, dt, 1, rng=np.random.default_rng(1), record_every=every)
        runs.append((name, J, K, frames))
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.9))
    cmap = plt.get_cmap("hsv")
    arts = []
    for ax, (name, J, K, frames) in zip(axs.ravel(), runs):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}\nJ={J:g}, K={K:g}", fontsize=8)
        sc = ax.scatter([], [], s=8, c=[], cmap=cmap, vmin=0, vmax=2 * np.pi)
        arts.append((ax, sc))
    axs.ravel()[-1].axis("off")
    axs.ravel()[-1].text(0.5, 0.5, "swarmalators\n(O'Keeffe et al. 2017)\ncolour = phase\n"
                         f"{T:.0f} time units from random starts", ha="center", va="center",
                         fontsize=8, color=INK2, transform=axs.ravel()[-1].transAxes)

    def update(k):
        for (ax, sc), (_, _, _, frames) in zip(arts, runs):
            x, th = frames[min(k, len(frames) - 1)]
            sc.set_offsets(x[0])
            sc.set_array(np.mod(th[0], 2 * np.pi))
            c = x[0].mean(0)
            ax.set_xlim(c[0] - 1.6, c[0] + 1.6)
            ax.set_ylim(c[1] - 1.6, c[1] + 1.6)
    return _write(fig, update, path)


# ---------------------------------------------------------------------------------------
# Fluids
# ---------------------------------------------------------------------------------------

def foil(path):
    from .bodies.fluids import karman_gust
    b = make("foil")
    p = b.params
    U, d, St = p["U"], 0.3, 0.2
    fs, T = 200.0, 5.0
    n = int(T * fs)
    rng = np.random.default_rng(11)
    g = karman_gust(n, 1 / fs, 1, U=U, d=d, St=St, amp=0.5, rng=rng)
    out = L.simulate_linear(b, n, 1 / fs, 1, u=g, channel="gust", parts=("plunge", "pitch"), rng=rng)
    h, al = out["plunge"][:, 0, 0], out["pitch"][:, 0, 0]
    Ah = _mag(0.15, np.abs(h).max())
    Aa = _mag(0.35, np.abs(al).max())
    chord = 2 * p["b"]
    f_shed = St * U / d
    lam = U / f_shed
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(-4.0, 0.8)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    s = np.linspace(0, np.pi, 40)
    xf = 0.5 * chord * (1 - np.cos(s))
    tf = 0.06 * chord * np.sin(s) * (1 - 0.6 * (xf / chord))
    foil_line, = ax.plot([], [], color=INK, lw=1.5)
    top = ax.scatter([], [], s=40, color=C1)
    bot = ax.scatter([], [], s=40, color=C2)
    ax.annotate("", xy=(-3.2, -0.45), xytext=(-3.9, -0.45),
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.text(-3.9, -0.55, "flow", fontsize=7, color=INK2)
    _label(ax, f"foil in a vortex street ({f_shed:.0f} Hz shedding, wavelength {lam:.1f} m); "
               f"plunge {_fmt(Ah)}, pitch {_fmt(Aa)}, real speed")

    def update(k):
        i = int(k * (n - 1) / (N_FRAMES - 1))
        t = i / fs
        th = -Aa * al[i]
        xc = np.concatenate([xf, xf[::-1]]) - chord / 2
        yc = np.concatenate([tf, -tf[::-1]])
        X = xc * np.cos(th) - yc * np.sin(th)
        Y = xc * np.sin(th) + yc * np.cos(th) - Ah * h[i]
        foil_line.set_data(X, Y)
        shift = (U * t) % lam                        # vortices travel downstream (+x)
        xt = np.arange(-6 * lam, 2 * lam, lam) + shift
        xb = xt + lam / 2
        top.set_offsets(np.c_[xt, np.full_like(xt, 0.2)])
        bot.set_offsets(np.c_[xb, np.full_like(xb, -0.2)])
    return _write(fig, update, path)


def bird(path):
    rng = np.random.default_rng(12)
    fs, T = 200.0, 5.0
    n = int(T * fs)
    w = np.zeros(n)
    t = np.arange(n) / fs
    for t0 in (0.5, 2.0, 3.5):
        m = (t > t0) & (t < t0 + 0.15)
        w[m] = 2.0 * np.sin(np.pi * (t[m] - t0) / 0.15)
    runs = []
    for ks in (20.0, 1e4):
        b = make("bird", k_s=ks)
        out = L.simulate_linear(b, n, 1 / fs, 1, u=w[:, None, None], channel="gust",
                                parts=("whole",), rng=rng, stationary=False)["whole"][:, 0]
        yb = np.cumsum(out[:, 1]) / fs
        k = int(fs)                                   # camera follows a 1 s moving average
        slow = np.convolve(yb, np.ones(k) / k, mode="same")
        yb = yb - slow
        yw = yb + out[:, 0]
        runs.append((ks, yb, yw))
    A = _mag(0.6, max(np.abs(r[2]).max() for r in runs))
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.6))
    arts = []
    for ax, (ks, yb, yw) in zip(axs, runs):
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis("off")
        ax.set_title("compliant shoulder" if ks < 1e3 else "stiff shoulder", fontsize=9)
        body = plt.Rectangle((-0.15, 0), 0.3, 0.2, color=C1)
        wing = plt.Rectangle((-0.8, 0), 1.6, 0.05, color=INK)
        ax.add_patch(wing)
        ax.add_patch(body)
        gust, = ax.plot([], [], color=C2, lw=3)
        arts.append((body, wing, gust))
    _label(axs[0], f"gust pulses (orange); camera follows the slow climb; heights {_fmt(A)}",
           "lower left")

    def update(k):
        i = int(k * (n - 1) / (N_FRAMES - 1))
        for (body, wing, gust), (_, yb, yw) in zip(arts, runs):
            body.set_y(A * yb[i] - 0.1)
            wing.set_y(A * yw[i] + 0.1)
            gust.set_data([0.9, 0.9], [-0.9, -0.9 + 0.3 * w[i]])
    return _write(fig, update, path)


def pendulum(path):
    """Stable spring, weak spring left alone, weak spring with the canonical (mirror)
    stabilizing feedback. Nonlinear pendulum, the same pushes for all three."""
    from .sampling import MechSim, mirror_feedback
    m, l, g, b = 1.0, 1.0, 9.81, 0.5
    dt, T = 1e-3, 6.0
    n = int(T / dt)
    every = n // N_FRAMES
    tt = np.arange(n) * dt
    push = np.zeros(n)
    for t0 in (1.0, 3.5):
        mask = (tt > t0) & (tt < t0 + 0.1)
        push[mask] = 3.0
    runs = []
    for k, ctrl in ((15.0, False), (5.0, False), (5.0, True)):
        sim = MechSim(H=np.array([[m * l * l]]), D=np.array([[b]]),
                      force=lambda q, p, k=k: m * g * l * np.sin(q) - k * q,
                      J=lambda q: np.full((q.shape[0], 1, 1), l), sigma_T2=1e-3, params={})
        env = lambda kk, q, v: np.array([[push[min(kk, n - 1)]]])
        u = env
        if ctrl:
            u, _ = mirror_feedback(sim, np.array([[l]]), base_input=env)
        rng = np.random.default_rng(13)
        out, _ = sim.simulate(n, dt, 1, u=u, rng=rng, parts=("whole",),
                              q_init=np.full((1, 1), 0.02), v_init=np.zeros((1, 1)))
        th = np.clip(out["whole"][::every, 0, 0], -np.pi / 2, np.pi / 2)   # the floor stops a fall
        runs.append(th)
    titles = ["stiff spring: stable", "weak spring: falls", "weak spring + mirror feedback"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0))
    arts = []
    for ax, ttl in zip(axs, titles):
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.2, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(ttl, fontsize=8.5)
        ax.axhline(0, color=INK2, lw=1)
        rod, = ax.plot([], [], color=INK, lw=2.5)
        bob, = ax.plot([], [], "o", color=C1, ms=10)
        arr, = ax.plot([], [], color=C2, lw=2)
        arts.append((rod, bob, arr))
    fig.text(0.5, 0.06, "nonlinear pendulum, the same two pushes for all three; real scale and speed",
             ha="center", fontsize=8, color=INK2)

    def update(k):
        k = min(k, len(runs[0]) - 1)
        i = min(k * every, n - 1)
        for (rod, bob, arr), th in zip(arts, runs):
            x, y = l * np.sin(th[k]), l * np.cos(th[k])
            rod.set_data([0, x], [0, y])
            bob.set_data([x], [y])
            arr.set_data([x - 0.4 * push[i] / 3, x], [y, y] if push[i] > 0 else [y, y])
    return _write(fig, update, path)


def hauser_network(path):
    """Hauser et al. (2011) network on the toolbox's Volterra task: learn a linear readout
    of the spring lengths, then run on new input and compare with the target."""
    from .bodies import hauser as H
    net = H.build(seed=0, ranges=H.VOLTERRA)
    d = H.volterra_data(120.0)
    s, n_learn, wash, n_show = 20000, 50001, 20000, 4000
    U, Y = d["u"][s:s + n_learn], d["y"][s:s + n_learn]
    Ut, Yt = d["u"][s + n_learn:s + n_learn + n_show], d["yn"][s + n_learn:s + n_learn + n_show]
    Lr, st = H.simulate(net, U)
    w = np.linalg.lstsq(Lr[wash:], Y[wash:], rcond=None)[0]
    # Test run, recording node positions for the drawing.
    P_hist = []
    state = st
    Lt = np.empty((n_show, net.n_springs))
    for k0 in range(0, n_show, 50):
        seg, state = H.simulate(net, Ut[k0:k0 + 50], state)
        Lt[k0:k0 + 50] = seg
        P_hist.append(state.P[0].copy())
    P_hist = np.array(P_hist)
    out = H.mapstd(Lt @ w)
    t = np.arange(n_show) * H.DT
    disp = P_hist - net.pos0[None]
    A = _mag(0.6, np.abs(disp).max())
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_axes([0.02, 0.36, 0.96, 0.62])
    ax2 = fig.add_axes([0.08, 0.07, 0.88, 0.22])
    ax.set_xlim(-0.8, 10.8)
    ax.set_ylim(-0.8, 10.8)
    ax.set_aspect("equal")
    ax.axis("off")
    lines = [ax.plot([], [], color=INK2, lw=0.8)[0] for _ in net.edges]
    nodes = ax.scatter(net.pos0[:, 0], net.pos0[:, 1], s=22, color=C1, zorder=3)
    fx = net.fixed
    ax.scatter(net.pos0[fx, 0], net.pos0[fx, 1], s=60, marker="s", color=INK, zorder=4)
    inp = np.flatnonzero(net.w_in)
    arrows = [ax.plot([], [], color=C2, lw=2)[0] for _ in inp]
    _label(ax, f"Hauser et al. 2011 network, Volterra task; displacements {_fmt(A)}\n"
               "squares fixed, orange arrows: input forces")
    ax2.plot(t, Yt, color=C2, lw=1.2, label="target (Volterra operator)")
    ax2.plot(t, out, color=C1, lw=1.0, ls="--", label="linear readout of spring lengths")
    cursor = ax2.axvline(0, color=INK, lw=0.8)
    ax2.legend(fontsize=7, loc="upper right", frameon=False)
    ax2.set_xlabel("time (s)", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.set_xlim(0, t[-1])

    def update(k):
        i = min(int(k * (len(P_hist) - 1) / (N_FRAMES - 1)), len(P_hist) - 1)
        P = net.pos0 + A * disp[i]
        for ln, (a, c) in zip(lines, net.edges):
            ln.set_data([P[a, 0], P[c, 0]], [P[a, 1], P[c, 1]])
        nodes.set_offsets(P)
        ti = i * 50
        for ar, j in zip(arrows, inp):
            f = net.w_in[j] * Ut[ti] / 0.2
            ar.set_data([P[j, 0], P[j, 0] + 1.2 * f], [P[j, 1], P[j, 1]])
        cursor.set_xdata([t[ti], t[ti]])
    return _write(fig, update, path)


def gull_glider(path):
    """Harvey and Inman (2022) gull, side view, after the paper's transverse gust (a 2 deg
    offset in angle of attack): a stable wrist angle and an unstable one, elbow 130 deg,
    15 deg forward sweep."""
    from scipy.linalg import expm
    cases = [(126, C1, "wrist 126 deg (stable)"), (166, C2, "wrist 166 deg (unstable)")]
    T, n = 3.0, N_FRAMES
    t = np.linspace(0, T, n)
    runs = []
    for w, col, lab in cases:
        b = make("gull_glider", elbow=130, wrist=w, sweep=-15, dihedral=20)
        x0 = np.array([0.0, np.radians(2.0), 0.0, 0.0])
        X = np.array([expm(b.A * tk) @ x0 for tk in t])
        p = b.params
        U = p["U0"] * (1 + X[:, 0])
        gam = p["gamma0"] + X[:, 3] - X[:, 1]
        dt = t[1] - t[0]
        # Position relative to a point moving along the trim glide path.
        xs = np.cumsum(U * np.cos(gam) - p["U0"] * np.cos(p["gamma0"])) * dt
        zs = np.cumsum(U * np.sin(gam) - p["U0"] * np.sin(p["gamma0"])) * dt
        theta = p["gamma0"] + np.radians(p["alpha0"]) + X[:, 3]
        runs.append(dict(X=X, xs=xs, zs=zs, theta=theta, col=col, lab=lab, p=p))
    fig = plt.figure(figsize=(7.2, 6.0))
    axes = [fig.add_axes([0.03 + 0.49 * k, 0.42, 0.46, 0.55]) for k in range(2)]
    ax2 = fig.add_axes([0.10, 0.08, 0.86, 0.27])
    body = np.array([[0.22, 0.0], [0.12, 0.035], [-0.05, 0.04], [-0.2, 0.012], [-0.32, 0.02],
                     [-0.3, -0.005], [-0.05, -0.035], [0.12, -0.03]])
    patches, trails = [], []
    for ax, r in zip(axes, runs):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.axis("off")
        g0 = r["p"]["gamma0"]
        ax.plot([-3 * np.cos(g0), 3 * np.cos(g0)], [-3 * np.sin(g0), 3 * np.sin(g0)],
                color=GRID, lw=1, ls="--")
        patches.append(ax.fill([], [], color=r["col"])[0])
        trails.append(ax.plot([], [], color=r["col"], lw=1)[0])
        ax.text(0.5, 0.02, f"{r['lab']}\nU0 = {r['p']['U0']:.1f} m/s, glide {np.degrees(-g0):.0f} deg",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color=INK2)
    _label(axes[0], "Harvey and Inman 2022 gull: 2 deg angle-of-attack gust at t = 0\n"
                    "dashed: trim glide path; path deviation to scale, bird drawn x3")
    for r in runs:
        ax2.plot(t, np.degrees(r["X"][:, 1]), color=r["col"], lw=1.2, label=r["lab"] + ": angle of attack")
        ax2.plot(t, np.degrees(r["X"][:, 3]), color=r["col"], lw=1.0, ls="--", label="pitch angle")
    ax2.set_ylim(-6, 6)
    ax2.set_xlim(0, T)
    ax2.set_xlabel("time (s)", fontsize=8)
    ax2.set_ylabel("perturbation (deg)", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=6.5, loc="lower left", frameon=False, ncol=2)
    cursor = ax2.axvline(0, color=INK, lw=0.8)

    def update(k):
        for r, pa, tr in zip(runs, patches, trails):
            th = r["theta"][k]
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            P = 3 * body @ R.T + [r["xs"][k], r["zs"][k]]
            pa.set_xy(P)
            tr.set_data(r["xs"][:k + 1], r["zs"][:k + 1])
        cursor.set_xdata([t[k], t[k]])
    return _write(fig, update, path)


DEMOS = dict(legs=legs, network=network, whisker=whisker, agents=agents, soft_arm=soft_arm,
             bistable_chain=bistable_chain, hopper=hopper, gripper=gripper,
             walker=walker, many_legged=many_legged, foil=foil, bird=bird, pendulum=pendulum,
             swarmalators=swarmalators, hauser_network=hauser_network,
             gull_glider=gull_glider)


def render(outdir, names=None):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    done = []
    for name in (names or DEMOS):
        path = DEMOS[name](outdir / f"{name}.mp4")
        print("wrote", path, flush=True)
        done.append(path)
    return done


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2:] or None)
