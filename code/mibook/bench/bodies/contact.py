"""Contact and switching benchmark bodies (sampling engine).

Contact is a penalty spring and damper that acts only while two bodies overlap. Contact
damping is deterministic (`velocity_force`); thermal noise comes from the body's own
dampers D. Bodies whose motion jumps at impacts, or whose time matters, override `step`.

- `hopper`: a vertical spring-loaded hopper on a ground of unknown stiffness.
- `gripper`: two compliant finger pads squeezing an object of unknown stiffness.
- `walker`: a rimless wheel, the simplest passive walker, on a slope.
- `tentacle`: the soft arm curled by a tendon around a round object.
- `many_legged`: a many-legged robot whose stance legs touch rough ground at random.
"""
from __future__ import annotations

import numpy as np

from .. import register
from ..sampling import MechSim
from .nonlinear import soft_arm

G = 9.81


def _sel(n, *idx):
    C = np.zeros((len(idx), 2 * n))
    for r, c in enumerate(idx):
        C[r, c] = 1.0
    return C


# ---------------------------------------------------------------------------------------
# Hopper
# ---------------------------------------------------------------------------------------

@register("hopper", "3-contact",
          "Vertical spring-loaded hopper (body and foot) on a ground of unknown stiffness.")
def hopper(M: float = 10.0, m_f: float = 1.0, k_l: float = 5000.0, l0: float = 0.5,
           b_l: float = 20.0, b_f: float = 2.0, b_b: float = 0.5, k_g: float = 1e5,
           b_g: float = 300.0, thrust: float = 300.0, sigma_T2: float = 1e-4) -> MechSim:
    """q = (body height, foot height). The leg is a spring k_l with damper b_l between them;
    the ground pushes the foot with k_g (h - y_f) - b_g y_f' while they overlap (h = 0).
    `thrust` is the size of a Raibert-style push between body and foot during stance while
    the body rises, which keeps the hopping going; `hop_input` gives it."""
    H = np.diag([M, m_f])
    a = np.array([1.0, -1.0])
    D = b_l * np.outer(a, a) + np.diag([b_b, b_f])
    params = dict(k_g=k_g, b_g=b_g, k_l=k_l)

    def force(q, p):
        s = q[:, 0] - q[:, 1] - l0                    # leg stretch
        f_leg = -p["k_l"] * s
        pen = np.clip(-q[:, 1], 0.0, None)            # foot below the ground
        F = np.stack([f_leg - M * G, -f_leg - m_f * G + p["k_g"] * pen], axis=1)
        return F

    def vforce(q, v, p):
        inside = q[:, 1] < 0.0
        Fg = np.where(inside, -p["b_g"] * v[:, 1], 0.0)
        # The ground can push but not pull.
        Fg = np.maximum(Fg, -p["k_g"] * np.clip(-q[:, 1], 0.0, None))
        return np.stack([np.zeros_like(Fg), Fg], axis=1)

    J = np.array([[1.0, -1.0]])
    q0 = np.array([l0 - (M * G) / k_l - (M + m_f) * G / k_g, -(M + m_f) * G / k_g])
    sim = MechSim(H=H, D=D, force=force,
                  J=lambda q: np.broadcast_to(J, (q.shape[0], 1, 2)),
                  sigma_T2=sigma_T2, params=params, velocity_force=vforce,
                  parts={"body": _sel(2, 0), "foot": _sel(2, 1), "leg": np.array([[1.0, -1.0, 0, 0]])},
                  name="hopper", q0=q0)

    def hop_input(k, q, v):
        stance = q[:, 1] < 0.0
        rising = v[:, 0] > 0.0
        return np.where(stance & rising, thrust, 0.0)[:, None]

    sim.hop_input = hop_input
    return sim


# ---------------------------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------------------------

@register("gripper", "3-contact",
          "Two compliant finger pads squeezing an object of unknown stiffness and size.")
def gripper(m_p: float = 0.05, m_o: float = 0.2, k_f: float = 200.0, b_f: float = 1.0,
            preload: float = 0.01, r_o: float = 0.03, k_o: float = 5000.0, b_o: float = 2.0,
            b_o_air: float = 0.05, sigma_T2: float = 1e-6) -> MechSim:
    """q = (left pad, object centre, right pad) along the squeeze axis. Each pad is held by a
    spring k_f whose rest point lies `preload` inside the object's surface, so the fingers
    squeeze. The object (half-width r_o) touches a pad with stiffness k_o and damping b_o
    while they overlap. Inputs: a disturbance force on the object."""
    H = np.diag([m_p, m_o, m_p])
    D = np.diag([b_f, b_o_air, b_f])
    params = dict(k_o=k_o, r_o=r_o, b_o=b_o)
    rest_L, rest_R = -r_o + preload, r_o - preload

    def force(q, p):
        xL, xo, xR = q[:, 0], q[:, 1], q[:, 2]
        penL = np.clip(xL - (xo - p["r_o"]), 0.0, None)
        penR = np.clip((xo + p["r_o"]) - xR, 0.0, None)
        fL = p["k_o"] * penL
        fR = p["k_o"] * penR
        return np.stack([-k_f * (xL - rest_L) - fL, fL - fR, -k_f * (xR - rest_R) + fR], axis=1)

    def vforce(q, v, p):
        xL, xo, xR = q[:, 0], q[:, 1], q[:, 2]
        inL = xL > xo - p["r_o"]
        inR = xR < xo + p["r_o"]
        dL = np.where(inL, p["b_o"] * (v[:, 0] - v[:, 1]), 0.0)
        dR = np.where(inR, p["b_o"] * (v[:, 1] - v[:, 2]), 0.0)
        return np.stack([-dL, dL - dR, dR], axis=1)

    J = np.array([[0.0, 1.0, 0.0]])
    # Static squeeze: each pad presses with k_f d where d solves k_f (preload - d) = k_o d.
    d = k_f * preload / (k_f + k_o)
    q0 = np.array([-r_o + d, 0.0, r_o - d])
    return MechSim(H=H, D=D, force=force,
                   J=lambda q: np.broadcast_to(J, (q.shape[0], 1, 3)),
                   sigma_T2=sigma_T2, params=params, velocity_force=vforce,
                   parts={"left_pad": _sel(3, 0), "pads": _sel(3, 0, 2), "object": _sel(3, 1)},
                   name="gripper", q0=q0)


# ---------------------------------------------------------------------------------------
# Passive walker: the rimless wheel
# ---------------------------------------------------------------------------------------

class RimlessWheel(MechSim):
    """theta is the stance spoke's angle from the slope normal; gravity gives
    theta'' = (g/l) sin(theta + gamma) - b theta' + thermal noise. When theta reaches alpha
    the next spoke strikes: theta -> theta - 2 alpha and theta' -> cos(2 alpha) theta'
    (angular momentum about the new contact); rocking back to -alpha strikes the rear spoke
    the same way. On a shallow slope the wheel comes to rest on two spokes, as a real rimless
    wheel does; an impact that leaves almost no speed puts it in that resting state.
    `steps` counts net footfalls: forward strikes minus rear strikes."""

    def _ou(self, dt):
        return {"dt": dt}

    def step(self, q, v, u, rng, dt, cache, a=None):
        p = self.params
        b = self.D[0, 0] / self.H[0, 0]
        rest = cache.setdefault("rest", np.zeros(q.shape, bool))
        acc = p["g_over_l"] * np.sin(q + p["gamma"]) - b * v
        noise = np.sqrt(2.0 * self.sigma_T2 * b / self.H[0, 0] * dt) * rng.standard_normal(v.shape)
        v = np.where(rest, 0.0, v + acc * dt + noise)
        q = np.where(rest, q, q + v * dt)
        hit = (q >= p["alpha"]) & (v > 0)
        back = (q <= -p["alpha"]) & (v < 0)               # rocking back onto the rear spoke
        q = np.where(hit, q - 2 * p["alpha"], np.where(back, q + 2 * p["alpha"], q))
        v = np.where(hit | back, np.cos(2 * p["alpha"]) * v, v)
        # An impact that leaves almost no speed means the wheel has settled on two spokes.
        v_min = 1e-3 * np.sqrt(p["g_over_l"])
        settle = (hit | back) & (np.abs(v) < v_min)
        rest |= settle
        q = np.where(settle, -p["alpha"], q)
        v = np.where(rest, 0.0, v)
        if "steps" in cache:
            cache["steps"] += hit[:, 0].astype(float) - back[:, 0]   # net spokes advanced
        return q, v, None


@register("walker", "3-contact", "Rimless wheel (simplest passive walker) on a slope.")
def walker(gamma: float = 0.08, alpha: float = np.pi / 8, g_over_l: float = G, b: float = 0.0,
           sigma_T2: float = 1e-5) -> MechSim:
    sim = RimlessWheel(H=np.eye(1), D=np.array([[max(b, 1e-3)]]),
                       force=lambda q, p: np.zeros_like(q),
                       J=lambda q: np.zeros((q.shape[0], 1, 1)), sigma_T2=sigma_T2,
                       params=dict(gamma=gamma, alpha=alpha, g_over_l=g_over_l),
                       parts={"angle": _sel(1, 0), "rate": _sel(1, 1)}, name="rimless wheel",
                       q0=np.array([0.0]))
    return sim


def walk(sim: RimlessWheel, batch: int, t_end: float, dt: float = 1e-3, rng=None,
         v0: float = 1.0) -> dict:
    """Roll `batch` wheels for t_end seconds; returns steps per second and mean speed."""
    rng = np.random.default_rng() if rng is None else rng
    q = np.zeros((batch, 1))
    v = np.full((batch, 1), v0)
    cache = {"dt": dt, "steps": np.zeros(batch)}
    n = int(t_end / dt)
    burn = n // 3
    for k in range(n):
        if k == burn:
            cache["steps"][:] = 0.0
        q, v, _ = sim.step(q, v, None, rng, dt, cache)
    T = (n - burn) * dt
    step_len = 2 * np.sin(sim.params["alpha"])
    return dict(rate=cache["steps"] / T, speed=cache["steps"] * step_len / T, v=v[:, 0])


# ---------------------------------------------------------------------------------------
# Tentacle grasping
# ---------------------------------------------------------------------------------------

@register("tentacle", "3-contact",
          "Soft arm curled by a tendon along one rail around a round object.")
def tentacle(r_obj: float = 0.04, cx: float = -0.06, cy: float = 0.2, k_obj: float = 2000.0,
             b_obj: float = 0.5, b_spring: float = 0.5, **arm_kw) -> MechSim:
    """The soft arm (`soft_arm`) with a tendon along its left rail: tension T pulls each
    left-rail segment's ends together, so the arm curls to the left. A rigid-ish disc of
    radius r_obj at (cx, cy) pushes back on any node inside it. Input "tendon": T (N)."""
    arm = soft_arm(b_spring=b_spring, **arm_kw)
    N, n, rest = arm.n_seg, arm.n, arm.rest
    left = arm.left
    base_L = arm.base[0]
    nn = n // 2
    base_force = arm.force

    def nodes(q):
        return q.T.reshape(nn, 2, -1) + rest[:, :, None]

    def J(q):
        X = nodes(q)
        Fg = np.zeros_like(X)
        prev = np.broadcast_to(base_L[:, None], X[left[0]].shape)
        for idx, i in enumerate(left):
            d = X[i] - prev
            u = d / np.sqrt(d[0] ** 2 + d[1] ** 2)
            Fg[i] -= u                               # pulled back toward the previous node
            if idx > 0:
                Fg[left[idx - 1]] += u
            prev = X[i]
        return Fg.reshape(n, -1).T[:, None, :]

    c = np.array([cx, cy])[:, None]

    def force(q, p):
        F = base_force(q, p).T.reshape(nn, 2, -1)
        X = nodes(q)
        d = X - c
        dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        pen = np.clip(p["r_obj"] - dist, 0.0, None)
        F += (p["k_obj"] * pen / np.maximum(dist, 1e-9))[:, None, :] * d
        return F.reshape(n, -1).T

    def vforce(q, v, p):
        X = nodes(q)
        V = v.T.reshape(nn, 2, -1)
        d = X - c
        dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        nrm = d / np.maximum(dist, 1e-9)[:, None, :]
        vn = (V * nrm).sum(axis=1)
        inside = dist < p["r_obj"]
        F = np.where(inside, -p["b_obj"] * vn, 0.0)[:, None, :] * nrm
        return F.reshape(n, -1).T

    params = dict(arm.params, r_obj=r_obj, k_obj=k_obj, b_obj=b_obj)
    sim = MechSim(H=arm.H, D=arm.D, force=force, J=J, sigma_T2=arm.sigma_T2, params=params,
                  parts=arm.parts, name="tentacle", velocity_force=vforce)
    sim.rest, sim.n_seg = rest, N

    def in_contact(q):
        X = nodes(q)
        d = X - c
        return np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) < sim.params["r_obj"]

    sim.in_contact = in_contact

    def grasp_input(tension: float = 1.0, ramp_steps: int = 2500):
        """Tendon tension ramped linearly to `tension` over `ramp_steps` steps, then held."""
        return lambda k, q, v: np.full((q.shape[0], 1), min(1.0, k / ramp_steps) * tension)

    sim.grasp_input = grasp_input
    return sim


# ---------------------------------------------------------------------------------------
# Hyper-redundant many-legged robot
# ---------------------------------------------------------------------------------------

class ManyLegged(MechSim):
    """Forward motion x of a body with n_legs legs spaced `spacing` apart. Leg i is in stance
    when sin(omega t - 2 pi i / wave) > 0; a stance leg pushes with force f_leg only if the
    ground under its foot is within reach: terrain height h(x_i) > -reach. Terrain is a
    random field of rugosity `rugosity` (standard deviation) and correlation length `corr`,
    one realization per batch member. Thermal noise from the drag b."""

    def _ou(self, dt):
        return {"dt": dt, "t": 0.0}

    def terrain(self, batch, rng, length=40.0, res=0.005):
        p = self.params
        n = int(length / res)
        w = rng.standard_normal((batch, n))
        k = max(1, int(p["corr"] / res))
        ker = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2)
        ker /= np.sqrt((ker ** 2).sum())
        h = np.stack([np.convolve(row, ker, mode="same") for row in w]) * p["rugosity"]
        return h, res

    def step(self, q, v, u, rng, dt, cache, a=None):
        p = self.params
        t = cache["t"]
        h, res = cache["terrain"]
        i = np.arange(p["n_legs"])
        stance = np.sin(p["omega"] * t - 2 * np.pi * i / p["wave"]) > 0.0
        xi = q[:, :1] + i[None, :] * p["spacing"]
        idx = np.clip((xi / res).astype(int), 0, h.shape[1] - 1)
        ground = np.take_along_axis(h, idx, axis=1)
        touch = stance[None, :] & (ground > -p["reach"])
        thrust = p["f_leg"] * touch.sum(axis=1, keepdims=True) / p["n_legs"]
        b = self.D[0, 0]
        m = self.H[0, 0]
        v = v + (thrust - b * v) / m * dt + np.sqrt(2 * self.sigma_T2 * b * dt) / m * rng.standard_normal(v.shape)
        q = q + v * dt
        cache["t"] = t + dt
        cache.setdefault("contacts", []).append(touch.sum(axis=1))
        return q, v, None


@register("many_legged", "3-contact",
          "Hyper-redundant many-legged robot whose stance legs find ground at random.")
def many_legged(n_legs: int = 16, spacing: float = 0.05, omega: float = 2 * np.pi * 2.0,
                wave: float = 4.0, f_leg: float = 10.0, reach: float = 0.01,
                rugosity: float = 0.01, corr: float = 0.02, mass: float = 1.0,
                drag: float = 20.0, sigma_T2: float = 1e-4) -> MechSim:
    return ManyLegged(H=np.array([[mass]]), D=np.array([[drag]]),
                      force=lambda q, p: np.zeros_like(q),
                      J=lambda q: np.zeros((q.shape[0], 1, 1)), sigma_T2=sigma_T2,
                      params=dict(n_legs=n_legs, spacing=spacing, omega=omega, wave=wave,
                                  f_leg=f_leg, reach=reach, rugosity=rugosity, corr=corr),
                      parts={"x": _sel(1, 0), "speed": _sel(1, 1)}, name="many-legged",
                      q0=np.array([0.0]))


def crawl(sim: ManyLegged, batch: int, t_end: float, dt: float = 1e-3, rng=None) -> dict:
    """Run `batch` robots, each on its own terrain; returns mean speeds and contact counts."""
    rng = np.random.default_rng() if rng is None else rng
    cache = sim._ou(dt)
    cache["terrain"] = sim.terrain(batch, rng)
    q = np.zeros((batch, 1))
    v = np.zeros((batch, 1))
    n = int(t_end / dt)
    for _ in range(n):
        q, v, _ = sim.step(q, v, None, rng, dt, cache)
    contacts = np.array(cache["contacts"])
    return dict(speed=q[:, 0] / t_end, contacts=contacts.mean(axis=0))
