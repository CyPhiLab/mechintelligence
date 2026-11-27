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
# Hopper (Haeufle et al. 2010; as specified in Ghazi-Zahedi et al. 2016, Sec. 4)
# ---------------------------------------------------------------------------------------

class Hopper:
    """The reduced hopping model of Haeufle et al. (2010), with the three leg actuators of
    Ghazi-Zahedi, Haeufle, Montufar, Schmitt and Ay (2016), Frontiers in Robotics and AI 3:42:

        m y'' = -m g + (F_L if y <= l0 else 0),      m = 80 kg, l0 = 1 m.

    actuator = "musfib": F_L = q F_fib(l_M, l_M'), with activation q' = (u - q)/tau,
        tau = 10 ms, 0.001 <= u <= 1, reflex stimulation u = G F_L(t - delta) + u0,
        delta = 15 ms, G = 2.4/F_max, u0 = 0.027, and the Hill-type force-length-velocity
        relation of their Eq. (6) (F_max = 2.5 kN, l_opt = 0.9 m, w = 0.45, c = 30,
        l'_max = -3.5 m/s, K = 1.5, N = 1.5). In contact l_M = y, l_M' = y'.
    actuator = "muslin": as musfib but F_fib replaced by F_max (1 - mu l_M'), mu = 0.25 s/m,
        G = 0.8/F_max, u0 = 0.19 (their Eq. 8).
    actuator = "dcmot": F_L = gamma k_T I with I' = (u_DC - k_T gamma y' - R I)/L, a PD
        controller u_DC = K_P (y_des - y) + K_D (y_des' - y') limited to +-48 V tracking the
        musfib trajectory, mass m_DC = 0.68 kg (their Eqs. 9-12).
    Each was tuned in the paper to hop periodically with apex 1.070 m. Optional additive
    white force noise of intensity `noise` (N^2 s) gives the body a declared resolution.
    """

    F_MAX, L_OPT, W, C, LDOT_MAX, K, N = 2500.0, 0.9, 0.45, 30.0, -3.5, 1.5, 1.5

    def __init__(self, actuator: str = "musfib", noise: float = 0.0, g: float = 9.81):
        self.actuator = actuator
        self.noise = noise
        self.g = g
        self.l0 = 1.0
        self.m = 0.68 if actuator == "dcmot" else 80.0
        self.name = f"hopper ({actuator})"
        self.params = dict(actuator=actuator, noise=noise)
        if actuator == "musfib":
            self.G, self.u0 = 2.4 / self.F_MAX, 0.027
        elif actuator == "muslin":
            self.G, self.u0 = 0.8 / self.F_MAX, 0.19
        self.tau, self.delta = 0.010, 0.015
        self.kT, self.gamma, self.R, self.Lw = 0.126, 100.0, 7.19, 1.6e-3
        self.KP, self.KD = 5000.0, 500.0

    def f_fib(self, lM, dlM):
        fl = np.exp(-self.C * np.abs((lM - self.L_OPT) / (self.L_OPT * self.W)) ** 3)
        lm = self.LDOT_MAX
        conc = (lm + dlM) / (lm - self.K * dlM)
        ecc = self.N + (self.N - 1) * (lm + dlM) / (-7.56 * self.K * dlM - lm)
        fv = np.where(dlM > 0, conc, ecc)
        return self.F_MAX * fl * np.maximum(fv, 0.0)

    def simulate(self, T: float = 8.0, dt: float = 1e-4, y0: float = 1.070, batch: int = 1,
                 rng=None, y_des=None):
        """Fixed-step integration (semi-implicit Euler, dt = 0.1 ms). Returns time, height,
        velocity, leg force and actuator command at every step. `y_des` = (t, y, y') arrays
        for the DC motor's PD target."""
        rng = np.random.default_rng() if rng is None else rng
        n = int(T / dt)
        y = np.full(batch, y0)
        v = np.zeros(batch)
        q = np.full(batch, getattr(self, "u0", 0.0))
        I = np.zeros(batch)
        lag = int(round(self.delta / dt))
        Fhist = np.zeros((lag + 1, batch))
        out = {k: np.empty((n, batch)) for k in ("y", "v", "F", "a")}
        for k in range(n):
            contact = y <= self.l0
            if self.actuator in ("musfib", "muslin"):
                F_del = Fhist[k % (lag + 1)]
                u = np.clip(self.G * F_del + self.u0, 0.001, 1.0)
                q = q + dt * (u - q) / self.tau
                lM = np.where(contact, y, self.l0)
                dlM = np.where(contact, v, 0.0)
                if self.actuator == "musfib":
                    F = q * self.f_fib(lM, dlM)
                else:
                    F = q * self.F_MAX * np.maximum(1 - 0.25 * dlM, 0.0)
                F = np.where(contact, F, 0.0)
                a_cmd = u
            else:
                td, yd, vd = y_des
                yd_k = np.interp(k * dt, td, yd)
                vd_k = np.interp(k * dt, td, vd)
                u = np.clip(self.KP * (yd_k - y) + self.KD * (vd_k - v), -48.0, 48.0)
                I = I + dt * (u - self.kT * self.gamma * v - self.R * I) / self.Lw
                F = np.where(contact, self.gamma * self.kT * I, 0.0)
                a_cmd = u
            Fhist[k % (lag + 1)] = F
            acc = -self.g + F / self.m
            if self.noise > 0:
                acc = acc + np.sqrt(self.noise / dt) * rng.standard_normal(batch) / self.m
            v = v + dt * acc
            y = y + dt * v
            out["y"][k], out["v"][k], out["F"][k], out["a"][k] = y, v, F, a_cmd
        out["t"] = np.arange(n) * dt
        return out


def apexes(y: np.ndarray, t: np.ndarray, t_min: float = 2.0) -> np.ndarray:
    """Heights of the local maxima of y(t) after t_min (one trajectory)."""
    m = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (t[1:-1] > t_min) & (y[1:-1] > 1.0)
    return y[1:-1][m]


@register("hopper", "3-contact",
          "Reduced hopper of Haeufle et al. (2010) with muscle-fibre, linear-muscle or DC-motor "
          "legs (Ghazi-Zahedi et al. 2016).")
def hopper(actuator: str = "musfib", noise: float = 0.0) -> Hopper:
    return Hopper(actuator=actuator, noise=noise)


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
# Many-legged robot (side view): rigid body, prescribed leg cycles, frictional contact
# ---------------------------------------------------------------------------------------

class LeggedRobot:
    """A many-legged robot seen from the side, in the spirit of the multilegged robophysical
    models of Chong et al. (2023) and He et al. (2025).

    Body: a rigid bar of length `body_len` and mass `mass`, with forward position x, height z
    and pitch th, under gravity, and pitch inertia m L^2 / 12.
    Legs: n_legs hips spread along the bar. Each foot follows a prescribed cycle relative
    to its hip, in the body frame, with phase p = omega t - psi_i. In stance (half the
    cycle) the foot moves straight back from +A to -A at constant speed, at depth l below
    the hip; in swing it returns forward along a raised arc of height `lift`. A = l sin(phi0).
    Successive legs lag by psi_i = 2 pi waves i / n_legs, a travelling wave from head to tail.
    Contact: each leg is a compliant spring k_leg in series with the ground contact k_c, with
    its nominal foot point `reach` below the ground plane, so a stance foot presses into
    hollows up to about that depth. A foot below the terrain h(x) feels a normal force
    k_eff pen - b_c pen' (never pulling), k_eff = (1/k_c + 1/k_leg)^-1, with b_c a fraction
    zeta_c of critical damping for the body's share per stance leg, and a Coulomb
    friction force -mu N tanh(v_slip / v_eps) along x. Legs are massless, so every contact
    force acts on the body at the foot. Thrust comes only from friction. On flat ground all
    stance feet move back together and the body moves at the kinematic speed 4 A f.
    Belly: points along the underside of the body (at the hips) touch the terrain with the
    ground stiffness and the same friction, so a robot that loses its footing drags on its
    belly instead of sinking through the ground.
    Terrain: a random field of standard deviation `rugosity` and correlation length `corr`,
    one realization per batch member. Thermal noise enters through a small air drag.
    """

    def __init__(self, n_legs=12, body_len=0.5, mass=1.0, leg_len=0.08, phi0=0.4, freq=2.0,
                 waves=1.0, lift=0.03, k_c=3e4, k_leg=150.0, reach=0.02, zeta_c=0.5, mu=0.6,
                 v_eps=0.005, rugosity=0.0, corr=0.03, b_air=0.05, sigma_T2=1e-6, g=G):
        k_eff = 1.0 / (1.0 / k_c + 1.0 / k_leg)
        # Contact damping as a fraction zeta_c of critical, for the body's share per stance leg.
        b_c = 2.0 * zeta_c * np.sqrt(k_eff * mass / (n_legs / 2.0))
        self.params = dict(n_legs=n_legs, body_len=body_len, mass=mass, leg_len=leg_len,
                           phi0=phi0, freq=freq, waves=waves, lift=lift, k_c=k_c, k_leg=k_leg,
                           reach=reach, zeta_c=zeta_c, b_c=b_c, mu=mu, v_eps=v_eps,
                           rugosity=rugosity, corr=corr, b_air=b_air, sigma_T2=sigma_T2, g=g)
        self.n = 3
        self.name = "many-legged robot"
        self.hips = np.linspace(-body_len / 2, body_len / 2, n_legs)
        self.psi = 2 * np.pi * waves * np.arange(n_legs) / n_legs
        self.inertia = mass * body_len ** 2 / 12.0

    def terrain(self, batch, rng, length=12.0, res=0.002):
        p = self.params
        n = int(length / res)
        if p["rugosity"] <= 0:
            return np.zeros((batch, n)), res
        k = max(1, int(p["corr"] / res))
        ker = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2)
        ker /= np.sqrt((ker ** 2).sum())
        w = rng.standard_normal((batch, n + len(ker)))
        h = np.stack([np.convolve(row, ker, mode="same")[:n] for row in w]) * p["rugosity"]
        return h, res

    def feet(self, q, v, t):
        """World positions and velocities of every foot, the lever arms from the body's
        centre, and the stance mask. q, v: (batch, 3) = (x, z, th)."""
        p = self.params
        om = 2 * np.pi * p["freq"]
        ph = np.mod(om * t - self.psi, 2 * np.pi)                 # (n_legs,)
        A = p["leg_len"] * np.sin(p["phi0"])
        stance = ph < np.pi
        sp = np.where(stance, ph / np.pi, (ph - np.pi) / np.pi)   # progress in the phase
        x = np.where(stance, A * (1 - 2 * sp), -A * np.cos(np.pi * sp))
        dx = np.where(stance, -2 * A * om / np.pi, A * om * np.sin(np.pi * sp))
        z = np.where(stance, 0.0, p["lift"] * np.sin(np.pi * sp)) - p["leg_len"] - p["reach"]
        dz = np.where(stance, 0.0, p["lift"] * om * np.cos(np.pi * sp))
        rel = np.stack([self.hips + x, z], -1)                     # body frame
        drel = np.stack([dx, dz], -1)
        swing = ~stance
        c, s = np.cos(q[:, 2]), np.sin(q[:, 2])
        R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)            # (batch, 2, 2)
        r = np.einsum("bij,nj->bni", R, rel)                                       # lever arms
        pos = q[:, None, :2] + r
        w = v[:, 2][:, None]
        vel = v[:, None, :2] + np.stack([-w * r[..., 1], w * r[..., 0]], -1) \
            + np.einsum("bij,nj->bni", R, drel)
        return pos, vel, r, ~swing

    def _ground(self, terrain, x):
        h, res = terrain
        idx = np.clip((x / res).astype(int) + h.shape[1] // 2, 0, h.shape[1] - 1)
        return np.take_along_axis(h, idx, axis=1)

    def _contact(self, pen, vz, vx, k, b):
        p = self.params
        N = np.where(pen > 0, np.maximum(k * pen - b * vz, 0.0), 0.0)
        return N, -p["mu"] * N * np.tanh(vx / p["v_eps"])

    def forces(self, q, v, t, terrain):
        p = self.params
        pos, vel, r, stance = self.feet(q, v, t)
        pen = self._ground(terrain, pos[..., 0]) - pos[..., 1]
        k_eff = 1.0 / (1.0 / p["k_c"] + 1.0 / p["k_leg"])
        N, Ft = self._contact(pen, vel[..., 1], vel[..., 0], k_eff, p["b_c"])
        # Belly points at the hips.
        c, s = np.cos(q[:, 2]), np.sin(q[:, 2])
        rb = np.stack([self.hips[None, :] * c[:, None], self.hips[None, :] * s[:, None]], -1)
        pb = q[:, None, :2] + rb
        w = v[:, 2][:, None]
        vb = v[:, None, :2] + np.stack([-w * rb[..., 1], w * rb[..., 0]], -1)
        penb = self._ground(terrain, pb[..., 0]) - pb[..., 1]
        bb = 2.0 * p["zeta_c"] * np.sqrt(p["k_c"] * p["mass"] / p["n_legs"])
        Nb, Fb = self._contact(penb, vb[..., 1], vb[..., 0], p["k_c"], bb)
        Fx = Ft.sum(1) + Fb.sum(1)
        Fz = N.sum(1) + Nb.sum(1) - p["mass"] * p["g"]
        tau = (r[..., 0] * N - r[..., 1] * Ft).sum(1) + (rb[..., 0] * Nb - rb[..., 1] * Fb).sum(1)
        return np.stack([Fx, Fz, tau], 1), dict(N=N, slip=vel[..., 0], stance=stance,
                                               touch=N > 0, pos=pos, belly=Nb.sum(1))

    def step(self, q, v, rng, dt, t, terrain):
        p = self.params
        F, info = self.forces(q, v, t, terrain)
        Mi = np.array([1 / p["mass"], 1 / p["mass"], 1 / self.inertia])
        drag = p["b_air"] * v
        noise = np.sqrt(2 * p["sigma_T2"] * p["b_air"] * dt) * rng.standard_normal(v.shape)
        v = v + ((F - drag) * dt + noise) * Mi
        q = q + v * dt
        return q, v, info

    def start(self, batch, terrain):
        p = self.params
        q = np.zeros((batch, 3))
        k_eff = 1.0 / (1.0 / p["k_c"] + 1.0 / p["k_leg"])
        q[:, 1] = p["leg_len"] + p["reach"] - p["mass"] * p["g"] / (k_eff * p["n_legs"] / 2)
        h, res = terrain
        q[:, 1] += h[:, h.shape[1] // 2]
        return q, np.zeros((batch, 3))


@register("many_legged", "3-contact",
          "Many-legged robot in side view: rigid body, prescribed leg cycles, compliant legs, "
          "frictional contact on rough terrain.")
def many_legged(n_legs: int = 12, rugosity: float = 0.0, seg_len: float = 0.045,
                m_leg: float = 1.0 / 12, **kw) -> LeggedRobot:
    """Built from identical modules, as the robophysical models are: each leg comes with a
    body segment of length seg_len and mass m_leg, so length and mass grow with n_legs and
    every leg carries the same share of the weight."""
    kw.setdefault("body_len", seg_len * n_legs)
    kw.setdefault("mass", m_leg * n_legs)
    return LeggedRobot(n_legs=n_legs, rugosity=rugosity, **kw)


def crawl(robot: LeggedRobot, batch: int, t_end: float, dt: float = 2e-4, rng=None,
          record_every: int = 0) -> dict:
    """Run `batch` robots, each on its own terrain. Returns the mean speed after the first
    cycle, the fraction of stance legs in contact, the mean slip speed of contacting feet,
    and (if asked) frames of body state and foot positions."""
    rng = np.random.default_rng() if rng is None else rng
    terr = robot.terrain(batch, rng)
    q, v = robot.start(batch, terr)
    n = int(t_end / dt)
    t0 = int(1.0 / robot.params["freq"] / dt)
    x0 = None
    touch_frac, slip, frames = [], [], []
    for k in range(n):
        t = k * dt
        q, v, info = robot.step(q, v, rng, dt, t, terr)
        if k == t0:
            x0 = q[:, 0].copy()
        if k > t0:
            st = info["stance"][None, :]
            touch_frac.append((info["touch"] & st).sum(1) / max(st.sum(), 1))
            m = info["touch"]
            slip.append(np.where(m, np.abs(info["slip"]), np.nan))
        if record_every and k % record_every == 0:
            frames.append(dict(q=q.copy(), pos=info["pos"].copy(), touch=info["touch"].copy(),
                               stance=info["stance"].copy(), t=t))
    T = (n - 1 - t0) * dt
    return dict(speed=(q[:, 0] - x0) / T, contact=np.mean(touch_frac, axis=0),
                slip=np.nanmean(np.concatenate(slip, axis=1), axis=1), frames=frames,
                terrain=terr, z=q[:, 1], pitch=q[:, 2])


def kinematic_speed(robot: LeggedRobot) -> float:
    """No-slip speed on flat ground: stance feet move back at 4 A f with A = l sin(phi0)."""
    p = robot.params
    return 4 * p["leg_len"] * np.sin(p["phi0"]) * p["freq"]
