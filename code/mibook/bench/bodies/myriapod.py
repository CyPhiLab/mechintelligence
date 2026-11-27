"""Work in progress, not yet registered in mibook.bench: the flat-ground speed still grows
with the number of modules, against the paper.

A many-legged robot on rugose terrain in MuJoCo, after Chong et al. (2023), "Multilegged
matter transport: a framework for locomotion on noisy landscapes", Science 380: 509-515,
and its Supplementary Materials (SM).

Robot (SM Sect. 1.2; Fig. S1-S2). A chain of N modules, each 15 cm long, joined by yaw
joints that carry the lateral body wave. Each module has a contralateral pair of legs on one
fore/aft (swing) joint, so the pair is rigidly connected in yaw. Each leg has an up/down
(lift) joint. The lift is cable driven: the servo can only pull a leg up, and a rubber band
pushes it back down, so a stance leg presses on the ground with the band's force and the
robot's height can vary along the body ("inward compliance"). Each leg has a knee that
bends only from head to tail, held straight by a rubber band ("longitudinal compliance").
Legs lift up to 60 degrees.

Gait (SM Eqs. S1-S3). Contact pattern c_l(tau_c, i) = 1 when mod(tau_c - 2 pi xi (i-1)/N,
2 pi) < 2 pi D, c_r = c_l shifted by pi; leg angle theta_l = Theta_leg cos(tau / (2D)) in
stance and -Theta_leg cos((tau - 2 pi D) / (2 (1-D))) in swing, theta_r shifted by pi; body
wave alpha(tau_b, i) = Theta_body cos(tau_b - 2 pi xi (i-1)/N); optimal coordination
tau_c = tau_b - (xi/N + 1/2) pi. Their choice: D = 0.5, xi = N/6, Theta_leg = Theta_body =
pi/6. With D = 0.5, theta_r = -theta_l, as a rigid pair requires.

Terrain (SM Sect. 1.1). A stepfield of 10 cm square blocks. Heights are drawn from a normal
distribution, N(6, 2) cm in 1 cm increments for Rg = 0.17 and N(6.25, 4) cm in 2.5 cm
increments for Rg = 0.32, truncated to 0-12 cm. Blocks are placed row by row from a 0 cm
block in the corner; each new block is drawn uniformly from the remaining heights that differ
by less than 8 cm from every placed neighbour. Field width 80 cm (Rg = 0.17) and 50 cm
(Rg = 0.32). The field here is made long enough for the whole robot to travel 60 cm on it.

Parameters from other papers by the same group:
- leg length 12 cm (Chong et al. 2023 PNAS 120: e2213698120, Methods; Chong et al. 2022
  Bioinspir. Biomim. 17: 046015, Sect. on the myriapod robot);
- module mass 0.125 kg: their eight-segment myriapod weighs about 1000 g (Chong et al.
  2022); this is an earlier version of the robot, so the value is approximate;
- gait period 6 s (He et al. 2025, IEEE T-RO 41: 4776, Appendix: the same robot at 1/6 Hz);
- friction coefficient 0.6 by default; Chong et al. (2023 PNAS, Fig. 2B) find step length
  unchanged from mu ~ 0.1 to mu ~ 0.6, and their dynamic model (SI) shows why: in a
  dissipation-dominated gait the steady speed does not depend on mu or mass.
Assumed here, with no published value: body width and height, hip height, rubber-band
stiffnesses, servo gains, the split of the leg at the knee. The tests check that the
published trends do not hinge on them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import register

MODULE_LEN = 0.15        # m, SM Sect. 1.2
LEG_LEN = 0.12           # m, Chong et al. 2023 PNAS
MODULE_MASS = 0.125      # kg, Chong et al. 2022 (about 1000 g / 8 segments)
PERIOD = 6.0             # s, He et al. 2025
LIFT_MAX = np.pi / 3     # 60 degrees, SM Sect. 1.2
BLOCK = 0.10             # m
TERRAIN = {              # SM Sect. 1.1: (mean, sd, increment) in cm, field width in m
    0.17: dict(mean=6.0, sd=2.0, step=1.0, width=0.80),
    0.32: dict(mean=6.25, sd=4.0, step=2.5, width=0.50),
}


def _mujoco():
    try:
        import mujoco
        from mujoco import rollout
    except ImportError as e:                       # pragma: no cover
        raise ImportError("the myriapod needs MuJoCo: uv run --extra physics ...") from e
    return mujoco, rollout


# ---------------------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------------------

def stepfield(rg: float, n_cols: int, n_rows: int, rng, limit: float = 8.0) -> np.ndarray:
    """Block heights (cm), shape (n_rows along travel, n_cols across), by the procedure of
    SM Sect. 1.1. When no remaining height fits all placed neighbours (the SM does not say
    what then), the remaining height closest to their mean is used."""
    if rg == 0:
        return np.zeros((n_rows, n_cols))
    p = TERRAIN[rg]
    n = n_rows * n_cols
    h = []
    while len(h) < n:                               # truncated normal, quantized
        x = rng.normal(p["mean"], p["sd"], 2 * n)
        x = p["step"] * np.round(x / p["step"])
        h.extend(x[(x >= 0) & (x <= 12)].tolist())
    pool = list(np.array(h[:n - 1]))
    grid = np.full((n_rows, n_cols), np.nan)
    grid[0, 0] = 0.0
    for r in range(n_rows):
        for c in range(n_cols):
            if r == 0 and c == 0:
                continue
            nb = [grid[rr, cc] for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                  if 0 <= rr < n_rows and 0 <= cc < n_cols and not np.isnan(grid[rr, cc])]
            arr = np.array(pool)
            ok = np.all(np.abs(arr[:, None] - np.array(nb)[None, :]) < limit, axis=1)
            if ok.any():
                k = rng.choice(np.flatnonzero(ok))
            else:
                k = int(np.argmin(np.abs(arr - np.mean(nb))))
            grid[r, c] = pool.pop(k)
    return grid


# ---------------------------------------------------------------------------------------
# Gait
# ---------------------------------------------------------------------------------------

@dataclass
class Gait:
    n: int                                   # number of modules (leg pairs)
    D: float = 0.5
    xi: float | None = None                  # default N/6
    theta_leg: float = np.pi / 6
    theta_body: float = np.pi / 6
    period: float = PERIOD

    def __post_init__(self):
        if self.xi is None:
            self.xi = self.n / 6.0

    def phases(self, t):
        """Contact phase of each module, shape (len(t), N)."""
        tau = 2 * np.pi * np.asarray(t)[:, None] / self.period
        return tau - 2 * np.pi * self.xi * np.arange(self.n)[None, :] / self.n

    def contact_left(self, t):
        return np.mod(self.phases(t), 2 * np.pi) < 2 * np.pi * self.D

    def leg_angle_left(self, t):
        tau = np.mod(self.phases(t), 2 * np.pi)
        D = self.D
        return np.where(tau < 2 * np.pi * D, self.theta_leg * np.cos(tau / (2 * D)),
                        -self.theta_leg * np.cos((tau - 2 * np.pi * D) / (2 * (1 - D))))

    def body_angles(self, t):
        """Yaw angles of the N - 1 body joints (joint i between modules i and i+1)."""
        tau_c = 2 * np.pi * np.asarray(t)[:, None] / self.period
        tau_b = tau_c + (self.xi / self.n + 0.5) * np.pi
        i = np.arange(self.n - 1)[None, :]
        return self.theta_body * np.cos(tau_b - 2 * np.pi * self.xi * i / self.n)


# ---------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------

@dataclass
class Myriapod:
    n: int = 6
    rg: float = 0.0
    mu: float = 0.6
    module_mass: float = MODULE_MASS
    hip_height: float = 0.05         # assumed
    k_band: float = 1.0              # N m/rad, lift rubber band (assumed)
    band_preload: float = 0.4        # rad below the ground-contact angle (assumed)
    k_knee: float = 0.3              # N m/rad, knee rubber band (assumed)
    knee_frac: float = 0.5           # upper-leg share of the leg length (assumed)
    body_sign: float = 1.0
    dt: float = 0.002
    seed: int = 0
    gait: Gait = field(default=None)
    travel: float = 0.6

    def __post_init__(self):
        if self.gait is None:
            self.gait = Gait(self.n)
        rng = np.random.default_rng(self.seed)
        length = self.n * MODULE_LEN + self.travel + 0.6
        width = TERRAIN[self.rg]["width"] if self.rg else 0.8
        self.n_rows = int(np.ceil(length / BLOCK))
        self.n_cols = int(round(width / BLOCK))
        self.heights = stepfield(self.rg, self.n_cols, self.n_rows, rng) / 100.0
        self.x_start = self.n * MODULE_LEN - 0.1          # head x at the start
        mujoco, _ = _mujoco()
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.model.opt.timestep = self.dt

    # -- geometry ----------------------------------------------------------------------
    @property
    def beta(self) -> float:
        """Leg angle below horizontal when the foot touches flat ground."""
        return float(np.arcsin(self.hip_height / LEG_LEN))

    def xml(self) -> str:
        L, n = MODULE_LEN, self.n
        b = self.beta
        lu, ll = self.knee_frac * LEG_LEN, (1 - self.knee_frac) * LEG_LEN
        m_leg = 0.01
        z0 = self.hip_height + float(self.heights.max()) + 0.03
        parts = []
        # Terrain: blocks along x from 0, centred on y = 0.
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                h = self.heights[r, c]
                if h <= 0:
                    continue
                x = (r + 0.5) * BLOCK
                y = (c + 0.5) * BLOCK - 0.5 * self.n_cols * BLOCK
                parts.append(f'<geom type="box" pos="{x:.4f} {y:.4f} {h / 2:.4f}" '
                             f'size="{BLOCK / 2:.4f} {BLOCK / 2:.4f} {h / 2:.4f}" rgba=".75 .72 .65 1"/>')
        terrain = "\n".join(parts)

        def leg(i, s):
            side = "l" if s > 0 else "r"
            d = np.array([0.0, s * np.cos(b), -np.sin(b)])
            pu, pl = lu * d, ll * d
            return f"""
        <body name="hip{i}{side}" pos="0 {s * 0.03:.3f} 0">
          <joint name="lift{i}{side}" type="hinge" axis="{s:.0f} 0 0" range="-0.8 1.2"
                 stiffness="{self.k_band}" springref="{-self.band_preload}" damping="0.02"/>
          <geom type="capsule" fromto="0 0 0 {pu[0]:.4f} {pu[1]:.4f} {pu[2]:.4f}" size="0.006"
                mass="{m_leg / 2}" rgba=".2 .2 .2 1"/>
          <body name="knee{i}{side}" pos="{pu[0]:.4f} {pu[1]:.4f} {pu[2]:.4f}">
            <joint name="knee{i}{side}" type="hinge" axis="0 0 {s:.0f}" range="0 1.4"
                   stiffness="{self.k_knee}" springref="0" damping="0.005"/>
            <geom type="capsule" fromto="0 0 0 {pl[0]:.4f} {pl[1]:.4f} {pl[2]:.4f}" size="0.006"
                  mass="{m_leg / 2}" rgba=".2 .2 .2 1"/>
            <geom name="foot{i}{side}" type="sphere" pos="{pl[0]:.4f} {pl[1]:.4f} {pl[2]:.4f}"
                  size="0.008" mass="0.002" rgba=".9 .4 .1 1"/>
          </body>
        </body>"""

        def module(i):
            inner = module(i + 1) if i + 1 < n else ""
            joint = ('<freejoint name="root"/>' if i == 0 else
                     f'<joint name="yaw{i - 1}" type="hinge" axis="0 0 1" pos="{L / 2} 0 0" '
                     f'range="-1.2 1.2" damping="0.05"/>'
                     f'<joint name="pitch{i - 1}" type="hinge" axis="0 1 0" pos="{L / 2} 0 0" '
                     f'range="-0.6 0.6" damping="0.05"/>')
            pos = f"{self.x_start} 0 {z0}" if i == 0 else f"{-L} 0 0"
            return f"""
      <body name="m{i}" pos="{pos}">
        {joint}
        <geom type="box" size="{0.4 * L} 0.03 0.015" mass="{self.module_mass}" rgba=".16 .47 .84 1"/>
        <body name="swing{i}">
          <joint name="swing{i}" type="hinge" axis="0 0 -1" range="-1 1" damping="0.02"/>
          <geom type="cylinder" size="0.012 0.01" mass="0.005" rgba=".3 .3 .3 1"/>
          {leg(i, +1)}
          {leg(i, -1)}
        </body>
        {inner}
      </body>"""

        acts = []
        for i in range(n - 1):
            acts.append(f'<position name="a_yaw{i}" joint="yaw{i}" kp="20" kv="0.2"/>')
            acts.append(f'<position name="a_pitch{i}" joint="pitch{i}" kp="20" kv="0.2"/>')
        for i in range(n):
            acts.append(f'<position name="a_swing{i}" joint="swing{i}" kp="10" kv="0.1"/>')
            for side in "lr":
                # Cable: pulls the leg up only (non-negative force).
                acts.append(f'<position name="a_lift{i}{side}" joint="lift{i}{side}" kp="8" '
                            f'kv="0.05" forcelimited="true" forcerange="0 5"/>')
        return f"""
<mujoco model="myriapod">
  <option timestep="{self.dt}" integrator="implicitfast"/>
  <default><geom friction="{self.mu} 0.005 0.0001" condim="3" solref="0.004 1"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" rgba=".85 .85 .82 1"/>
    {terrain}
    {module(0)}
  </worldbody>
  <actuator>
    {"".join(acts)}
  </actuator>
</mujoco>"""

    # -- controls ----------------------------------------------------------------------
    def controls(self, t: np.ndarray, settle: float = 1.0) -> np.ndarray:
        """Actuator targets over time t (s). The gait starts after `settle` seconds with all
        legs in stance."""
        g = self.gait
        tg = np.clip(t - settle, 0.0, None)
        on = (t >= settle)[:, None]
        cl = g.contact_left(tg)
        cr = np.mod(g.phases(tg) + np.pi, 2 * np.pi) < 2 * np.pi * g.D
        th = g.leg_angle_left(tg) * on
        al = self.body_sign * g.body_angles(tg) * on
        u = np.zeros((len(t), self.model.nu))
        k = 0
        for i in range(self.n - 1):
            u[:, k] = al[:, i]
            u[:, k + 1] = 0.0
            k += 2
        for i in range(self.n):
            u[:, k] = th[:, i]
            u[:, k + 1] = np.where(on[:, 0] & ~cl[:, i], LIFT_MAX, -1.0)
            u[:, k + 2] = np.where(on[:, 0] & ~cr[:, i], LIFT_MAX, -1.0)
            k += 3
        return u

    # -- simulation --------------------------------------------------------------------
    def run(self, cycles: float, settle: float = 1.0, record_every: int = 0) -> dict:
        """Simulate; return head x (forward) over time and, if asked, body poses."""
        mujoco, _ = _mujoco()
        m = self.model
        d = mujoco.MjData(m)
        n_steps = int(round((settle + cycles * self.gait.period) / self.dt))
        t = np.arange(n_steps) * self.dt
        u = self.controls(t, settle)
        head = np.empty(n_steps)
        frames = []
        bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"m{i}") for i in range(self.n)]
        for k in range(n_steps):
            d.ctrl[:] = u[k]
            mujoco.mj_step(m, d)
            head[k] = d.xpos[bid[0], 0]
            if record_every and k % record_every == 0:
                frames.append(dict(t=t[k], xpos=d.xpos.copy(), xmat=d.xmat.copy(),
                                   geom_xpos=d.geom_xpos.copy(), geom_xmat=d.geom_xmat.copy()))
        i0 = int(round(settle / self.dt))
        return dict(t=t[i0:] - settle, head=head[i0:] - head[i0], frames=frames)

    def transport_time(self, D: float = 0.6, max_cycles: float = 40.0) -> float:
        """T[D]: gait cycles for the head to advance D (SM: time for the head module to move
        a distance D). Returns inf if it does not arrive within max_cycles."""
        r = self.run(max_cycles)
        k = np.flatnonzero(r["head"] >= D)
        return float(r["t"][k[0]] / self.gait.period) if len(k) else np.inf


@register("myriapod", "3-contact",
          "Many-legged robot after Chong et al. 2023 (Science): N 15 cm modules, compliant "
          "legs, open-loop centipede gait, on a stepfield of 10 cm blocks (MuJoCo).")
def myriapod(n: int = 6, rg: float = 0.0, **kw) -> Myriapod:
    return Myriapod(n=n, rg=rg, **kw)
