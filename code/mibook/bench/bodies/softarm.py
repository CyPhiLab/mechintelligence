"""A soft arm as a chain of rigid links with elastic joints, simulated with MuJoCo (T12).

Model. The arm is an Euler-Bernoulli beam lumped into rigid links joined by rotational
springs: the pseudo-rigid-body discretization of compliant beams (Howell 2001), here with
one joint at the middle of each of n equal segments and stiffness k_j = E I(s_j) / ds, so
that the bending energy of a segment, k theta^2 / 2 with theta = kappa ds, equals
E I kappa^2 ds / 2. With joints at segment midpoints the static tip deflection under a tip
load is F L^3 / (3 E I) (1 - 1/(4 n^2)), so 20 links are already within 0.1 percent.
Dynamics, contact and batching are MuJoCo's (Todorov, Erez and Tassa 2012).

The arm is anchored to the silicone arm of Nakajima et al. (2014, 2015): Ecoflex 00-30,
44.7 cm long, a physical reservoir under water. The radii, Young's modulus and damping are
assumptions (flagged below). Water is represented by neutral buoyancy (no gravity).

Damping is material (Kelvin-Voigt) damping, b_j = tau_d k_j at each joint, and the thermal
forces are generalized joint forces with variance 2 sigma_T2 b_j / dt, the fluctuation
partner of that damping, so the arm settles to the Gibbs law at temperature sigma_T2.

Inputs (all MuJoCo actuators, so a batch runs in MuJoCo's threaded rollout):
- `tip_force`: a force at the tip in the world x and z directions (environment channel);
- `curl`: a uniform bending couple along the arm, as a tendon-driven arm would apply
  (actuation channel);
- thermal forces on every joint.
Optionally a fixed cylinder to grasp; the links touch the cylinder but not each other.

Checks against textbook results (`check_cantilever`, `check_frequency`): the static tip
deflection under a small tip load, F L^3 / (3 E I), and the first bending frequency,
1.875^2 sqrt(E I / (rho A)) / (2 pi L^2), for a uniform rod.
"""
from __future__ import annotations

import numpy as np

from .. import register

# Assumptions, not taken from the papers: Ecoflex 00-30's modulus is of order 1e5 Pa; the
# cone's radii give a slender arm of the published length; the damping time is chosen to
# damp the first mode lightly.
LENGTH = 0.447
R_BASE, R_TIP = 0.015, 0.005
E_SILICONE = 1.0e5
RHO = 1070.0
TAU_D = 0.01


def _mujoco():
    try:
        import mujoco
        from mujoco import rollout
    except ImportError as err:                       # the physics extra is optional
        raise ImportError("the soft arm needs MuJoCo: uv sync --extra physics") from err
    return mujoco, rollout


class SoftArm:
    def __init__(self, n_links: int = 20, length: float = LENGTH, r_base: float = R_BASE,
                 r_tip: float = R_TIP, E: float = E_SILICONE, rho: float = RHO,
                 tau_d: float = TAU_D, sigma_T2: float = 1e-7, dt: float = 5e-4,
                 cylinder: dict | None = None):
        self.params = dict(n_links=n_links, length=length, r_base=r_base, r_tip=r_tip, E=E,
                           rho=rho, tau_d=tau_d, sigma_T2=sigma_T2, dt=dt, cylinder=cylinder)
        self.name = "soft arm (link chain)"
        n, L = n_links, length
        self.ds = L / n
        self.s_joint = (np.arange(n) + 0.5) * self.ds
        radius = lambda s: r_base + (r_tip - r_base) * s / L
        self.k = E * np.pi * radius(self.s_joint) ** 4 / 4 / self.ds
        self.b = tau_d * self.k
        mujoco, _ = _mujoco()
        self.model = mujoco.MjModel.from_xml_string(self._xml(radius))
        self.n = n
        # Joint springs are integrated explicitly (MuJoCo's implicit schemes treat damping,
        # not stiffness, implicitly), so the stiffest mode must be resolved by the step.
        d = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, d)
        M = np.zeros((n, n))
        mujoco.mj_fullM(self.model, d, M)
        self.M0 = M
        self.omega_max = float(np.sqrt(np.max(np.linalg.eigvals(np.linalg.solve(M, np.diag(self.k))).real)))
        if dt * self.omega_max > 1.0:
            raise ValueError(f"time step {dt} too large for the stiffest mode "
                             f"({self.omega_max:.0f} rad/s); use dt < {1.0 / self.omega_max:.2e}")

    def _xml(self, radius) -> str:
        p, n, ds = self.params, self.params["n_links"], self.ds
        L = p["length"]
        body = ""
        for j in range(n):
            lo = self.s_joint[j]
            hi = self.s_joint[j + 1] if j + 1 < n else L
            r = radius(0.5 * (lo + hi))
            pos = ds / 2 if j == 0 else ds
            ln = hi - lo
            # The link's mass and inertia are those of its cylindrical slice of the arm; the
            # capsule is only its contact shape (its end caps would add mass a beam lacks).
            mass = p["rho"] * np.pi * r ** 2 * ln
            Ixx = mass * (3 * r ** 2 + ln ** 2) / 12
            body += (f'<body name="l{j}" pos="0 0 {pos:.9g}">'
                     f'<inertial pos="0 0 {ln / 2:.9g}" mass="{mass:.9g}" '
                     f'diaginertia="{Ixx:.9g} {Ixx:.9g} {mass * r ** 2 / 2:.9g}"/>'
                     f'<joint name="j{j}" type="hinge" axis="0 1 0" stiffness="{self.k[j]:.9g}" '
                     f'damping="{self.b[j]:.9g}"/>'
                     f'<geom type="capsule" fromto="0 0 0 0 0 {ln:.9g}" size="{r:.9g}" '
                     f'contype="1" conaffinity="0"/>')
        body += f'<site name="tip" pos="0 0 {L - self.s_joint[-1]:.9g}"/>' + "</body>" * n
        cyl = ""
        if p["cylinder"] is not None:
            c = p["cylinder"]
            cyl = (f'<geom type="cylinder" pos="{c["x"]} 0 {c["z"]}" zaxis="0 1 0" '
                   f'size="{c["r"]} 0.1" contype="0" conaffinity="1"/>')
        acts = "".join(f'<motor joint="j{j}"/>' for j in range(n))
        acts += ('<general site="tip" refsite="base" gear="1 0 0 0 0 0"/>'
                 '<general site="tip" refsite="base" gear="0 0 1 0 0 0"/>')
        return (f'<mujoco><option timestep="{p["dt"]}" gravity="0 0 0" '
                f'integrator="implicitfast"/><worldbody><site name="base"/>{cyl}{body}'
                f'</worldbody><actuator>{acts}</actuator></mujoco>')

    def controls(self, steps: int, batch: int, tip_force=None, curl=None, rng=None):
        """Actuator inputs (batch, steps, n + 2): thermal joint forces, the curl couple
        spread over the joints, and the tip force. `tip_force` is an array (steps, batch, 2)
        or a callable t -> (batch, 2); `curl` is a callable t -> (batch,) couples in N m."""
        p, n = self.params, self.n
        rng = np.random.default_rng() if rng is None else rng
        t = np.arange(steps) * p["dt"]
        u = np.zeros((batch, steps, n + 2))
        if p["sigma_T2"] > 0:
            u[:, :, :n] = np.sqrt(2 * p["sigma_T2"] * self.b / p["dt"]) * rng.standard_normal(
                (batch, steps, n))
        if curl is not None:
            # A couple c on every link is a generalized force c (n - j) on joint j.
            c = np.array([np.broadcast_to(curl(tk), (batch,)) for tk in t]).T / n
            u[:, :, :n] += c[:, :, None] * (n - np.arange(n))[None, None, :]
        if tip_force is not None:
            F = (np.asarray(tip_force) if not callable(tip_force)
                 else np.array([np.broadcast_to(tip_force(tk), (batch, 2)) for tk in t]))
            u[:, :, n:] = np.transpose(F, (1, 0, 2))
        return u

    def simulate(self, T: float, batch: int = 1, tip_force=None, curl=None, rng=None,
                 q0=None, v0=None, nthread: int = 4) -> dict:
        """Run `batch` arms for T seconds from rest, or from joint angles q0 and velocities
        v0 (one vector, or one row per arm). Returns joint
        angles q (batch, steps, n), velocities v, and the time step."""
        mujoco, rollout = _mujoco()
        m, p = self.model, self.params
        steps = int(round(T / p["dt"]))
        u = self.controls(steps, batch, tip_force, curl, rng)
        d = mujoco.MjData(m)
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        s0 = np.zeros((batch, mujoco.mj_stateSize(m, spec)))
        q0 = np.zeros((batch, m.nq)) if q0 is None else np.broadcast_to(q0, (batch, m.nq))
        v0 = np.zeros((batch, m.nv)) if v0 is None else np.broadcast_to(v0, (batch, m.nv))
        for b in range(batch):
            d.qpos[:], d.qvel[:] = q0[b], v0[b]
            mujoco.mj_getState(m, d, s0[b], spec)
        datas = [mujoco.MjData(m) for _ in range(nthread)]
        with rollout.Rollout(nthread=nthread) as r:
            state, _ = r.rollout(m, datas, s0, u)
        q = state[:, :, 1:1 + m.nq]
        v = state[:, :, 1 + m.nq:1 + m.nq + m.nv]
        return dict(q=q, v=v, dt=p["dt"], u=u)

    def gibbs_start(self, batch: int, rng=None):
        """Joint angles and velocities drawn from the Gibbs law at rest (straight arm):
        q ~ N(0, sigma_T2 K^-1), v ~ N(0, sigma_T2 M^-1)."""
        rng = np.random.default_rng() if rng is None else rng
        s2 = self.params["sigma_T2"]
        q = rng.standard_normal((batch, self.n)) * np.sqrt(s2 / self.k)
        v = rng.multivariate_normal(np.zeros(self.n), s2 * np.linalg.inv(self.M0), batch)
        return q, v

    def centreline(self, q: np.ndarray) -> np.ndarray:
        """Points along the arm (..., n + 2, 2) as (x, z): base, joints, tip."""
        n, ds = self.n, self.ds
        phi = np.cumsum(q, axis=-1)
        seg = np.full(n, ds)
        seg[-1] = ds / 2
        steps = seg * np.stack([np.sin(phi), np.cos(phi)], axis=-1).swapaxes(-1, -2)
        pts = np.cumsum(steps, axis=-1)
        first = np.zeros(q.shape[:-1] + (2, 1))
        first[..., 1, 0] = ds / 2
        pts = np.concatenate([np.zeros_like(first), first, first + pts], axis=-1)
        return pts.swapaxes(-1, -2)


@register("soft_arm", "2-nonlinear",
          "Soft arm as a chain of rigid links with elastic joints (MuJoCo), anchored to the "
          "silicone arm of Nakajima et al.; tip force, curl couple and an optional object "
          "to grasp.")
def soft_arm(n_links: int = 20, grasp: bool = False, **kw) -> SoftArm:
    cyl = dict(x=-0.06, z=0.28, r=0.035) if grasp else None
    return SoftArm(n_links=n_links, cylinder=cyl, **kw)


CHECK_ROD = dict(length=0.2, r_base=0.01, r_tip=0.01, E=1.0e6, dt=1e-4)   # a stiffer uniform rod


def check_cantilever(F: float = 1e-3, n_links: int = 20, T: float = 3.0) -> dict:
    """Uniform rod, small static tip load, heavy damping: tip deflection against
    F L^3 / (3 E I)."""
    arm = SoftArm(n_links=n_links, tau_d=0.05, sigma_T2=0.0, **CHECK_ROD)
    steps = int(round(T / arm.params["dt"]))
    out = arm.simulate(T, 1, tip_force=np.tile([F, 0.0], (steps, 1, 1)))
    tip = arm.centreline(out["q"][0, -1])[-1]
    p = arm.params
    I = np.pi * p["r_base"] ** 4 / 4
    return dict(deflection=float(tip[0]), predicted=float(F * p["length"] ** 3 / (3 * p["E"] * I)))


def check_frequency(n_links: int = 20, T: float = 6.0) -> dict:
    """Uniform rod released from a short tip push, light damping: first bending frequency
    from the lowest peak of the tip's spectrum, against the Euler-Bernoulli value."""
    arm = SoftArm(n_links=n_links, tau_d=2e-4, sigma_T2=0.0, **CHECK_ROD)
    dt = arm.params["dt"]
    out = arm.simulate(T, 1, tip_force=lambda t: np.array([[1e-3 if t < 0.02 else 0.0, 0.0]]))
    x = arm.centreline(out["q"][0])[:, -1, 0]
    xs = (x - x.mean()) * np.hanning(len(x))
    nfft = 16 * len(xs)
    spec = np.abs(np.fft.rfft(xs, nfft))
    freqs = np.fft.rfftfreq(nfft, dt)
    band = (freqs > 0.5) & (freqs < 5.0)
    p = arm.params
    A, I = np.pi * p["r_base"] ** 2, np.pi * p["r_base"] ** 4 / 4
    f_pred = 1.875 ** 2 * np.sqrt(p["E"] * I / (p["rho"] * A)) / (2 * np.pi * p["length"] ** 2)
    return dict(measured=float(freqs[band][np.argmax(spec[band])]), predicted=float(f_pred))


def check_grasp(curl: float = -0.04, T: float = 3.0) -> dict:
    """The default grasp: a curling couple ramps up over 1.5 s and the arm wraps the
    cylinder. Returns, for the distal third of the arm, the smallest and largest distance
    from the cylinder's axis in units of its radius (wrapped: close to 1, never well below)."""
    arm = soft_arm(grasp=True, sigma_T2=0.0)
    out = arm.simulate(T, 1, curl=lambda t: curl * min(1.0, t / 1.5))
    c = arm.params["cylinder"]
    pts = arm.centreline(out["q"][0, -1])
    d = np.hypot(pts[:, 0] - c["x"], pts[:, 1] - c["z"]) / c["r"]
    distal = d[-(arm.n // 3):]
    return dict(min=float(distal.min()), max=float(distal.max()))
