"""A soft arm as a Cosserat rod, simulated with PyElastica (thread T12).

The rod model is the Cosserat-rod formulation implemented by PyElastica (Gazzola, Dudte,
McCormick and Mahadevan 2018), the standard model for soft slender arms. The arm is anchored
to the silicone arm of Nakajima et al. (2014, 2015): Ecoflex 00-30, 44.7 cm long, used as a
physical reservoir underwater. The radii and Young's modulus here are assumptions (flagged
below); water is represented by neutral buoyancy (no gravity) and linear drag.

Environment channels: a force at the tip (`tip_force`, a callable t -> 3-vector per arm),
and optionally a rigid cylinder to grasp. Actuation: a uniform curling torque
(`curl`, a callable t -> torque in N m). Noise: random nodal forces matched to the drag by
fluctuation-dissipation at a declared temperature sigma_T2.

Many arms are simulated in one PyElastica system, which vectorizes over them, so a batch
costs little more than one arm.

Checks against textbook results (`check_cantilever`, `check_frequency`): the tip deflection
under a small static tip load, F L^3 / (3 E I), and the first bending frequency,
1.875^2 sqrt(E I / (rho A)) / (2 pi L^2), for a uniform rod.
"""
from __future__ import annotations

import numpy as np

from .. import register

# Assumptions, not taken from the papers: Ecoflex 00-30's modulus is of order 1e5 Pa; the
# cone's radii are chosen to give a slender arm of the published length.
LENGTH = 0.447
R_BASE, R_TIP = 0.015, 0.005
E_SILICONE = 1.0e5
RHO = 1070.0


def _elastica():
    try:
        import elastica as ea
    except ImportError as err:                       # the physics extra is optional
        raise ImportError("the soft arm needs PyElastica: uv sync --extra physics") from err
    return ea


class SoftArm:
    def __init__(self, n_elem: int = 30, length: float = LENGTH, r_base: float = R_BASE,
                 r_tip: float = R_TIP, E: float = E_SILICONE, rho: float = RHO,
                 drag: float = 2.0, sigma_T2: float = 1e-7, cylinder: dict | None = None):
        self.params = dict(n_elem=n_elem, length=length, r_base=r_base, r_tip=r_tip, E=E,
                           rho=rho, drag=drag, sigma_T2=sigma_T2, cylinder=cylinder)
        self.name = "soft arm (Cosserat rod)"

    def build(self, batch: int, dt: float, tip_force=None, curl=None, rng=None):
        ea = _elastica()
        p = self.params
        rng = np.random.default_rng() if rng is None else rng

        class Sim(ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping,
                  ea.CallBacks, ea.Contact):
            pass

        sim = Sim()
        n = p["n_elem"]
        radius = np.linspace(p["r_base"], p["r_tip"], n)
        rods = []
        for b in range(batch):
            rod = ea.CosseratRod.straight_rod(
                n, np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]),
                p["length"], radius, p["rho"], youngs_modulus=p["E"],
                shear_modulus=p["E"] / (2 * 1.5))
            sim.append(rod)
            sim.constrain(rod).using(ea.OneEndFixedBC, constrained_position_idx=(0,),
                                     constrained_director_idx=(0,))
            # uniform_damping_constant is a rate (1/s): force -drag m v on every node, which
            # is what the thermal forces below are matched to.
            sim.dampen(rod).using(ea.AnalyticalLinearDamper,
                                  uniform_damping_constant=p["drag"], time_step=dt)
            if p["sigma_T2"] > 0:
                sim.add_forcing_to(rod).using(ThermalForces, drag=p["drag"],
                                              sigma_T2=p["sigma_T2"], dt=dt,
                                              seed=int(rng.integers(2 ** 31)))
            if tip_force is not None:
                sim.add_forcing_to(rod).using(TipForce, fn=tip_force, index=b)
            if curl is not None:
                sim.add_forcing_to(rod).using(CurlTorque, fn=curl, index=b)
            rods.append(rod)
        if p["cylinder"] is not None:
            c = p["cylinder"]
            cyl = ea.Cylinder(start=np.array([c["x"], -0.1, c["z"]]),
                              direction=np.array([0.0, 1.0, 0.0]),
                              normal=np.array([1.0, 0.0, 0.0]), base_length=0.2,
                              base_radius=c["r"], density=1e4)
            sim.append(cyl)
            sim.constrain(cyl).using(ea.FixedConstraint, constrained_position_idx=(0,),
                                     constrained_director_idx=(0,))
            for rod in rods:
                sim.detect_contact_between(rod, cyl).using(ea.RodCylinderContact, k=50.0,
                                                           nu=0.1)
        sim.finalize()
        return sim, rods

    def simulate(self, T: float, batch: int = 1, dt: float | None = None, tip_force=None,
                 curl=None, rng=None, record_every: int = 0):
        """Run `batch` arms for T seconds. Returns final node positions (batch, 3, n+1) and,
        if asked, recorded positions every `record_every` steps."""
        ea = _elastica()
        p = self.params
        dt = 0.02 * p["length"] / p["n_elem"] * np.sqrt(p["rho"] / p["E"]) if dt is None else dt
        sim, rods = self.build(batch, dt, tip_force, curl, rng)
        stepper = ea.PositionVerlet()
        n_steps = int(T / dt)
        frames = []
        t = 0.0
        for k in range(n_steps):
            t = stepper.step(sim, t, dt)                 # PyElastica >= 1.0 interface
            if record_every and k % record_every == 0:
                frames.append(np.stack([r.position_collection.copy() for r in rods]))
        pos = np.stack([r.position_collection.copy() for r in rods])
        return dict(pos=pos, frames=frames, dt=dt, t=t)


try:                                                    # the physics extra is optional
    from elastica import NoForces as _NoForces
except ImportError:                                     # pragma: no cover
    _NoForces = object


class ThermalForces(_NoForces):
    """Random nodal forces with variance 2 sigma_T2 m drag / dt per component: the
    fluctuation partner of PyElastica's linear damper with constant `drag` (1/s)."""

    def __init__(self, drag, sigma_T2, dt, seed):
        super().__init__()
        self.drag, self.s2, self.dt = drag, sigma_T2, dt
        self.rng = np.random.default_rng(seed)

    def apply_forces(self, system, time=0.0):
        amp = np.sqrt(2.0 * self.s2 * self.drag * system.mass / self.dt)
        system.external_forces += amp[None, :] * self.rng.standard_normal(
            system.external_forces.shape)


class TipForce(_NoForces):
    """A force at the tip node: fn(t) returns an array (batch, 3); row `index` is used."""

    def __init__(self, fn, index):
        super().__init__()
        self.fn, self.index = fn, index

    def apply_forces(self, system, time=0.0):
        system.external_forces[:, -1] += np.asarray(self.fn(time))[self.index]


class CurlTorque(_NoForces):
    """A uniform bending couple along the arm, applied about each element's second director
    (the y axis for the straight arm), spread evenly over the elements."""

    def __init__(self, fn, index):
        super().__init__()
        self.fn, self.index = fn, index

    def apply_torques(self, system, time=0.0):
        tau = float(np.asarray(self.fn(time))[self.index])
        system.external_torques[1, :] += tau / system.n_elems


@register("soft_arm", "2-nonlinear",
          "Soft arm as a Cosserat rod (PyElastica), anchored to the silicone arm of "
          "Nakajima et al.; tip force and an optional object to grasp.")
def soft_arm(n_elem: int = 30, grasp: bool = False, **kw) -> SoftArm:
    cyl = dict(x=-0.06, z=0.28, r=0.035) if grasp else None
    return SoftArm(n_elem=n_elem, cylinder=cyl, **kw)


CHECK_ROD = dict(length=0.2, r_base=0.01, r_tip=0.01, E=1.0e6)   # a stiffer uniform rod (about 2 Hz)


def check_cantilever(F: float = 1e-3, n_elem: int = 30, T: float = 3.0, drag: float = 10.0) -> dict:
    """Uniform rod, small static tip load, damping near critical: tip deflection against
    F L^3 / (3 E I)."""
    arm = SoftArm(n_elem=n_elem, drag=drag, sigma_T2=0.0, **CHECK_ROD)
    out = arm.simulate(T, 1, tip_force=lambda t: np.array([[F, 0.0, 0.0]]))
    p = arm.params
    I = np.pi * p["r_base"] ** 4 / 4
    predicted = F * p["length"] ** 3 / (3 * p["E"] * I)
    return dict(deflection=float(out["pos"][0, 0, -1]), predicted=float(predicted))


def check_frequency(n_elem: int = 30, T: float = 6.0) -> dict:
    """Uniform rod released from a short tip push, light damping: first bending frequency
    from the lowest peak of the tip's spectrum (an impulse also excites higher modes, which
    spoil zero-crossing counts), against the Euler-Bernoulli value."""
    arm = SoftArm(n_elem=n_elem, drag=0.05, sigma_T2=0.0, **CHECK_ROD)
    push = lambda t: np.array([[1e-3 if t < 0.02 else 0.0, 0.0, 0.0]])
    out = arm.simulate(T, 1, tip_force=push, record_every=10)
    x = np.array([f[0, 0, -1] for f in out["frames"]])
    tt = np.arange(len(x)) * 10 * out["dt"]
    xs = (x - x.mean()) * np.hanning(len(x))
    dt_rec = 10 * out["dt"]
    nfft = 16 * len(xs)
    spec = np.abs(np.fft.rfft(xs, nfft))
    freqs = np.fft.rfftfreq(nfft, dt_rec)
    band = (freqs > 0.5) & (freqs < 5.0)
    f_meas = freqs[band][np.argmax(spec[band])]
    p = arm.params
    A, I = np.pi * p["r_base"] ** 2, np.pi * p["r_base"] ** 4 / 4
    f_pred = 1.875 ** 2 * np.sqrt(p["E"] * I / (p["rho"] * A)) / (2 * np.pi * p["length"] ** 2)
    return dict(measured=float(f_meas), predicted=float(f_pred))


def check_grasp(curl: float = -0.04, T: float = 3.0, n_elem: int = 20) -> dict:
    """The default grasp: a curling couple ramps up over 1.5 s and the arm wraps the
    cylinder. Returns, for the distal third of the arm, the smallest and largest distance
    from the cylinder's axis in units of its radius (wrapped: close to 1, never well below)."""
    arm = soft_arm(n_elem=n_elem, grasp=True, sigma_T2=0.0)
    out = arm.simulate(T, 1, curl=lambda t: np.array([curl * min(1.0, t / 1.5)]))
    c = arm.params["cylinder"]
    x, z = out["pos"][0, 0], out["pos"][0, 2]
    d = np.hypot(x - c["x"], z - c["z"]) / c["r"]
    distal = d[-(n_elem // 3):]
    return dict(min=float(distal.min()), max=float(distal.max()))
