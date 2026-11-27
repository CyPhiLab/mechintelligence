"""One baseline number per benchmark: the "ready to go" check of thread T12.

    uv run python -m mibook.bench.baselines

Each row runs a small, fast computation that exercises the body and one measure. The
numbers are smoke-level baselines, not studies; sizes are kept small so the table builds in
about a minute.
"""
from __future__ import annotations

import time

import numpy as np

from . import make
from . import linear as L
from . import sampling as S


def _rows(rng):
    from .bodies.contact import crawl, walk
    from .bodies.fluids import body_acceleration_rms, flutter_speed
    from .bodies.nonlinear import mesostate

    yield "rigid", "bits about a 100 Hz force profile, whole body", L.profile_bits(make("rigid"))
    sea = make("sea", ks=70.0)
    yield "sea", "rotor bits at ks = 70 (whole body: same as rigid)", L.profile_bits(sea, "rotor", eps=1e-9)
    net = make("network")
    yield "network", "bits about the input force, node 0 / whole", (
        L.profile_bits(net, "node0", sigma_a=1.0, eps=1e-12), L.profile_bits(net, sigma_a=1.0))
    w = make("whisker")
    I = L.fisher_rate(w, "contact", "follicle", log=False, channel="force", level=1e-8, band_hz=200)
    yield "whisker", "contact-point error after 1 s, from the follicle (mm)", 1e3 / np.sqrt(I)

    arm = make("soft_arm")
    fs, sub, T, k = 1000.0, 5, 0.1, 8
    n_t = int(T * fs) + 1
    Phi = L.dct_basis(n_t, k)
    batch = 1500
    W = rng.standard_normal((batch, k)) * 0.1
    u = np.repeat((W @ Phi.T).T, sub, axis=0)[:, :, None]
    out, _ = arm.simulate(n_t * sub, 1 / fs / sub, batch, u=u, rng=rng,
                          parts=("tip_x",), burn_in=300)
    yield "soft_arm", "bits about a tip-force profile, tip (linear-estimate bound)", \
        S.lmmse_bits(W, out["tip_x"][::sub, :, 0][::4].T)

    ch = make("bistable_chain")
    batch, dt = 200, 0.01
    cache = ch._ou(dt)
    q = np.tile(ch.q0, (batch, 1))
    v = np.zeros((batch, ch.n))
    ramp = np.linspace(0, 1.3 * ch.critical_forces.max(), 2500)
    first = np.full((batch, ch.n), np.nan)
    for F in ramp:
        q, v, _ = ch.step(q, v, np.full((batch, 1), F), rng, dt, cache)
        st = mesostate(ch, q)
        first[st & np.isnan(first)] = F
    yield "bistable_chain", "switching force / critical force, weakest element", \
        float(np.nanmedian(first[:, np.argmin(ch.critical_forces)]) / ch.critical_forces.min())

    from .bodies.contact import apexes
    ho = make("hopper", actuator="musfib").simulate(T=4.0)
    yield "hopper", "apex height, muscle-fibre leg (m; published 1.070)", \
        float(apexes(ho["y"][:, 0], ho["t"]).mean())

    gp = make("gripper")
    fi, _ = S.path_fisher(gp, "k_o", n_steps=1000, dt=1e-4, batch=300, rng=rng, burn_in=500,
                          q_init=np.tile(gp.q0, (300, 1)), v_init=np.zeros((300, 3)))
    yield "gripper", "Fisher information about log object stiffness in 0.1 s", fi

    r = walk(make("walker"), 60, 8.0, rng=rng)
    yield "walker", "steps per second on a 0.08 rad slope", float(r["rate"].mean())

    tn = make("tentacle")
    _, (q, _) = tn.simulate(6000, 2e-4, 10, u=tn.grasp_input(1.0), rng=rng, parts=())
    yield "tentacle", "arm nodes touching the object after a 1 N grasp", \
        float(tn.in_contact(q).sum(axis=0).mean())

    rr = crawl(make("many_legged", n_legs=12, rugosity=0.015), 12, 2.0, rng=rng)
    yield "many_legged", "speed on rough ground (sd 1.5 cm), compliant legs (m/s) / fraction of " \
        "stance feet touching", (float(rr["speed"].mean()), float(rr["contact"].mean()))
    from .bodies.swarmalators import Swarmalators, order_parameters
    swl = Swarmalators(N=60, J=1.0, K=0.0, D_x=0.0, D_theta=0.0, seed=0)
    xs, ths, _ = swl.simulate(1500, 0.05, 1, rng=np.random.default_rng(0))
    yield "swarmalators", "order parameter S of the static phase wave (J=1, K=0)", \
        float(order_parameters(xs, ths)["S"][0])
    yield "foil", "flutter speed of the default section (m/s)", flutter_speed()
    yield "bird", "body acceleration in turbulence, compliant / stiff shoulder", (
        body_acceleration_rms(make("bird", k_s=20.0)), body_acceleration_rms(make("bird", k_s=1e4)))
    cl = L.mirror_stabilize(make("pendulum", k=5.0), "motor")
    yield "pendulum", "weak spring: least data rate (bits/s) / control power of the canonical stabilizer", (
        cl.min_data_rate, cl.control_power)
    sw = make("agents")
    kw = dict(channel="common", sigma_a=0.1, t_window=2.0, fs=50.0, band_hz=2.0)
    yield "agents", "bits about a common force, one agent / whole network", (
        L.profile_bits(sw, "agent0", eps=1e-9, **kw), L.profile_bits(sw, **kw))


def table(seed: int = 0) -> list[tuple[str, str, object, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    t = time.time()
    for name, what, val in _rows(rng):
        now = time.time()
        rows.append((name, what, val, now - t))
        t = now
    return rows


def _fmt(v):
    if isinstance(v, tuple):
        return " / ".join(f"{x:.4g}" for x in v)
    return f"{v:.4g}"


if __name__ == "__main__":
    for name, what, val, sec in table():
        print(f"{name:15s} {_fmt(val):>22s}  {what}  ({sec:.1f} s)")
