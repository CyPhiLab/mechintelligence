"""Item 2 of the 2026-11-25 priorities: the measures of different approaches, computed at
the book's resolution on the same linear bodies (thread T12).

A *family* is a set of items that should be ranked against each other: designs of a leg,
nodes of a network, parts of a whisker, members of a swarm. For each item every measure
is computed, and the measures are compared by how they rank the items (Spearman
correlation), since their units differ.

Measures (all about the environment input of the family unless stated):
- bits:  the book's information about a band-limited input profile (window, history);
- di:    directed-information rate about a white input (bits per sample);
- te1:   one-step transfer entropy (sensorimotor-studies form);
- mc:    linear memory capacity of a snapshot (reservoir computing);
- snap:  information in one snapshot about the input's whole past;
- cap:   capacity of the same channel (input port to part) under an input-power limit: the
         most any input law could put into the part;
- emp:   empowerment in the reverse direction: an actuator at the part acting on the
         contact (the environment's port), read at the contact;
- mcw:   Zahedi and Ay's MC_W of the passive body (about the body's own state; one value
         per body, so it varies only across designs);
- fisher: Fisher-information rate about an environmental property, where the family has one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

from . import linear as L
from . import make

MEASURES = ("bits", "di", "te1", "mc", "snap", "cap", "emp", "mcw", "fisher")


@dataclass
class Family:
    name: str
    items: list                    # (label, body, part)
    channel: str
    fs: float
    q: float                       # white-input variance per sample
    sigma_a: float                 # profile coefficient scale
    t_window: float
    band_hz: float
    horizon: int
    eps: float = 1e-9
    fisher: tuple | None = None    # (param, kwargs for fisher_rate)
    note: str = ""


def families() -> list[Family]:
    legs = [("rigid", make("rigid"), "rotor")] + [
        (f"SEA {ks:g}", make("sea", ks=ks), "rotor") for ks in (70.0, 200.0, 1e3, 3e3, 1e4, 1e5)]
    net = make("network")
    nodes = [(f"node{i}", net, f"node{i}") for i in range(net.params["n_nodes"])]
    w = make("whisker")
    n = w.H.shape[0]
    beam_parts = []
    for i in (0, 3, 7, 11, 15, 19, 23):                    # base angle, then displacements
        C = np.zeros((1, 2 * n))
        C[0, i] = 1.0
        beam_parts.append((f"dof{i}", w, C))
    sw = make("swarm")
    agents = [(f"agent{i}", sw, f"agent{i}") for i in range(sw.params["n_agents"])]
    return [
        Family("legs", legs, "force", 4000.0, 1e4, 100.0, 0.085, 100.0, 40,
               note="designs of the leg; part = rotor"),
        Family("network", nodes, "force", 1000.0, 1.0, 1.0, 0.5, 20.0, 100,
               fisher=("k_anchor", dict(band_hz=50.0, n_w=600)),
               note="nodes of one random network; property = anchor stiffness"),
        Family("whisker", beam_parts, "force", 4000.0, 1e-6, 1e-3, 0.085, 100.0, 100,
               fisher=("contact", dict(log=False, channel="force", level=1e-8, band_hz=200.0,
                                       n_w=600)),
               note="coordinates along the beam; property = contact point (through the force)"),
        Family("swarm", agents, "common", 50.0, 0.01, 0.1, 2.0, 2.0, 100,
               fisher=("k_0", dict(band_hz=5.0, n_w=600)),
               note="members of one swarm; property = tether stiffness"),
    ]


def evaluate(fam: Family) -> dict:
    rows = {}
    mcw_cache = {}
    for label, body, part in fam.items:
        kw = dict(fs=fam.fs)
        r = {}
        r["bits"] = L.profile_bits(body, part, fam.channel, t_window=fam.t_window, fs=fam.fs,
                                   band_hz=fam.band_hz, sigma_a=fam.sigma_a, eps=fam.eps)
        r["di"] = L.di_rate(body, part, fam.channel, fam.q, eps=fam.eps, **kw)
        r["te1"] = L.one_step_te(body, part, fam.channel, fam.q, eps=fam.eps, **kw)
        m = L.memory_capacity(body, part, fam.channel, fam.q, **kw)
        r["mc"], r["snap"] = m["mc"], m["nats"] / np.log(2)
        r["cap"] = L.empowerment(body, fam.channel, part, horizon=fam.horizon,
                                 power=fam.horizon * fam.q, eps=fam.eps, **kw)
        r["emp"] = reverse_empowerment(body, part, fam.channel, fam.horizon,
                                       fam.horizon * fam.q, fam.eps, fam.fs)
        key = id(body)
        if key not in mcw_cache:
            mcw_cache[key] = L.mc_w(body, **kw)
        r["mcw"] = mcw_cache[key]
        if fam.fisher is not None:
            param, fkw = fam.fisher
            r["fisher"] = L.fisher_rate(body, param, part, **fkw)
        else:
            r["fisher"] = np.nan
        rows[label] = r
    return rows


def reverse_empowerment(body, part, channel: str, horizon: int, power: float, eps: float,
                        fs: float) -> float:
    """Empowerment of an actuator at the part (forces on the part's position coordinates)
    over the contact, read where the environment's channel acts (its position)."""
    import dataclasses
    C = body.part(part) if isinstance(part, str) else np.atleast_2d(part)
    n = body.n
    act = C[:, :n]
    act = act[np.abs(act).sum(axis=1) > 0]
    J = np.atleast_2d(body.inputs[channel])
    contact = np.hstack([J, np.zeros_like(J)])
    b2 = dataclasses.replace(body, inputs={**body.inputs, "_act": act})
    return L.empowerment(b2, "_act", contact, horizon=horizon, power=power, eps=eps, fs=fs)


def rank_agreement(rows: dict) -> tuple[list[str], np.ndarray]:
    """Spearman correlations between measures across the items of a family. Measures that
    are constant across the items (or missing) are left out."""
    names = [m for m in MEASURES
             if np.all(np.isfinite([r[m] for r in rows.values()]))
             and np.ptp([r[m] for r in rows.values()]) > 1e-12 * max(1.0, abs(np.mean([r[m] for r in rows.values()])))]
    X = np.array([[r[m] for m in names] for r in rows.values()])
    if X.shape[0] < 3 or not names:
        return names, np.zeros((len(names), len(names)))
    rho = spearmanr(X).correlation
    rho = np.atleast_2d(rho)
    return names, rho


def run() -> dict:
    out = {}
    for fam in families():
        rows = evaluate(fam)
        names, rho = rank_agreement(rows)
        out[fam.name] = dict(rows=rows, names=names, rho=rho, note=fam.note)
    return out


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=140)
    res = run()
    for fam, r in res.items():
        print(f"\n== {fam}: {r['note']}")
        print("item".ljust(10), " ".join(m.rjust(9) for m in MEASURES))
        for label, row in r["rows"].items():
            print(label.ljust(10), " ".join(f"{row[m]:9.3g}" for m in MEASURES))
        print("spearman", r["names"])
        print(r["rho"])
