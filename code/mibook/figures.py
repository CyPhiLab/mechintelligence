"""Figures for the textbook, drawn from `mibook.checks`.

Inside the book each figure is a code cell that calls one `fig_*` function; the figure is
left open so Quarto displays it. Run as a script to write PDFs instead:

    uv run python -m mibook.figures OUTDIR
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from mibook import checks

OUT: pathlib.Path | None = None    # set by the script entry point; None inside the book

# Categorical slots, fixed order (blue, orange, aqua, yellow); text never wears series color.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4, "legend.frameon": False,
    "pdf.fonttype": 42, "font.family": "serif", "mathtext.fontset": "cm",
})


def _save(fig, name):
    """Write a PDF when run as a script; inside the book, leave the figure for display."""
    if OUT is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        print("wrote", OUT / f"{name}.pdf")
    return fig


def fig_invariance():
    r = checks.check_invariance()
    names = list(r["rows"])
    e = [r["rows"][n]["energy"] for n in names]
    s = [r["rows"][n]["state"] for n in names]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 2.1))
    ax.scatter(s, y, color=C2, s=28, zorder=3, label="per-coordinate noise (\"state\")",
               edgecolor="white", linewidth=0.8)
    ax.scatter(e, y, color=C1, s=28, zorder=4, marker="D",
               label="energy-whitened noise", edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, [n.replace("$", "$") for n in names])
    ax.set_xlabel("bits about the contact-force profile")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_xlim(60, 190)
    _save(fig, "invariance")
    return r


def fig_stiff_limit():
    r = checks.check_stiff_limit()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    a.semilogx(r["ks"], r["gap_state"], color=C2)
    a.semilogx(r["ks"], r["gap_energy"], color=C1)
    a.axhline(0.5 * r["n_modes"], color=INK2, lw=0.8, ls=":")
    a.axhline(0.0, color=INK2, lw=0.6)
    a.text(1e10, 0.5 * r["n_modes"] - 1.3, "8.5 bits = 17 modes / 2", color=INK2,
           fontsize=7, ha="center")
    a.text(2e5, 9.6, "per-coordinate", color=INK, fontsize=7.5)
    a.text(1e8, 0.7, "energy-whitened", color=INK, fontsize=7.5)
    a.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    a.set_ylabel("bits(SEA) $-$ bits(rigid)")
    a.set_title("(a) the redundant coordinate", loc="left")

    b.loglog(r["ks"], r["off_spring"], color=C1)
    b.loglog(r["ks"], r["off_rotor"], color=C3)
    ks = r["ks"]
    ref = r["off_spring"][-1] * (ks / ks[-1]) ** -0.5
    b.loglog(ks[ks > 1e6], ref[ks > 1e6], color=INK2, lw=0.8, ls="--")
    ref2 = r["off_rotor"][-1] * (ks / ks[-1]) ** -1.0
    b.loglog(ks[ks > 1e6], ref2[ks > 1e6], color=INK2, lw=0.8, ls="--")
    b.text(3e9, 3e-6, r"$\sqrt{k_s}\,\delta\ \ (\propto k_s^{-1/2})$", fontsize=7.5)
    b.text(3e5, 2e-9, r"$\sqrt{I_{\rm rot}}\,\dot\delta\ \ (\propto k_s^{-1})$", fontsize=7.5)
    b.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    b.set_ylabel(r"$\|\cdot\|$ over window and modes")
    b.set_title("(b) its whitened size vanishes", loc="left")
    fig.tight_layout()
    _save(fig, "stiff_limit")
    return r


def fig_snr():
    r = checks.check_snr()
    m = r["mults"]
    rig = r["sys"]["rigid"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    a.loglog(m, rig["exact"], color=C1)
    ok = rig["high"] > 0
    a.loglog(m[ok], rig["high"][ok], color=INK2, lw=0.8, ls="--")
    lo = rig["low"] < 1e3
    a.loglog(m[lo], rig["low"][lo], color=INK2, lw=0.8, ls=":")
    a.text(1.2e-3, 60, r"$\frac{r}{2}\log_2\sigma^{-2} + \frac{1}{2}\log_2\det G_0$", fontsize=7.5)
    a.text(20, 0.03, r"$\mathrm{tr}\,G_0 / (2\sigma^2\ln 2)$", fontsize=7.5)
    a.set_xlabel(r"noise scale $\sigma/\sigma_{\rm ref}$")
    a.set_ylabel("bits (rigid leg)")
    a.set_title("(a) two regimes", loc="left")

    b.semilogx(m, r["gap"], color=C1)
    b.axhline(r["det_gap"], color=INK2, lw=0.8, ls="--")
    b.axhline(0.0, color=INK2, lw=0.6)
    b.text(1.5e-3, r["det_gap"] + 0.003, r"$\frac{1}{2}\log_2\frac{\det G_{\rm SEA}}{\det G_{\rm rigid}}$",
           fontsize=7.5)
    b.set_xlabel(r"noise scale $\sigma/\sigma_{\rm ref}$")
    b.set_ylabel(r"bits(SEA $10^4$) $-$ bits(rigid)")
    b.set_title("(b) the design gap", loc="left")
    fig.tight_layout()
    _save(fig, "snr")
    return r


def fig_shapley():
    ks = np.logspace(1.8, 9, 25)
    phi = np.array([checks.shapley_coordinate_bits(checks.series_elastic(float(k),
                                                                         readout="energy"))
                    for k in ks])
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    labels = [r"$\theta_j$", r"$\theta_m$", r"$\dot\theta_j$", r"$\dot\theta_m$"]
    for i, (c, ls) in enumerate(zip((C1, C2, C3, C4), ("-", "--", "-", "--"))):
        ax.semilogx(ks, phi[:, i], color=c, ls=ls)
        ax.text(ks[-1] * 1.3, phi[-1, i] + (1.6 if i == 0 else -1.6 if i == 1 else 0),
                labels[i], color=INK, fontsize=8, va="center")
    ax.set_xlim(ks[0], ks[-1] * 6)
    ax.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    ax.set_ylabel("Shapley bits")
    _save(fig, "shapley")
    return phi


def fig_rod():
    r = checks.check_rod()
    N = r["N"]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.semilogx(N, r["per_node"], color=C2, marker="o", base=2)
    ax.semilogx(N, r["tip"], color=C3, marker="s", base=2)
    ax.semilogx(N, r["energy"], color=C1, marker="D", base=2)
    ax.text(N[-1] * 1.15, r["per_node"][-1], "per-node isotropic", va="center", fontsize=7.5)
    ax.text(N[-1] * 1.15, r["tip"][-1], "tip motion only", va="center", fontsize=7.5)
    ax.text(N[-1] * 1.15, r["energy"][-1], "energy-whitened", va="center", fontsize=7.5)
    ax.set_xlim(N[0] / 1.3, N[-1] * 12)
    ax.set_xlabel("finite elements $N$")
    ax.set_ylabel("bits about the tip force")
    _save(fig, "rod")
    return r


def fig_ellipse():
    """Mass on a spring: energy-shaped noise follows the constant-energy ellipses."""
    k, m, sig = 4.0, 1.0, 0.25
    rng = np.random.default_rng(1)
    th = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7), sharex=True, sharey=True)
    for ax, title, cov in ((axes[0], "(a) energy-shaped noise", np.diag([sig ** 2 / k, sig ** 2 / m])),
                           (axes[1], "(b) per-coordinate noise", np.diag([0.09, 0.09]) * sig ** 2 / 0.25)):
        for e in (0.5, 1.0, 1.5):
            ax.plot(e * sig * np.cos(th) / np.sqrt(k), e * sig * np.sin(th) / np.sqrt(m),
                    color=INK2, lw=0.7)
        pts = rng.multivariate_normal([0, 0], cov, 400)
        ax.scatter(pts[:, 0], pts[:, 1], s=3, color=C1 if ax is axes[0] else C2, alpha=0.6,
                   linewidths=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("position $q$")
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"velocity $\dot q$")
    fig.tight_layout()
    _save(fig, "ellipse")


def fig_free_motion():
    r = checks.check_free_motion()
    t = r["t"] * 1e3
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    for ax, key, title in ((a, "undamped", "(a) undamped"), (b, "damped", "(b) with the leg's damping")):
        res = r[key]
        ax.plot(t, res["per-coordinate"], color=C2, label="per-coordinate")
        ax.plot(t, res["kinetic"], color=C4, ls="--", label=r"kinetic ($\omega_{\rm ref}$)")
        ax.plot(t, res["action"], color=C3, ls=":", label="action ($p=-1$)")
        ax.plot(t, res["energy"], color=C1, label="energy")
        ax.axhline(2.0, color=INK2, lw=0.6)
        ax.set_xlabel("time since the signal was stored (ms)")
        ax.set_title(title, loc="left")
    a.set_ylabel("bits about the stored signal")
    a.legend(loc="lower right", ncol=2, fontsize=7)
    a.set_ylim(0, 4.1)
    fig.tight_layout()
    _save(fig, "free_motion")
    return r


def fig_alternatives():
    sea = checks.check_alternatives_sea()
    rod = checks.check_alternatives_rod()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ks = sea["ks"]
    styles = {"p=-1": (C3, ":"), "p=0": (C1, "-"), "p=1": (C4, "--"), "p=2": (INK2, "-."),
              "p=3": (C2, "-")}
    for name, (c, ls) in styles.items():
        a.semilogx(ks, np.abs(sea["gaps"][name]) + 1e-6, color=c, ls=ls)
    a.set_yscale("log")
    a.set_ylim(1e-6, 2e2)
    a.text(1.4e9, 120, "$p=3$", fontsize=7.5)
    a.text(1.4e9, 2.2, "$p=2$", fontsize=7.5)
    a.text(1.4e9, 1.4e-3, "$p=1$", fontsize=7.5)
    a.text(2e6, 3e-6, "$p\\leq0$", fontsize=7.5)
    a.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    a.set_ylabel(r"$|$bits(SEA) $-$ bits(rigid)$|$")
    a.set_title(r"(a) stiff limit, $W_p=\sum_i\omega_i^p E_i$", loc="left")

    N = rod["N"]
    for name, (c, ls, lab) in {"p=-1": (C3, ":", "action $p=-1$"), "p=0": (C1, "-", "energy $p=0$"),
                               "p=1": (C4, "--", "$p=+1$"), "kinetic": (INK2, "-.", "kinetic")}.items():
        v = rod["bits"][name]
        b.semilogx(N, v - v[-1], color=c, ls=ls, marker="o", markersize=3, base=2)
        b.text(N[-1] * 1.15, v[0] - v[-1] if name == "p=1" else (v[-1] - v[-1]), "", fontsize=7)
    b.text(5, -11, "$p=+1$ keeps growing", fontsize=7.5)
    b.text(17, -2.2, "$p\\leq 0$, kinetic: converge", fontsize=7.5)
    b.set_xlabel("finite elements $N$")
    b.set_ylabel(r"bits $-$ bits($N=64$)")
    b.set_title("(b) finite-element rod", loc="left")
    fig.tight_layout()
    _save(fig, "alternatives")
    return sea, rod


def fig_langevin():
    r = checks.check_langevin_designs()
    ks = r["ks"]
    rows = r["rows"]
    get = lambda key: np.array([rows[k][key] for k in ks]) - rows["rigid"][key]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    a.semilogx(ks, get("whole"), color=INK2, ls="-.")
    a.semilogx(ks, get("rotor"), color=C1, marker="o", markersize=3)
    a.semilogx(ks, get("static_rotor"), color=C3, marker="D", markersize=3)
    a.semilogx(ks, get("percoord_rotor"), color=C2, marker="s", markersize=3, lw=0.8, ls=":")
    a.axhline(0, color=INK2, lw=0.6)
    a.text(75, 2.5, "Langevin, whole body", fontsize=7.5)
    a.text(1.3e3, -24, "Langevin, rotor", fontsize=7.5)
    a.text(1.6e3, -42, "static energy noise, rotor", fontsize=7.5)
    a.text(4e3, 9, "fixed angle noise, rotor", fontsize=7.5)
    a.set_ylim(top=15)
    a.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    a.set_ylabel("bits(SEA) $-$ bits(rigid)")
    a.set_title("(a) design gap by noise model", loc="left")

    share = lambda part, whole: np.array([rows[k][part] / rows[k][whole] for k in ks])
    b.semilogx(ks, share("rotor", "whole"), color=C1, marker="o", markersize=3, label="Langevin")
    b.semilogx(ks, share("static_rotor", "static_whole"), color=C3, marker="D", markersize=3,
               label="static energy noise")
    rigid = rows["rigid"]["rotor"] / rows["rigid"]["whole"]
    b.axhline(rigid, color=INK2, lw=0.8, ls="--")
    b.text(80, rigid - 0.1, "rigid transmission (Langevin)", fontsize=7.5)
    b.set_ylim(0, 1.05)
    b.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    b.set_ylabel("rotor bits / whole-body bits")
    b.set_title("(b) share held by the rotor", loc="left")
    b.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    _save(fig, "langevin")
    return r


def fig_propagation():
    """Spectral propagation rho(f) to the SEA's rotor and link, and the finite-window share."""
    from mibook import propagation as pg
    f = np.linspace(0.5, 100.0, 400)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ks, c in ((70.0, C2), (1e3, C4), (1e4, C1)):
        a.plot(f, pg.sea_propagation_closed_form(ks, 2 * np.pi * f, "rotor"), color=c,
               label=f"$k_s={ks:g}$")
        a.plot(f, pg.sea_propagation_closed_form(ks, 2 * np.pi * f, "link"), color=c, ls="--")
    a.set_ylim(-0.02, 1.05)
    a.set_xlabel("force frequency (Hz)")
    a.set_ylabel(r"$\rho(f)=b_c/b_{\rm eff}(f)$")
    a.set_title("(a) solid: rotor; dashed: link", loc="left")
    a.legend(loc="lower right", fontsize=7)

    ks_list = np.array([70.0, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 1e6])
    for part, P, col, mk in (("rotor", pg.ROTOR["sea"], C1, "o"), ("link", pg.LINK, C3, "s")):
        fin, pred = [], []
        for ks in ks_list:
            s = pg.series_elastic(float(ks), readout="energy")
            fin.append(pg.information_share(s, P))
            pred.append(pg.predicted_share(s, P))
        b.semilogx(ks_list, fin, color=col, marker=mk, markersize=3, label=part)
        b.semilogx(ks_list, pred, color=col, ls=":", lw=1.0)
    b.set_ylim(0, 1.05)
    b.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    b.set_ylabel("share of the body's information")
    b.set_title("(b) finite window (dots: prediction)", loc="left")
    b.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return _save(fig, "propagation")


def fig_chain():
    """Share by node along the uniform chain, and the rod under mesh refinement."""
    from mibook import chain as ch
    r = ch.check_chain_shares()
    nodes = np.arange(r["share"].size)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    a.plot(nodes, r["share"], color=C1, marker="o", markersize=3, label="share, finite window")
    a.plot(nodes, r["rho_mean"], color=C3, marker="s", markersize=3, label=r"band mean of $\rho$")
    a.plot(nodes, r["static"], color=INK2, ls="--", lw=0.9, label=r"static law, $\rho(0)$")
    a.set_ylim(0, 1.05)
    a.set_xlabel("node (0 touches the ground)")
    a.set_ylabel("fraction of the body's information")
    a.set_title("(a) uniform chain", loc="left")
    a.legend(loc="upper right", fontsize=7)

    m = ch.check_rod_mesh()
    for N, row, col in zip(m["N"], m["rho_mean"], (C2, C4, C1, C3, INK2)):
        b.plot(m["fracs"], row, color=col, marker="o", markersize=3, lw=0.9, label=f"$N={N}$")
    b.plot(m["fracs"], m["static_law"], color=INK2, ls="--", lw=0.9)
    b.text(0.03, m["static_law"][1] - 0.06, r"static law, $\rho(0)$", fontsize=7.5)
    b.set_ylim(0.4, 1.05)
    b.set_xlabel("distance from the contact, $x/L$")
    b.set_ylabel(r"band mean of $\rho$ (0 to 400 Hz)")
    b.set_title("(b) free rod, mesh refinement", loc="left")
    b.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    return _save(fig, "chain")


def fig_terrain():
    """Whole-body bits about terrain height and about force as the ground changes."""
    from mibook import terrain as tr
    r = tr.check_contact_sweeps()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax, key, xlabel in ((a, "k", r"ground stiffness / baseline ($b_c$ fixed)"),
                            (b, "b", r"ground damping / baseline ($k_c$ fixed)")):
        x = [row["factor"] for row in r[key]]
        ax.semilogx(x, [row["z"] for row in r[key]], color=C1, marker="o", markersize=3,
                    label="terrain height")
        ax.semilogx(x, [row["f"] for row in r[key]], color=C3, marker="s", markersize=3,
                    label="contact force")
        ax.semilogx(x, [row["z_pred"] for row in r[key]], color=C1, ls=":", lw=1.0)
        ax.semilogx(x, [row["f_pred"] for row in r[key]], color=C3, ls=":", lw=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 80)
    a.set_xticks([0.25, 0.5, 1, 2, 4])
    a.set_xticklabels(["1/4", "1/2", "1", "2", "4"])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_ylabel("bits held by the whole body")
    a.set_title("(a) stiffer ground", loc="left")
    b.set_title("(b) more damped ground", loc="left")
    a.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return _save(fig, "terrain")


def fig_ground_stiffness():
    """Fisher information about log k_c by part, and the rotor's share next to force."""
    from mibook import parameters as pm
    r = pm.check_parts()
    ks = np.array([k for k in r if k != "rigid"])
    get = lambda key: np.array([r[k][key] for k in ks])
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    a.loglog(ks, get("whole"), color=INK2, ls="-.", label="whole body")
    a.loglog(ks, get("link"), color=C3, marker="s", markersize=3, label="link")
    a.loglog(ks, get("rotor"), color=C1, marker="o", markersize=3, label="rotor")
    a.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    a.set_ylabel(r"Fisher information about $\log k_c$")
    a.set_title("(a) thermal motion, 85 ms", loc="left")
    a.legend(loc="lower right", fontsize=7)
    b.semilogx(ks, get("rotor_share"), color=C1, marker="o", markersize=3,
               label="ground stiffness")
    b.semilogx(ks, get("rotor_force_share"), color=C2, marker="D", markersize=3,
               label="contact force")
    b.set_ylim(0, 1.05)
    b.set_xlabel(r"series stiffness $k_s$ (N m/rad)")
    b.set_ylabel("rotor / whole body")
    b.set_title("(b) share held by the rotor", loc="left")
    b.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return _save(fig, "ground_stiffness")


def fig_resolution():
    """Fisher rate against the band of motion resolved: mass and damping grow, stiffness settles."""
    from mibook import bath
    r = bath.check_band()
    B = r["bands"]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.loglog(B, r["mass_white"], color=C1, marker="o", markersize=3, label="link mass")
    ax.loglog(B, r["mass_tau3"], color=C1, marker="o", markersize=3, ls="--",
              label=r"link mass, bath with $\tau=1$ ms")
    ax.loglog(B, r["damp_white"], color=C2, marker="D", markersize=3, label="ground damping")
    ax.loglog(B, r["stiff_white"], color=C3, marker="s", markersize=3, label="ground stiffness")
    ax.loglog(B, 4 * r["eps"] ** 2 * B, color=INK2, ls=":", lw=0.9)
    ax.set_xlabel("band of motion resolved (Hz)")
    ax.set_ylabel("Fisher information per second")
    ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    return _save(fig, "resolution")


def fig_bistable():
    """How long a bit lasts: a bistable element against a harmonic one."""
    from mibook import bistable as bs
    g = bs.GAMMA
    t = np.geomspace(0.05, 2e4, 300)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for beta, col in ((4.0, C2), (6.0, C4), (8.0, C1)):
        a.semilogx(t, bs.bit_information(t, bs.kramers_rate(beta, g)), color=col,
                   label=rf"bistable, $\Delta E={beta:g}\,\sigma_T^2$")
        a.semilogx(t, bs.linear_information(t, g, beta), color=col, ls="--", lw=0.9)
    a.set_ylim(0, 1.05)
    a.set_xlabel(r"time ($\sqrt{m a^2/\sigma_T^2}$)")
    a.set_ylabel("bits about the push")
    a.set_title("(a) solid: bistable; dashed: harmonic", loc="left")
    a.legend(loc="lower left", fontsize=6.5)
    r = bs.check_lifetimes()
    b.semilogy(r["betas"], r["bistable"], color=C1, marker="o", markersize=3, label="bistable")
    b.semilogy(r["betas"], r["linear"], color=C3, marker="s", markersize=3, label="harmonic")
    k = bs.check_kramers()
    from scipy.optimize import brentq
    p = brentq(lambda q: bs.binary_entropy(q) - 0.5, 1e-6, 0.5 - 1e-9)
    sim = {beta: -np.log(1 - 2 * p) / (2 * v["simulated"]) for beta, v in k.items()}
    b.semilogy(list(sim), list(sim.values()), ls="none", marker="x", color=INK2,
               label="bistable, simulated")
    b.set_xlabel(r"barrier $\Delta E/\sigma_T^2$")
    b.set_ylabel("half-life of the bit")
    b.set_title("(b) time to fall to half a bit", loc="left")
    b.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return _save(fig, "bistable")


def fig_write():
    """Writing the bit: a sharp kick against a steady push."""
    from mibook import bistable as bs
    r = bs.check_writes()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for beta, col in ((4.0, C2), (8.0, C1)):
        v = r[beta]
        a.plot(v["step"]["energy"], v["step"]["bits"], color=col, marker="o", markersize=3,
               label=rf"steady push, $\Delta E={beta:g}\,\sigma_T^2$")
        a.plot(v["kick"]["energy"], v["kick"]["bits"], color=col, marker="s", markersize=3,
               ls="--", label=rf"sharp kick, $\Delta E={beta:g}\,\sigma_T^2$")
        b.plot(v["step"]["size"], v["step"]["q"], color=col, marker="o", markersize=3,
               label=rf"$\Delta E={beta:g}\,\sigma_T^2$")
    a.set_xlabel(r"energy put in by the push / $\Delta E$")
    a.set_ylabel("bits recorded")
    a.set_ylim(0, 1.35)
    a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a.set_title("(a) bits per write", loc="left")
    a.legend(loc="upper center", fontsize=5.5, ncol=2, frameon=False)
    b.axvline(1.0, color=INK2, lw=0.6, ls=":")
    b.set_xlabel(r"steady push / critical force $F_c$")
    b.set_ylabel("chance of landing in the right well")
    b.set_ylim(0, 1.05)
    b.set_title("(b) steady push", loc="left")
    b.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return _save(fig, "write")


def fig_bitbody():
    """What the rest of the body reads about a stored bit."""
    from mibook import bitbody as bb
    r = bb.check_reading(windows=(0.0, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02))
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    styles = {("latch", "link"): (C1, "-", "latch: link"), ("latch", "rotor"): (C1, ":", "latch: rotor"),
              ("series", "link"): (C3, "-", "series: link"), ("series", "rotor"): (C3, ":", "series: rotor")}
    for key, (col, ls, lab) in styles.items():
        v = r[key]
        a.plot(1e3 * v["windows"], v["bits"], color=col, ls=ls, marker="o", markersize=3, label=lab)
    a.set_ylim(-0.03, 1.05)
    a.set_xlabel("window of history (ms)")
    a.set_ylabel("bits about the stored bit")
    a.set_title("(a) reading the bit", loc="left")
    a.legend(loc="center right", fontsize=7)
    up, su = bb.simulate_window_means("latch", "link", +1, 0.002, n=2000, seed=0)
    dn, sd = bb.simulate_window_means("latch", "link", -1, 0.002, n=2000, seed=1)
    shift = bb.static_reading("latch", "link")["shift"]
    sdev = np.sqrt(bb.window_mean_variance("latch", "link", 0.002))
    bins = np.linspace(-0.5 * shift - 5 * sdev, 0.5 * shift + 5 * sdev, 50)
    b.hist(1e3 * up[su], bins=1e3 * bins, density=True, color=C1, alpha=0.5, label="bit = 1")
    b.hist(1e3 * dn[sd], bins=1e3 * bins, density=True, color=C3, alpha=0.5, label="bit = 0")
    xs = np.linspace(bins[0], bins[-1], 400)
    for m, col in ((0.5 * shift, C1), (-0.5 * shift, C3)):
        g = np.exp(-(xs - m) ** 2 / (2 * sdev ** 2)) / np.sqrt(2 * np.pi) / sdev / 1e3
        b.plot(1e3 * xs, g, color=col, lw=1.0)
    b.set_xlabel(r"link angle averaged over 2 ms ($10^{-3}$ rad)")
    b.set_ylabel("density")
    b.set_title("(b) latch: the link's view, simulated", loc="left")
    b.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return _save(fig, "bitbody")


def fig_erasure():
    """Erasing the bit: work against the protocol's duration, and the error it leaves."""
    from mibook import erasure as er
    r = er.check_erasure()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    a.errorbar(r["tau"], r["work"], yerr=2 * r["work_err"], color=C1, marker="o", markersize=3,
               capsize=2, label="simulated erasure")
    tt = np.geomspace(r["tau"][0], r["tau"][-1] * 3, 200)
    a.plot(tt, r["w_inf"] + r["b"] / tt, color=C1, ls=":", lw=0.9, label=r"$W_\infty+B/\tau$")
    a.axhline(np.log(2.0), color=INK2, lw=0.8, ls="--")
    a.text(r["tau"][0], np.log(2.0) - 0.08, r"Landauer: $\sigma_T^2\ln2$", fontsize=7.5)
    a.set_xscale("log")
    a.set_ylim(0.5, None)
    a.set_xlabel(r"duration of each stage $\tau$")
    a.set_ylabel(r"mean work / $\sigma_T^2$")
    a.set_title("(a) cost of erasing", loc="left")
    a.legend(loc="upper right", fontsize=7)
    b.semilogx(r["tau"], 100 * (1 - r["left"]), color=C3, marker="s", markersize=3)
    b.set_xlabel(r"duration of each stage $\tau$")
    b.set_ylabel("ended in the wrong well (%)")
    b.set_title("(b) error left by the erasure", loc="left")
    fig.tight_layout()
    return _save(fig, "erasure")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
    for f in (fig_invariance, fig_stiff_limit, fig_snr, fig_shapley, fig_rod, fig_ellipse,
              fig_free_motion, fig_alternatives, fig_langevin, fig_propagation, fig_chain, fig_terrain, fig_ground_stiffness, fig_resolution, fig_bistable, fig_write, fig_bitbody, fig_erasure):
        f()
