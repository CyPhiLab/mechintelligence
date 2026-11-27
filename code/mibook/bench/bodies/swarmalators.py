"""Swarmalators: agents that swarm in space and synchronize in phase, each affecting the
other (O'Keeffe, Ceron and Petersen 2017, Nature Communications 8, 1504).

For agents i = 1..N with position x_i in the plane and phase theta_i,

    x_i' = v_i + (1/N) sum_{j != i} [ (x_j - x_i)/|x_j - x_i| (A + J cos(theta_j - theta_i))
                                       - B (x_j - x_i)/|x_j - x_i|^2 ]
    theta_i' = omega_i + (K/N) sum_{j != i} sin(theta_j - theta_i) / |x_j - x_i|

A and B set spatial attraction and repulsion; J > 0 makes agents with similar phases
attract more; K > 0 drives phases together, K < 0 apart. With identical agents
(v = omega = 0) and A = B = 1, the published phase diagram has five states: static sync,
static async, static phase wave, splintered phase wave and active phase wave. The
benchmark reproduces them (`check_states`).

Noise: independent Gaussian noise on positions (diffusion D_x) and phases (D_theta), the
resolution of this body. Environment channels: a common drift velocity u(t) of the medium
(added to every x_i') and a common periodic drive of strength F at frequency Omega on the
phases (F sin(Omega t - theta_i), a zeitgeber).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import register


@dataclass
class Swarmalators:
    N: int = 100
    A: float = 1.0
    B: float = 1.0
    J: float = 1.0
    K: float = 0.0
    D_x: float = 1e-4
    D_theta: float = 1e-4
    omega_spread: float = 0.0
    seed: int = 0
    name: str = "swarmalators"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.omega = self.omega_spread * rng.standard_normal(self.N)
        self.params = dict(N=self.N, A=self.A, B=self.B, J=self.J, K=self.K, D_x=self.D_x,
                           D_theta=self.D_theta, omega_spread=self.omega_spread, seed=self.seed)

    def initial(self, batch: int, rng) -> tuple[np.ndarray, np.ndarray]:
        """Positions uniform in [-1, 1]^2, phases uniform: the published initial condition."""
        x = rng.uniform(-1.0, 1.0, (batch, self.N, 2))
        th = rng.uniform(-np.pi, np.pi, (batch, self.N))
        return x, th

    def rates(self, x, th, drift=None, drive=None, t: float = 0.0):
        """Deterministic rates. x: (batch, N, 2), th: (batch, N)."""
        d = x[:, None, :, :] - x[:, :, None, :]                 # d[b, i, j] = x_j - x_i
        r2 = np.einsum("bijk,bijk->bij", d, d)
        np.einsum("bii->bi", r2)[...] = np.inf                   # no self-interaction
        r = np.sqrt(r2)
        dth = th[:, None, :] - th[:, :, None]                    # theta_j - theta_i
        att = (self.A + self.J * np.cos(dth)) / r
        rep = self.B / r2
        xdot = np.einsum("bij,bijk->bik", att - rep, d) / self.N
        thdot = self.omega[None, :] + self.K / self.N * np.sum(np.sin(dth) / r, axis=2)
        if drift is not None:
            xdot = xdot + np.asarray(drift)[:, None, :]
        if drive is not None:
            F, Om = drive
            thdot = thdot + F * np.sin(Om * t - th)
        return xdot, thdot

    def step(self, x, th, rng, dt: float, t: float = 0.0, drift=None, drive=None):
        """Euler-Maruyama step with Heun's correction for the deterministic part."""
        f1x, f1t = self.rates(x, th, drift, drive, t)
        xp, tp = x + dt * f1x, th + dt * f1t
        f2x, f2t = self.rates(xp, tp, drift, drive, t + dt)
        x = x + 0.5 * dt * (f1x + f2x) + np.sqrt(2 * self.D_x * dt) * rng.standard_normal(x.shape)
        th = th + 0.5 * dt * (f1t + f2t) + np.sqrt(2 * self.D_theta * dt) * rng.standard_normal(th.shape)
        return x, th

    def simulate(self, n_steps: int, dt: float, batch: int = 1, rng=None, record_every: int = 0,
                 drift=None, drive=None):
        """drift: callable t -> (batch, 2) or None; drive: (F, Omega) or None."""
        rng = np.random.default_rng() if rng is None else rng
        x, th = self.initial(batch, rng)
        frames = []
        for k in range(n_steps):
            t = k * dt
            dr = drift(t) if callable(drift) else drift
            x, th = self.step(x, th, rng, dt, t, dr, drive)
            if record_every and k % record_every == 0:
                frames.append((x.copy(), th.copy()))
        return x, th, frames


def order_parameters(x: np.ndarray, th: np.ndarray) -> dict:
    """S+ and S- (phase-position correlation, O'Keeffe et al.), R (phase sync), per batch.
    phi is each agent's angle about the swarm's centre."""
    c = x - x.mean(axis=1, keepdims=True)
    phi = np.arctan2(c[..., 1], c[..., 0])
    Sp = np.abs(np.mean(np.exp(1j * (phi + th)), axis=1))
    Sm = np.abs(np.mean(np.exp(1j * (phi - th)), axis=1))
    R = np.abs(np.mean(np.exp(1j * th), axis=1))
    return dict(S_plus=Sp, S_minus=Sm, S=np.maximum(Sp, Sm), R=R)


STATES = {                                    # (J, K) of the published examples
    "static sync": (0.1, 1.0),
    "static async": (0.1, -1.0),
    "static phase wave": (1.0, 0.0),
    "splintered phase wave": (1.0, -0.1),
    "active phase wave": (1.0, -0.75),
}


@register("swarmalators", "5-collective",
          "Swarmalators (O'Keeffe et al. 2017): agents that swarm and sync, each shaping the other.")
def swarmalators(N: int = 100, J: float = 1.0, K: float = 0.0, D_x: float = 1e-4,
                 D_theta: float = 1e-4, seed: int = 0) -> Swarmalators:
    return Swarmalators(N=N, J=J, K=K, D_x=D_x, D_theta=D_theta, seed=seed)


def check_states(N: int = 100, T: float = 200.0, dt: float = 0.05, seed: int = 0) -> dict:
    """The five published example states, without noise: order parameters at the end, mean
    speed and phase speed over a further 10 time units, and the number of phase clusters
    (gaps wider than 0.3 rad between sorted phases). Noise is off because its drift is as
    large as the small motions that tell the splintered state from the static one."""
    out = {}
    for name, (J, K) in STATES.items():
        sw = Swarmalators(N=N, J=J, K=K, D_x=0.0, D_theta=0.0, seed=seed)
        x, th, _ = sw.simulate(int(T / dt), dt, 1, rng=np.random.default_rng(seed))
        o = {k: float(v[0]) for k, v in order_parameters(x, th).items()}
        rng = np.random.default_rng(seed + 1)
        disp = dph = 0.0
        n = int(10.0 / dt)
        for _ in range(n):
            x2, th2 = sw.step(x, th, rng, dt)
            disp += float(np.mean(np.linalg.norm(x2 - x, axis=-1)))
            dph += float(np.mean(np.abs(np.angle(np.exp(1j * (th2 - th))))))
            x, th = x2, th2
        srt = np.sort(np.mod(th[0], 2 * np.pi))
        gaps = np.diff(np.r_[srt, srt[0] + 2 * np.pi])
        o.update(speed=disp / 10.0, phase_speed=dph / 10.0, clusters=int((gaps > 0.3).sum()))
        out[name] = o
    return out
