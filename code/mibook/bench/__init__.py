"""Benchmark bodies and the engines that compute information measures on them (thread T12).

Layers:
- bodies (`mibook.bench.bodies`): each benchmark registers a factory with `register`;
- engines: `mibook.bench.linear` (exact, linear Gaussian) and `mibook.bench.sampling`
  (batched simulation for nonlinear and contact bodies);
- measures: functions in the engines that take a body and return numbers.

    from mibook import bench
    bench.available()                    # names, tiers and descriptions
    body = bench.make("sea", ks=1e3)     # a LinearBody, or a simulated benchmark
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

_REGISTRY: dict[str, "Entry"] = {}


@dataclass(frozen=True)
class Entry:
    name: str
    tier: str
    description: str
    factory: Callable


def register(name: str, tier: str, description: str):
    """Decorator: register a factory under `name`."""
    def deco(factory):
        _REGISTRY[name] = Entry(name, tier, description, factory)
        return factory
    return deco


def _load():
    from .bodies import legs, networks, beam, nonlinear, contact, fluids, swarm, pendulum  # noqa: F401


def make(name: str, **kw):
    _load()
    return _REGISTRY[name].factory(**kw)


def available() -> list[Entry]:
    _load()
    return sorted(_REGISTRY.values(), key=lambda e: (e.tier, e.name))
