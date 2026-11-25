"""Fluid and collective benchmark bodies of mibook.bench (thread T12)."""
import numpy as np
import pytest

from mibook import bench
from mibook.bench import linear as L
from mibook.bench.bodies.fluids import body_acceleration_rms, flutter_speed, karman_gust


def test_foil_flutters_where_the_textbook_says():
    U = flutter_speed(mu=20, r_alpha2=6 / 25, x_alpha=0.1, omega_h=10.0, omega_alpha=25.0,
                      a=-0.2, b=0.15)
    assert U / (0.15 * 25.0) == pytest.approx(2.17, abs=0.02)


def test_foil_below_flutter_is_stable_and_responds_to_gusts():
    f = bench.make("foil")
    assert np.linalg.eigvals(f.A).real.max() < 0
    # A street shed at 4 Hz (cylinder 0.3 m at 6 m/s), near the pitch mode.
    out = L.simulate_linear(f, 1000, 1e-3, 20, u=karman_gust(1000, 1e-3, 20, d=0.3), channel="gust",
                            parts=("pitch",), rng=np.random.default_rng(0))
    quiet = L.simulate_linear(f, 1000, 1e-3, 20, parts=("pitch",), rng=np.random.default_rng(0))
    assert out["pitch"].std() > 10 * quiet["pitch"].std()
    assert L.input_rate(f, "structure", channel="gust", band_hz=50, level=1e-3) >= \
        L.input_rate(f, "pitch", channel="gust", band_hz=50, level=1e-3)


def test_bird_is_stable_for_any_shoulder():
    for ks in (5.0, 50.0, 1e4):
        b = bench.make("bird", k_s=ks)
        assert np.linalg.eigvals(b.A).real.max() < 0
        assert body_acceleration_rms(b) > 0


def test_swarm_one_member_holds_less_than_the_swarm():
    sw = bench.make("swarm")
    kw = dict(channel="common", sigma_a=0.1, t_window=2.0, fs=50.0, band_hz=2.0)
    one = L.profile_bits(sw, "agent0", eps=1e-9, **kw)
    whole = L.profile_bits(sw, **kw)
    assert 0 < one < whole


def test_every_registered_body_has_a_baseline():
    from mibook.bench import available
    from mibook.bench.baselines import table
    rows = table()
    assert {r[0] for r in rows} == {e.name for e in available()}
    for name, _, val, _ in rows:
        vals = val if isinstance(val, tuple) else (val,)
        assert all(np.isfinite(x) for x in vals), name
