"""Routing efficiency in the Langevin body (research note 2026-11-21, thread T2)."""
import numpy as np
import pytest

from mibook import routing as rt


def test_state_space_matches_closed_forms():
    """The SEA motor and joint formulas for rho(w)."""
    assert rt.check_closed_forms()["max_abs_err"] < 1e-12


def test_no_internal_damping_means_perfect_routing():
    """Corollary: without internal dampers or readout noise every sensor routes everything."""
    r = rt.check_perfect_routing()
    assert r["rigid_internal_norm"] == 0.0
    for key in ("rigid_motor", "rigid_velocity", "sea_full_state"):
        np.testing.assert_allclose(r[key], 1.0, atol=1e-12)


def test_rho_does_not_depend_on_coordinates():
    """rho belongs to the physical sensor, not to the coordinates."""
    assert rt.check_reparametrization()["max_abs_err"] < 1e-10


def test_low_snr_efficiency_is_the_band_average_of_rho():
    r = rt.check_low_snr_limit()
    assert r["eta_low"] == pytest.approx(r["rho_mean"], rel=1e-5)
    assert r["eta_default"] > r["rho_mean"]          # SNR compresses the losses


def test_finite_window_efficiency_tracks_the_prediction():
    rows = rt.check_prediction(ks_list=(70.0, 1e4))
    for v in rows.values():
        assert 0.0 <= v["finite"] <= 1.0
        assert v["finite"] == pytest.approx(v["predicted"], abs=0.03)
    assert rows[(70.0, "joint")]["finite"] > 0.95       # joint side routes almost everything
    assert rows[(70.0, "motor")]["finite"] < 0.1        # motor side, behind the series damper
    assert rows[("rigid", "motor")]["finite"] > 0.998


def test_less_series_damping_means_more_routing():
    r = rt.check_series_damping()
    motor = [r[z]["motor"] for z in (0.1, 0.03, 0.01)]
    assert motor[0] < motor[1] < motor[2]
    full = [r[z]["full"] for z in r]
    assert max(full) - min(full) < 1e-5                  # the ceiling depends only on b_c


def test_encoder_noise_lowers_routing():
    for v in rt.check_encoder_noise().values():
        assert v["encoder"] < v["noiseless"]
