"""Information propagation through the Langevin body (research note 2026-11-21, thread T2)."""
import numpy as np
import pytest

from mibook import propagation as pg


def test_state_space_matches_closed_forms():
    """The rotor and link formulas for rho(w)."""
    assert pg.check_closed_forms()["max_abs_err"] < 1e-12


def test_no_internal_damper_means_lossless_propagation():
    """Corollary: without internal dampers every part holds everything the body holds."""
    r = pg.check_lossless_propagation()
    assert r["rigid_internal_norm"] == 0.0
    for key in ("rigid_angle", "rigid_velocity", "sea_whole_body"):
        np.testing.assert_allclose(r[key], 1.0, atol=1e-12)


def test_rho_does_not_depend_on_coordinates():
    """rho belongs to the physical part, not to the coordinates."""
    assert pg.check_reparametrization()["max_abs_err"] < 1e-10


def test_a_coordinate_and_its_velocity_hold_the_same_spectral_share():
    r = pg.check_angle_equals_part()
    np.testing.assert_allclose(r["angle"], r["part"], rtol=1e-9)


def test_low_snr_share_is_the_band_average_of_rho():
    r = pg.check_low_snr_limit()
    assert r["share_low"] == pytest.approx(r["rho_mean"], rel=1e-5)
    assert r["share_default"] > r["rho_mean"]           # SNR compresses the losses


def test_finite_window_share_tracks_the_prediction():
    rows = pg.check_prediction(ks_list=(70.0, 1e4))
    for v in rows.values():
        assert 0.0 <= v["finite"] <= 1.0 + 1e-9
        assert v["finite"] == pytest.approx(v["predicted"], abs=0.03)
    assert rows[(70.0, "link")]["finite"] > 0.95       # contact side of the series damper
    assert rows[(70.0, "rotor")]["finite"] < 0.1       # behind it
    assert rows[("rigid", "rotor")]["finite"] > 0.998


def test_less_series_damping_means_more_information_at_the_rotor():
    r = pg.check_series_damping()
    rotor = [r[z]["rotor"] for z in (0.1, 0.03, 0.01)]
    assert rotor[0] < rotor[1] < rotor[2]
    whole = [r[z]["whole"] for z in r]
    assert max(whole) - min(whole) < 1e-5              # the ceiling depends only on b_c
