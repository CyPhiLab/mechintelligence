"""What the body holds about the ground's stiffness (research note, thread T11)."""
import numpy as np
import pytest

from mibook import parameters as pm


def test_snapshot_law():
    r = pm.check_snapshot_law()
    assert r["max_abs_err"] < 1e-6
    for (design, part), v in r["rows"].items():
        if part == "whole":
            assert v["fisher"] == pytest.approx(0.5, abs=1e-6)      # held only by the ground


def test_whole_body_rate_is_half_the_ground_corner_frequency():
    r = pm.check_rate_law()
    for row in r["b"]:
        assert row["rate"] == pytest.approx(row["predicted"], rel=1e-4)
    rates = list(r["designs"].values())
    assert max(rates) - min(rates) < 1e-3 * max(rates)                 # the same for every design


def test_finite_window_grows_at_the_rate():
    r = pm.check_window()
    np.testing.assert_allclose(r["slopes"], r["predicted"], rtol=1e-3)


def test_finite_differences_are_stable():
    a = pm.fisher_thermal(1e3, "rotor", dtheta=1e-3)
    b = pm.fisher_thermal(1e3, "rotor", dtheta=3e-3)
    assert a == pytest.approx(b, rel=1e-4)


def test_parts():
    r = pm.check_parts(ks_list=(70.0, 1e3, 1e4))
    whole = [r[k]["whole"] for k in (70.0, 1e3, 1e4)] + [r["rigid"]["whole"]]
    assert max(whole) - min(whole) < 1e-3 * max(whole)
    rotor = [r[k]["rotor_share"] for k in (70.0, 1e3, 1e4)]
    assert rotor[0] < rotor[1] < rotor[2]
    for k in (70.0, 1e3, 1e4):
        assert r[k]["rotor_share"] < r[k]["rotor_force_share"]        # stiffness reaches it less
        assert r[k]["push_whole"] > 50 * r[k]["whole"]                 # a known push dominates
