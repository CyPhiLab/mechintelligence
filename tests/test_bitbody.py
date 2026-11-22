"""A mechanical bit inside a body (research note, thread T7)."""
import numpy as np
import pytest

from mibook import bitbody as bb


def test_body_pushing_back_lowers_the_barrier():
    e = bb.effective("latch", bb.beta_for_barrier("latch"))
    assert e["barrier_over_T"] == pytest.approx(bb.BETA, rel=1e-12)
    assert bb.beta_for_barrier("latch") > 1.4 * bb.BETA         # the latch alone needs more
    assert bb.effective("series", bb.BETA)["k_eff"] == pytest.approx(0.0, abs=1e-9)


def test_a_part_off_the_load_path_reads_nothing():
    r = bb.static_reading("series", "link")
    assert abs(r["shift"]) < 1e-15 and r["snapshot_d2"] < 1e-20


@pytest.fixture(scope="module")
def reading():
    return bb.check_reading()


def test_reading_grows_with_the_window(reading):
    v = reading[("latch", "link")]
    assert v["bits"][0] == pytest.approx(0.379, abs=0.01)
    assert v["bits"][3] > 0.97                                 # 10 ms
    np.testing.assert_allclose(v["approx"][1:], v["d2"][1:], rtol=0.08)
    assert reading[("latch", "rotor")]["bits"][0] > 0.999


def test_nonlinear_simulation_agrees_away_from_the_element():
    rows = bb.check_simulation(cases=(("latch", "link"), ("series", "link")), n=1000)
    v = rows[("latch", "link")]
    assert v["shift_sim"] == pytest.approx(v["shift_pred"], rel=0.07)
    assert v["sd_sim"] == pytest.approx(v["sd_pred"], rel=0.07)
    w = rows[("series", "link")]
    assert abs(w["shift_sim"]) < 4 * w["sd_of_shift"]
