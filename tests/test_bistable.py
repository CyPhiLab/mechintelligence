"""A mechanical bit (research note, thread T7)."""
import numpy as np
import pytest

from mibook import bistable as bs


def test_simulated_escape_rate_is_close_to_kramers():
    for beta, v in bs.check_kramers(betas=(4.0, 6.0), n=2000).items():
        assert 0.75 < v["simulated"] / v["kramers"] < 1.05, beta


def test_a_barrier_keeps_the_bit_exponentially_longer():
    r = bs.check_lifetimes()
    growth = r["bistable"][1:] / r["bistable"][:-1] / np.exp(np.diff(r["betas"]))
    assert np.all((growth > 0.5) & (growth < 1.5))            # grows like e^beta
    assert r["linear"][-1] < 2 * r["linear"][0]                 # grows slowly
    assert r["bistable"][-1] > 1000 * r["linear"][-1]


def test_a_history_window_holds_what_its_first_state_holds():
    for v in bs.check_window().values():
        assert v["window"] == pytest.approx(v["state"], rel=1e-9)


@pytest.fixture(scope="module")
def writes():
    return bs.check_writes(kicks=(1.0, 2.0, 3.0), steps=(0.5, 1.1, 1.3), n=1500)


def test_a_steady_push_beyond_the_critical_force_writes_reliably(writes):
    v = writes[8.0]
    assert v["q0"] < 0.02                                   # the bit does not write itself
    assert v["step"]["q"][1] > 0.95 and v["step"]["bits"][1] > 0.85
    assert v["step"]["q"][0] < 0.3                          # half the critical force mostly fails
    assert 3.0 < v["step"]["energy"][1] < 4.5               # about 3 to 4 barriers of work


def test_a_sharp_kick_bounces_back(writes):
    v = writes[8.0]
    assert max(v["kick"]["bits"]) < max(v["step"]["bits"]) - 0.1
    assert v["kick"]["q"][2] < v["kick"]["q"][1]            # a harder kick overshoots


def test_a_low_barrier_leaks(writes):
    assert writes[4.0]["q0"] > 0.04
    assert max(writes[4.0]["step"]["bits"]) < 0.7
