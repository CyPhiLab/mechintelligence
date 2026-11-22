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
