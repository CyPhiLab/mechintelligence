"""Erasing a mechanical bit (research note, thread T7)."""
import numpy as np
import pytest

from mibook import erasure as er


def test_landauer_bound_formula():
    assert er.landauer_bound(0.0) == pytest.approx(np.log(2.0), abs=1e-12)
    assert er.landauer_bound(0.5) == pytest.approx(0.0, abs=1e-12)


@pytest.fixture(scope="module")
def runs():
    return [er.erase(t, n=1000, seed=3) for t in (5.0, 50.0)]


def test_slower_erasure_costs_less_and_stays_above_the_bound(runs):
    fast, slow = runs
    assert fast["work"] > slow["work"] + 0.3
    for r in runs:
        assert r["left"] > 0.98
        assert r["work"] > er.landauer_bound(1.0 - r["left"]) - 3 * r["work_err"]
    assert slow["work"] < 0.85                       # close to ln 2 = 0.693
