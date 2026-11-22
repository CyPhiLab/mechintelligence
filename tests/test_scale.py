"""Which properties have a finite information rate without an outside scale."""
import pytest

from mibook import scale as sc


@pytest.fixture(scope="module")
def crit():
    return sc.check_criterion(n_freq=1500)


def test_growth_per_hertz_matches_the_leading_coefficient(crit):
    for key, v in crit["rows"].items():
        assert v["slope"] == pytest.approx(v["predicted"], abs=0.01 + 0.01 * v["predicted"]), key


def test_a_part_learns_without_limit_only_about_what_it_touches(crit):
    rows = crit["rows"]
    unlimited = {("mass", "link"), ("b_c", "link"), ("b_s", "link"), ("b_s", "rotor")}
    for key, v in rows.items():
        if key in unlimited:
            assert v["predicted"] > 1e-3, key
        else:
            assert v["predicted"] < 1e-9, key                 # finite rate
            assert abs(v["rates"][-1] - v["rates"][-2]) < 1.0, key
