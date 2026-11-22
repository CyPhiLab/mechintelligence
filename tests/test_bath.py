"""A thermal bath with finite memory, and the band rule for masses and dampings."""
import numpy as np
import pytest

from mibook import bath


def test_bath_obeys_fluctuation_dissipation_and_gives_back_the_white_body():
    r = bath.check_bath()
    assert r["lyapunov"] < 1e-12
    assert r["force"][1e-6] == pytest.approx(r["white"], abs=0.01)


@pytest.fixture(scope="module")
def band():
    return bath.check_band(n_freq=1500)


def test_mass_information_grows_by_4_eps2_per_hertz_whatever_the_bath(band):
    target = 4 * band["eps"] ** 2
    for key in ("mass_white", "mass_tau4", "mass_tau3"):
        slope = np.diff(band[key])[-1] / np.diff(band["bands"])[-1]
        assert slope == pytest.approx(target, rel=0.03), key


def test_damping_grows_by_one_per_hertz_and_stiffness_settles(band):
    slope = np.diff(band["damp_white"])[-1] / np.diff(band["bands"])[-1]
    assert slope == pytest.approx(1.0, rel=0.03)
    assert band["stiff_white"][-1] - band["stiff_white"][-2] < 1.0


def test_a_stated_band_gives_a_finite_answer(band):
    assert band["mass_white"][0] == pytest.approx(205.3, abs=1.0)    # 0 to 100 Hz
    assert band["mass_tau4"][0] == pytest.approx(band["mass_white"][0], rel=0.01)
