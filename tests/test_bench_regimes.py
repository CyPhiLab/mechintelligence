"""Item 3: holding, acting and paying across stiffness and frequency (thread T12)."""
import numpy as np
import pytest

from mibook.bench import regimes as R


@pytest.fixture(scope="module")
def leg():
    return R.leg_maps()


@pytest.fixture(scope="module")
def bird():
    return R.bird_maps()


def test_rotor_never_perceives_more_than_the_rigid_leg(leg):
    assert np.all(leg["percep"][1:] <= leg["percep"][0] * (1 + 1e-9))


def test_series_spring_never_saves_mechanical_work(leg):
    assert np.all(leg["work"][1:] >= leg["work"][0] * (1 - 1e-9))


def test_compliance_helps_action_near_resonance_and_hurts_above(leg):
    f = leg["f"]
    for i in range(1, len(leg["labels"])):
        fr = R.sea_resonance_hz(leg["labels"][i])
        j = np.argmin(abs(f - 0.8 * fr))
        k = np.argmin(abs(f - 3.0 * fr))
        if f[j] < 0.9 * f[-1] and 3.0 * fr < f[-1]:
            assert leg["action"][i, j] > leg["action"][0, j]
            assert leg["action"][i, k] < leg["action"][0, k]
            assert leg["shock"][i, k] < leg["shock"][0, k]          # isolation above resonance


def test_shoulder_perception_does_not_depend_on_its_stiffness(bird):
    P = bird["percep"]
    assert np.max(np.ptp(P, axis=0) / P.mean(axis=0)) < 1e-9


def test_soft_shoulder_rejects_fast_gusts_and_loses_fast_authority(bird):
    soft, stiff = 0, -1
    assert bird["gust"][soft, -1] < 0.1 * bird["gust"][stiff, -1]
    assert bird["action"][soft, -1] < 0.1 * bird["action"][stiff, -1]
