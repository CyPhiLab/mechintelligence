"""What the body holds about terrain height (research note 2026-11-21, thread T11)."""
import numpy as np
import pytest

from mibook import terrain as tr


@pytest.fixture(scope="module")
def designs():
    return tr.check_designs()


@pytest.fixture(scope="module")
def sweeps():
    return tr.check_contact_sweeps()


def test_whole_body_height_bits_do_not_depend_on_the_design(designs):
    z = [v["whole_z"] for v in designs.values()]
    assert max(z) - min(z) < 1e-3


def test_parts_hold_nearly_the_same_share_of_height_as_of_force(designs):
    for v in designs.values():
        assert v["rotor_z"] == pytest.approx(v["rotor_f"], abs=0.01)


def test_stiffer_ground_helps_height_and_not_force(sweeps):
    z = [row["z"] for row in sweeps["k"]]
    f = [row["f"] for row in sweeps["k"]]
    assert np.all(np.diff(z) > 5)
    assert max(f) - min(f) < 1e-3


def test_ground_damping_hurts_force_always_and_height_only_up_to_a_point(sweeps):
    z = np.array([row["z"] for row in sweeps["b"]])
    f = np.array([row["f"] for row in sweeps["b"]])
    assert np.all(np.diff(f) < 0)
    i = int(np.argmin(z))
    assert 0 < i < len(z) - 1 and z[-1] > z[i] + 20


def test_mode_by_mode_formula(sweeps):
    for key in ("k", "b"):
        for row in sweeps[key]:
            assert row["z_pred"] == pytest.approx(row["z"], abs=0.5)
            assert row["f_pred"] == pytest.approx(row["f"], abs=0.1)
