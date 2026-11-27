"""The soft arm as a MuJoCo link chain, checked against textbook beam results."""
import numpy as np
import pytest

pytest.importorskip("mujoco")

from mibook.bench.bodies import softarm as S  # noqa: E402


def test_cantilever_tip_deflection():
    # Midpoint joints: F L^3/(3EI) (1 - 1/(4 n^2)), within 0.1 percent at 20 links.
    r = S.check_cantilever()
    assert r["deflection"] == pytest.approx(r["predicted"], rel=0.01)


def test_first_bending_frequency():
    r = S.check_frequency()
    assert r["measured"] == pytest.approx(r["predicted"], rel=0.01)


def test_thermal_motion_keeps_the_gibbs_law():
    arm = S.soft_arm()
    rng = np.random.default_rng(0)
    q0, v0 = arm.gibbs_start(400, rng)
    q = arm.simulate(1.0, 400, rng=rng, q0=q0, v0=v0)["q"][:, -1]
    ratio = q.var(0) * arm.k / arm.params["sigma_T2"]
    assert ratio.mean() == pytest.approx(1.0, abs=0.05)


def test_step_guard():
    with pytest.raises(ValueError):
        S.SoftArm(n_links=40, **{**S.CHECK_ROD, "dt": 5e-4})


def test_soft_arm_wraps_the_object():
    r = S.check_grasp()
    assert r["min"] > 0.85          # no deep penetration
    assert r["max"] < 1.4           # the distal third lies along the object
