"""The soft arm as a Cosserat rod (PyElastica), checked against textbook beam results."""
import pytest

pytest.importorskip("elastica")

from mibook.bench.bodies import softarm as S  # noqa: E402


def test_cantilever_tip_deflection():
    r = S.check_cantilever(n_elem=30)
    assert r["deflection"] == pytest.approx(r["predicted"], rel=0.06)


def test_first_bending_frequency():
    r = S.check_frequency(n_elem=30)
    assert r["measured"] == pytest.approx(r["predicted"], rel=0.05)


def test_soft_arm_wraps_the_object():
    from mibook.bench.bodies.softarm import check_grasp
    r = check_grasp()
    assert r["min"] > 0.85          # no deep penetration
    assert r["max"] < 1.4           # the distal third lies along the object
