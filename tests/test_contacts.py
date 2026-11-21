"""Two feet on the ground (research note 2026-11-21, thread T2)."""
import numpy as np
import pytest

from mibook import contacts as ct


def test_whole_body_sees_the_forces_through_one_damping_matrix():
    r = ct.check_whole_body()
    assert r["freq_rel_err"] < 1e-8                 # the same at every frequency
    assert r["series_rel_err"] < 1e-12              # B_c + b_x L, b_x = path dampers in series


def test_push_rule_with_two_feet():
    assert ct.check_push_rule()["max_rel_err"] < 1e-10


def test_static_law_with_two_feet():
    assert ct.check_two_foot_static()["max_rel_err"] < 1e-8


@pytest.fixture(scope="module")
def stance():
    return ct.check_stance()


def test_total_load_ignores_the_path_between_the_feet(stance):
    sums = [v["sd"][0] for v in stance["whole"].values()]
    diffs = [v["sd"][1] for v in stance["whole"].values()]
    assert max(sums) - min(sums) < 1e-4                # exact in the infinite-window limit
    assert diffs[0] > diffs[1] > diffs[2]
    assert diffs[2] < sums[2] - 5.0


def test_what_the_parts_hold_in_stance(stance):
    p = stance["parts"]
    assert p["stance foot"]["f"][0] > 45 and p["stance foot"]["f"][1] < 0.1   # its own force only
    assert p["second foot"]["f"][1] > 45 and p["second foot"]["f"][0] < 0.1
    t = p["trunk"]
    assert t["sd"][0] > 2 * t["sd"][1]                       # the trunk knows the total load
    assert t["f"]["all"] > t["f"][0] + t["f"][1] + 4          # and confuses the split
