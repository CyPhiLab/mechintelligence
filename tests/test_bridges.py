"""Bridges between the book's measures and other approaches (research note of 2026-11-25)."""
import numpy as np
import pytest

from mibook import bridges as br


def test_sampling_never_beats_the_continuum_and_reaches_it():
    r = br.check_sampling_bound()
    for name, row in r["rows"].items():
        ratio = row["ratio"]
        assert np.all(ratio <= 1.0 + 1e-9), name                 # g <= dt J D^+ J^T / 2 sigma^2
        assert abs(1.0 - ratio[-1]) < 1e-9, name                 # and reaches it as dt -> 0
        gap = 1.0 - ratio
        if gap[0] > 1e-9:                                        # above round-off: it closes
            assert np.all(np.diff(gap[:3]) < 0.0), name


def test_gap_closes_at_least_linearly():
    r = br.check_sampling_bound()
    for name in ("rigid", "SEA 70", "chain 3, all damped"):
        gap = 1.0 - r["rows"][name]["ratio"]
        assert gap[2] / gap[1] < 0.1 * 1.01                      # a decade of dt, at least a decade of gap


def test_singular_damping_is_covered():
    r = br.check_sampling_bound()
    row = r["rows"]["chain 4, singular D"]
    assert row["rank_D"] < row["n"]
    assert np.all(row["ratio"] <= 1.0 + 1e-9)


def test_outside_the_hypothesis_the_rate_diverges():
    g = br.check_sampling_bound()["outside_g_per_dt"]
    assert np.all(np.diff(np.log(g)) > np.log(30.0))             # grows fast as dt shrinks


def test_memory_capacity_and_information_share_a_spectrum():
    r = br.check_memory_capacity()
    assert r["lam_max"] > 1.0                                    # a regime where the two differ
    assert r["mc_sum"] == pytest.approx(r["mc_spectrum"], rel=1e-9)
    assert r["info_direct"] == pytest.approx(r["info_spectrum"], rel=1e-9)
    assert r["identity_err"] < 1e-12                             # C_j = 1 - exp(-2 I_j)
    assert r["mc_sum"] <= r["n_modes"]
    assert r["tail"] < 1e-12


def test_mcw_is_the_excess_entropy():
    for name, row in br.check_excess_entropy().items():
        for window, mi in row["past_future"].items():
            assert mi == pytest.approx(row["mcw_nats"], rel=1e-8), (name, window)


def test_series_rule():
    for row in br.check_series_rule()["rows"]:
        assert row["slope"] == pytest.approx(row["share"], rel=1e-6)


def test_directed_information_whole_state_and_part():
    r = br.check_directed_information()
    for n in r["n_t"]:
        assert r["whole"][n] == pytest.approx((n - 1) * r["per_step_te"], rel=1e-9)
    assert r["rotor_te1"] < 1e-3 * r["rotor_di_rate"]           # one-step TE misses nearly all


def test_zahedi_ay_measures_in_closed_form():
    z = br.check_zahedi_ay()
    p = z["passive"]
    for key in ("MC_W", "MI_MI", "MI_CA"):
        assert p[key] == pytest.approx(z["mcw_book"], rel=1e-9)
    assert abs(p["MC_A"]) < 1e-9
    assert z["open_loop"]["MC_A"] == pytest.approx(z["mc_a_formula"], rel=1e-8)
    assert z["open_loop"]["MC_W"] > z["open_loop"]["MI_MI"] > z["open_loop"]["MI_CA"]
    mcw = [v["MC_W"] for v in z["closed"].values()]
    assert np.all(np.diff(mcw) > 0)                              # feedback raises MC_W here


def test_empowerment_bounds_the_information_at_equal_power():
    r = br.check_empowerment_bound()
    assert 0 < r["info"] <= r["empowerment"]


def test_effective_damping_is_dissipation_per_contact_speed():
    for row in br.check_dissipation()["rows"]:
        assert row["from_power"] == pytest.approx(row["beff"], rel=1e-10)
