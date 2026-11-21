"""Where the contact force goes in a chain (research note 2026-11-21, thread T2)."""
import numpy as np
import pytest

from mibook import chain as ch


def test_reciprocity_formula_matches_the_state_space():
    """Drive at the part; each internal damper costs b_j |stretch / contact motion|^2."""
    r = ch.check_reciprocity()
    assert r["chain_max_abs_err"] < 1e-10
    assert r["sea_max_abs_err"] < 1e-8


def test_cost_of_a_damper_is_the_same_for_every_part_beyond_it():
    assert ch.check_accumulation()["max_rel_spread"] < 1e-10


def test_static_law_on_random_chains():
    assert ch.check_static_law()["max_abs_err"] < 1e-8


def test_uniform_chain_shares():
    r = ch.check_chain_shares()
    share = r["share"]
    assert share[0] > 0.95 and share[1] < 0.4                 # the first damper costs most
    assert share[7] > share[6]                                # the free end has no backflow
    assert np.all(np.diff(r["static"]) < 0)                   # statically, monotone decline
    assert r["static"][1] == pytest.approx(0.2116, abs=1e-4)


def test_rod_share_is_mesh_independent_and_obeys_the_static_law():
    r = ch.check_rod_mesh()
    assert np.max(np.abs(r["rho_mean"][-1] - r["rho_mean"][-2])) < 1e-3
    np.testing.assert_allclose(r["static"], np.broadcast_to(r["static_law"], r["static"].shape),
                               atol=1e-6)
    assert r["rho_mean"][-1][0] < r["rho_mean"][-1][-1]       # the contact node holds least


def test_sea_spring_deflection():
    r = ch.check_deflection()
    assert r["max_abs_err"] < 1e-12
    s = [r["share"][k] for k in (70.0, 1e3, 1e4)]
    assert s[0] > s[1] > s[2]
    assert r["rho_near_dc"] < 1e-12                           # the deflection holds nothing at DC


def test_contact_node_loses_at_the_clamped_body_modes():
    for name, r in ch.check_contact_dips().items():
        dips, modes = r["dips_hz"], r["clamped_hz"]
        nearest = np.array([np.min(np.abs(modes - d)) / d for d in dips])
        assert dips.size >= 2 and np.all(nearest < 0.02), name


def test_several_coordinates_take_the_best_force_pattern():
    r = ch.check_matrix_reciprocity()
    assert r["max_rel_err"] < 1e-8
    assert r["n_beaten"] == 0
    assert r["reach_rel_err"] < 1e-8


def test_a_set_of_nodes_holds_statically_what_its_nearest_node_holds():
    assert ch.check_set_static_law()["max_abs_err"] < 1e-8


def test_contact_side_of_a_bridge_costs_the_same_for_everything_beyond():
    assert ch.check_bridge()["max_rel_spread"] < 1e-10


def test_branching_body():
    r = ch.check_tree()
    light, heavy = r[0.05], r[0.2]
    for v in r.values():
        np.testing.assert_allclose(v["static"], v["path_law"], atol=1e-8)
    assert abs(light["whole"] - heavy["whole"]) < 1e-6           # the ceiling ignores them
    np.testing.assert_allclose(light["static"][:4], heavy["static"][:4], atol=1e-12)
    assert heavy["share"][3] < light["share"][3] - 0.02           # backflow from the swing legs


def test_a_far_node_adds_in_band_what_it_cannot_add_statically():
    r = ch.check_pairs()
    assert r[(1, 7)] > r[(1,)] + 0.1
    assert r[(6, 7)] > max(r[(6,)], r[(7,)])
