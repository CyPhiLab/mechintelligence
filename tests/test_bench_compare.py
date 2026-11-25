"""Item 2: measures of different approaches on the same linear bodies (thread T12)."""
import numpy as np
import pytest

from mibook.bench import compare


@pytest.fixture(scope="module")
def res():
    return compare.run()


def test_leg_designs_force_measures_agree_and_mcw_reverses(res):
    names, rho = res["legs"]["names"], res["legs"]["rho"]
    force = [names.index(m) for m in ("bits", "di", "te1", "mc", "snap", "cap")]
    assert np.allclose(rho[np.ix_(force, force)], 1.0)
    assert np.allclose(rho[names.index("mcw"), force], -1.0)


def test_leg_empowerment_peaks_at_intermediate_stiffness(res):
    rows = res["legs"]["rows"]
    emp = {k: v["emp"] for k, v in rows.items()}
    assert emp["SEA 3000"] > emp["rigid"] > emp["SEA 70"]


def test_network_property_and_input_live_in_different_nodes(res):
    names, rho = res["network"]["names"], res["network"]["rho"]
    assert rho[names.index("bits"), names.index("fisher")] < -0.5
    assert rho[names.index("bits"), names.index("di")] > 0.9
    assert rho[names.index("bits"), names.index("mc")] < 0.6        # snapshot readout differs


def test_swarm_members_act_alike_but_perceive_differently(res):
    rows = res["swarm"]["rows"]
    emp = np.array([r["emp"] for r in rows.values()])
    bits = np.array([r["bits"] for r in rows.values()])
    assert np.ptp(emp) < 1e-9 * emp.mean()                         # momentum: same for all
    assert np.ptp(bits) > 0.05 * bits.mean()
    names, rho = res["swarm"]["names"], res["swarm"]["rho"]
    assert rho[names.index("bits"), names.index("di")] < 0          # the band decides who knows
