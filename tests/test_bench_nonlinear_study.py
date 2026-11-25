"""Item 5: nonlinear comparisons on the soft arm and the bistable chain (threads T7, T12)."""
from mibook.bench import nonlinear_study as N


def test_arm_degree2_capacity_needs_large_motion():
    small = N.arm_capacities(alpha=0.0, amp=0.02, batch=1000, history=20, max_lag2=8)
    large = N.arm_capacities(alpha=0.0, amp=0.1, batch=1000, history=20, max_lag2=8)
    assert large["deg2"] > 3 * small["deg2"]
    assert small["deg1"] > 2.0 and large["deg1"] > 2.0          # linear memory stays
    assert large["deg1"] + large["deg2"] <= large["dims"]


def test_chain_memory_lasts_longer_behind_a_higher_barrier():
    low = N.chain_memory(barrier=6.0, waits=(0.0, 100.0), batch=800)
    high = N.chain_memory(barrier=12.0, waits=(0.0, 100.0), batch=800)
    assert 0 < low["info"][0.0] <= low["ideal"] + 0.1
    assert high["info"][100.0] > low["info"][100.0] + 0.2
    assert high["info"][0.0] - high["info"][100.0] < low["info"][0.0] - low["info"][100.0]
    assert high["work"] > low["work"] > 0
