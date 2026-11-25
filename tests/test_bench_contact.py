"""Contact-tier benchmark bodies of mibook.bench (thread T12)."""
import numpy as np

from mibook import bench
from mibook.bench import sampling as S
from mibook.bench.bodies.contact import crawl, walk


def test_hopper_stands_without_thrust_and_hops_with_it():
    hp = bench.make("hopper")
    rng = np.random.default_rng(0)
    q0 = np.tile(hp.q0, (100, 1))
    out, _ = hp.simulate(2000, 5e-4, 100, rng=rng, parts=("foot",), q_init=q0,
                         v_init=np.zeros((100, 2)))
    assert (out["foot"][:, :, 0] > 0).mean() < 0.01                 # standing
    out, _ = hp.simulate(4000, 5e-4, 100, u=hp.hop_input, rng=rng, parts=("foot",), q_init=q0,
                         v_init=np.zeros((100, 2)))
    assert (out["foot"][2000:, :, 0] > 0).mean() > 0.3               # hopping: flight phases


def test_hopper_learns_the_ground_only_through_contact():
    hp = bench.make("hopper")
    fi, se = S.path_fisher(hp, "k_g", n_steps=1000, dt=5e-4, batch=200,
                           rng=np.random.default_rng(1), u=hp.hop_input, burn_in=2000,
                           q_init=np.tile(hp.q0, (200, 1)), v_init=np.zeros((200, 2)))
    assert fi > 10 * se > 0


def test_gripper_holds_its_squeeze():
    gp = bench.make("gripper")
    out, _ = gp.simulate(1500, 1e-4, 300, rng=np.random.default_rng(2), parts=("pads",),
                         q_init=np.tile(gp.q0, (300, 1)), v_init=np.zeros((300, 3)))
    np.testing.assert_allclose(out["pads"][-1].mean(0), gp.q0[[0, 2]], atol=2e-4)


def test_walker_rolls_steadily_and_faster_downhill():
    rng = np.random.default_rng(3)
    slow = walk(bench.make("walker", gamma=0.10), 60, 8.0, rng=rng)
    fast = walk(bench.make("walker", gamma=0.14), 60, 8.0, rng=rng)
    stall = walk(bench.make("walker", gamma=0.02), 60, 8.0, rng=rng)
    assert np.all(stall["v"] == 0.0)                                  # too shallow: it stops
    assert np.abs(stall["rate"]).mean() < 0.3                          # (a rock or two at most)
    assert slow["rate"].mean() > 0
    assert slow["rate"].std() < 0.1 * slow["rate"].mean()
    assert fast["speed"].mean() > slow["speed"].mean()


def test_tentacle_wraps_the_object():
    rng = np.random.default_rng(4)
    tn = bench.make("tentacle")
    _, (q, _) = tn.simulate(6000, 2e-4, 10, u=tn.grasp_input(1.0), rng=rng, parts=())
    assert tn.in_contact(q).sum(axis=0).mean() >= 1
    free = bench.make("tentacle", r_obj=1e-4)
    _, (q, _) = free.simulate(6000, 2e-4, 10, u=free.grasp_input(1.0), rng=rng, parts=())
    assert free.in_contact(q).sum() == 0


def test_more_legs_make_crawling_more_reliable_on_rough_ground():
    rng = np.random.default_rng(5)
    cv = {}
    for n in (6, 16):
        r = crawl(bench.make("many_legged", n_legs=n, rugosity=0.02), 150, 2.0, rng=rng)
        cv[n] = r["speed"].std() / r["speed"].mean()
    assert cv[16] < cv[6]
    smooth = crawl(bench.make("many_legged", rugosity=0.002), 100, 2.0, rng=rng)
    rough = crawl(bench.make("many_legged", rugosity=0.02), 100, 2.0, rng=rng)
    assert rough["speed"].mean() < smooth["speed"].mean()
