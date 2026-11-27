"""The AvInertia translation (mibook.bench.avinertia) against AvInertia's own example and
against numbers published in Harvey et al. (2022), Nature 603, 648-653.

The tests marked `full` need the whole specimen dataset (figshare item 14935659, collection
doi:10.6084/m9.figshare.c.5503989). Set AVINERTIA_DATA to the folder that holds its files
to run them; otherwise they are skipped and only the shipped subset is used.
"""
import csv
import os

import numpy as np
import pytest

from mibook.bench import avinertia as av
from mibook.bench.avinertia import model as M

EX = av.DATA / "package_example"
FULL = os.environ.get("AVINERTIA_DATA")
full = pytest.mark.skipif(not FULL, reason="set AVINERTIA_DATA to the full Harvey et al. dataset")


def _rows(name):
    with open(EX / name) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------

def test_parallel_axis_round_trip():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    I = A @ A.T
    d = rng.normal(size=3)
    back = M.parallelaxis(M.parallelaxis(I, d, 2.0, "CG"), d, 2.0, "A")
    np.testing.assert_allclose(back, I, atol=1e-12)


def test_bone_matches_a_thin_rod_limit():
    # A bone with a thin wall and long shaft along y: I_xx about its start -> m l^2 / 3.
    m, l = 0.01, 0.2
    r_out, r_in = 1e-4, 0.99e-4
    rho = m / (np.pi * (r_out ** 2 - r_in ** 2) * l)       # caps of (nearly) zero thickness
    out = M.massprop_bones(m, l, r_out, r_in, rho, np.zeros(3), np.array([0.0, l, 0.0]))
    assert out.I[0, 0] == pytest.approx(m * l ** 2 / 3, rel=1e-3)
    np.testing.assert_allclose(out.CG, [0, l / 2, 0], atol=1e-12)


# ---------------------------------------------------------------------------------------
# AvInertia's packaged example: pigeon 20_0300, test 4, frame 317 (R package data/*.rda)
# ---------------------------------------------------------------------------------------

def test_inputs_match_the_package_example():
    t = av.load_tables()
    sp = av.load_specimen("col_liv", "20_0300", tables=t)
    pkg = _rows("dat_bird_curr.csv")[0]
    checked = 0
    for k, v in pkg.items():
        if k in sp.bird and v not in ("", "NA") and isinstance(sp.bird[k], float):
            assert sp.bird[k] == pytest.approx(float(v), rel=1e-9, abs=1e-12), k
            checked += 1
    assert checked >= 75                       # every numeric measurement used
    for r in _rows("dat_bone_curr.csv"):
        mine = sp.bones[r["bone"]]
        for q, j in (("bone_mass", 0), ("bone_len", 1), ("bone_out_rad", 2), ("bone_in_rad", 3)):
            if j < len(mine):
                assert mine[j] == pytest.approx(float(r[q]), rel=1e-9), (r["bone"], q)
    feats = {f["feather"]: f for f in sp.feathers}
    for r in _rows("dat_feat_curr.csv"):
        f = feats[r["feather"]]
        assert f["m_f"] == pytest.approx(float(r["m_f"]), rel=1e-9)
        if r["feather"] == "alula":
            continue
        for q in ("l_vane", "l_cal", "w_cal", "vane_angle"):
            assert f[q] == pytest.approx(float(r[q]), rel=1e-9), (r["feather"], q)
        # The package's example vane widths carry the feather size scale s twice; the
        # paper's run_script.R applies it once (as here). s = l_vane / (vane length in m).
        s = float(r["l_vane"]) / (0.01 * float(r["vane length"]))
        for q in ("w_vd", "w_vp"):
            assert float(r[q]) == pytest.approx(s * f[q], rel=1e-9), (r["feather"], q)
    w = av.load_wings(sp, t)
    k = np.flatnonzero((w.frame_id == 317) & (w.test_id == 4))[0]
    rows = list(csv.reader(open(EX / "clean_pts.csv")))[1:]
    pkg_pts = np.array([[float(x) for x in r[1:]] for r in rows])
    np.testing.assert_allclose(w.clean[k], pkg_pts, atol=1e-15)


def test_whole_bird_on_the_shipped_wings():
    t = av.load_tables()
    for key in av.specimens(t):
        sp = av.load_specimen(*key, tables=t)
        w = av.load_wings(sp, t)
        parts = M.massprop_birdwing(sp, w.clean)
        bird = M.combine_inertialprop(M.massprop_restbody(sp), parts["wing"])
        # the skin takes the rest of the measured wing mass, so the wing weighs what was weighed
        assert parts["wing"].m == pytest.approx(sp.bird["wing_mass"], rel=1e-12)
        assert np.all(np.isfinite(bird.I)) and np.all(np.linalg.eigvalsh(bird.I) > 0)
        I = np.linalg.eigvalsh(bird.I)                       # triangle inequality of a body
        assert np.all(I[:, 0] + I[:, 1] >= I[:, 2] * (1 - 1e-9))
        assert np.allclose(bird.CG[:, 1], 0.0)
        st = av.stability(bird, w)
        assert np.all(np.isfinite(st.sm_nd))


def test_masses_match_the_published_plot():
    # bird_masses_2022.csv, supplied by the author: whole-bird masses read from the paper's
    # plots (provenance "Plot-derived estimate"). AvInertia's mass is the sum of the parts
    # (head, neck, torso and tail, and two wings), so it needs only the measurement sheets.
    names = {"Mallard": "ana_pla", "Canada goose": "bra_can", "Himalayan monal": "lop_imp",
             "Lady Amherst's pheasant": "chr_amh", "Silver pheasant": "lop_nyc",
             "Western grebe": "aec_occ", "Pigeon": "col_liv", "Black swift": "cyp_nig",
             "Common nighthawk": "cho_min", "American white pelican": "pel_ery",
             "Great blue heron": "ard_her", "Leach's storm petrel": "oce_leu",
             "Glaucous-winged gull": "lar_gla", "Cooper's hawk": "acc_coo",
             "Sharp-shinned hawk": "acc_str", "Barn owl": "tyt_alb", "Northern flicker": "col_aur",
             "Merlin": "fal_col", "Peregrine falcon": "fal_per", "Belted kingfisher": "meg_alc",
             "Common raven": "cor_cor", "Steller's jay": "cya_ste"}
    plot = {}
    for r in csv.DictReader(open(av.DATA / "plot_masses.csv")):
        plot.setdefault(names[r["species"]], []).append(float(r["mass_g"]))
    t = av.load_tables()
    birds = av.bird_rows(t)
    ours = {}
    for (s, b), r in birds.items():
        if (s, b) in av.data.SKIP or r["use_body"] != "Y":
            continue
        m = r["head_mass"] + r["neck_mass"] + r["torsotail_mass"] + 2 * r["wing_mass"]
        ours.setdefault(s, []).append(1000 * m)
    assert set(ours) == set(plot)
    for s in plot:
        a, b = sorted(ours[s]), sorted(plot[s])
        assert len(a) == len(b), s
        np.testing.assert_allclose(a, b, rtol=0.002, err_msg=s)
    # and the full pipeline gives the same mass
    sp = av.load_specimen("lar_gla", "20_0341", tables=t)
    bird = av.bird_inertia(sp, av.load_wings(sp, t).clean[:1])
    assert 1000 * bird.m == pytest.approx(1015.5, abs=0.5)


# ---------------------------------------------------------------------------------------
# Published results that need every wing shape (full dataset)
# ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def results():
    t = av.load_tables(FULL)
    birds = av.bird_rows(t)
    out = {}
    for key in av.specimens(t):
        sp = av.load_specimen(*key, tables=t, birds=birds)
        w = av.load_wings(sp, t)
        bird = av.bird_inertia(sp, w.clean)
        b = sp.bird
        length = b["torsotail_length"] + b["head_length"] + (b["neck_length"] if b["extend_neck"] else 0)
        out[key] = dict(sp=sp, w=w, bird=bird, st=av.stability(bird, w), length=length)
    return out


@full
def test_pigeon_wing_roll_inertia(results):
    # Nature 2022, Results: "the pigeon ... Ixx_wing (x 10^4) was between 1.42 and 1.92 kg m^2".
    # Per specimen: the shape with the largest wing Ixx, moved to the humeral head
    # (AnalysisFunctions/calc_MOIparams.R).
    vals = []
    for (s, b), r in results.items():
        if s != "col_liv":
            continue
        k = np.argmax(r["bird"].wing.I[:, 0, 0])
        vals.append(av.shift_origin(r["bird"].wing, r["w"].pts["pt1"])[k, 0, 0] * 1e4)
    assert len(vals) == 3
    assert min(vals) == pytest.approx(1.42, abs=0.005)
    assert max(vals) == pytest.approx(1.92, abs=0.005)


@full
def test_centre_of_gravity_shifts(results):
    # Nature 2022, Results: the largest CG shifts from elbow and wrist motion "were 3% (great
    # blue heron ..., 2.0 cm) and 2% (barn owl ..., 0.7 cm)" along x and z, as a fraction of
    # the bird's full length.
    dx = {k: np.ptp(r["bird"].CG[:, 0]) for k, r in results.items()}
    dz = {k: np.ptp(r["bird"].CG[:, 2]) for k, r in results.items()}
    kx = max(dx, key=lambda k: dx[k] / results[k]["length"])
    assert kx[0] == "ard_her"
    assert dx[kx] == pytest.approx(0.020, abs=0.0005)
    assert dx[kx] / results[kx]["length"] == pytest.approx(0.03, abs=0.005)
    owl = [k for k in dz if k[0] == "tyt_alb"]
    assert any(dz[k] == pytest.approx(0.007, abs=0.0005) for k in owl)
    assert max(dz[k] / results[k]["length"] for k in dz) == pytest.approx(0.02, abs=0.0015)


@full
def test_shoulder_rotation_shift(results):
    # Nature 2022, Results: rotating the wings 90 degrees about the humeral head gives
    # "maximum dxCG and dzCG shifts ... 18% (10.9 cm) for the great blue heron"
    # (process_outputdata.R, 'extremes of the CG position due to shoulder motion').
    best = (0, None)
    for k, r in results.items():
        b, w = r["bird"], r["w"]
        j = np.argmax(np.abs(b.wing.CG[:, 1] - w.pts["pt1"][:, 1]))
        wm, fm = b.wing.m, b.m
        rest_m = fm - 2 * wm
        wx, wy, wz = b.wing.CG[j]
        px, py, pz = w.pts["pt1"][j]
        rest_x = (fm * b.CG[j, 0] - 2 * wm * wx) / rest_m
        rest_z = (fm * b.CG[j, 2] - 2 * wm * wz) / rest_m
        fwd = (rest_m * rest_x + 2 * wm * ((wy - py) + px)) / fm
        back = (rest_m * rest_x + 2 * wm * (-(wy - py) + px)) / fm
        up = (rest_m * rest_z + 2 * wm * (-(wy - py) + pz)) / fm
        down = (rest_m * rest_z + 2 * wm * ((wy - py) + pz)) / fm
        shift = max(fwd - back, down - up)
        if shift / r["length"] > best[0]:
            best = (shift / r["length"], k, shift)
    assert best[1][0] == "ard_her"
    assert best[2] == pytest.approx(0.109, abs=0.001)
    assert best[0] == pytest.approx(0.18, abs=0.005)


@full
def test_inertia_ranges(results):
    # Nature 2022, Results: "elbow and wrist extension provided a more than 11-fold Ixx
    # increase (heron) and a 3-fold Izz increase (heron and owl)".
    ratio = lambda r, i: r["bird"].I[:, i, i].max() / r["bird"].I[:, i, i].min()
    assert max(ratio(r, 0) for k, r in results.items() if k[0] == "ard_her") > 11
    best = max(results, key=lambda k: ratio(results[k], 0))
    assert best[0] == "ard_her"
    for s in ("ard_her", "tyt_alb"):
        assert max(ratio(r, 2) for k, r in results.items() if k[0] == s) == pytest.approx(3, abs=0.25)


@full
def test_stability_classes(results):
    # Nature 2022, Results: from each species' mean maximum and minimum static margin, "four
    # species were entirely stable, one species was entirely unstable, and 17 species had the
    # capacity to shift between stable and unstable flight".
    by_species = {}
    for (s, b), r in results.items():
        by_species.setdefault(s, []).append((r["st"].sm_nd.min(), r["st"].sm_nd.max()))
    classes = []
    for v in by_species.values():
        lo, hi = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
        classes.append("stable" if lo > 0 else "unstable" if hi < 0 else "both")
    assert len(by_species) == 22
    assert (classes.count("stable"), classes.count("unstable"), classes.count("both")) == (4, 1, 17)
