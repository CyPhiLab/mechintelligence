"""Gull glider (Harvey and Inman 2022, PNAS 119: e2204847119), ported from AvianDynStab.
Every target below is quoted from the paper's text, with its location."""
import numpy as np
import pytest

from mibook.bench import make
from mibook.bench.bodies import glider as gl


@pytest.fixture(scope="module")
def grid():
    return gl.sweep_grid()


def test_counts_of_trimmed_and_stable_configurations(grid):
    """Results, 'Trim States': n = 1,457 trimmed configurations within the bounds;
    'Static Stability': n = 1,331 statically stable."""
    sel = gl.paper_selection(grid)
    assert sel.sum() == 1457
    assert (sel & grid["static_stable"]).sum() == 1331


def test_trim_speed_and_glide_angle_ranges(grid):
    """'the trim speed ranged from 11.8 to 29.8 m/s, and the shallowest trim glide angle was
    12.2 deg'."""
    sel = gl.paper_selection(grid)
    tr = grid["trim"]
    assert round(tr.U0[sel].min(), 1) == 11.8 and round(tr.U0[sel].max(), 1) == 29.8
    assert round(np.degrees(-tr.gamma[sel]).min(), 1) == 12.2


def test_mode_frequency_ranges(grid):
    """'The short-period frequency ranged from 8.9 to 41.0 rad/s'; the phugoid 'ranging
    from 0.45 to 1.10 rad/s'."""
    st = gl.paper_selection(grid) & grid["static_stable"]
    md = grid["modes"]
    assert round(np.nanmin(md["omega_sp"][st]), 1) == 8.9
    assert round(np.nanmax(md["omega_sp"][st]), 1) == 41.0
    assert round(np.nanmin(md["omega_ph"][st]), 2) == 0.45
    assert round(np.nanmax(md["omega_ph"][st]), 2) == 1.10


def test_statically_stable_means_dynamically_stable(grid):
    """'all the statically stable configurations had eigenvalues with negative real values
    for both oscillatory modes'."""
    st = gl.paper_selection(grid) & grid["static_stable"]
    assert np.all(grid["modes"]["eig"][st].real < 0)


def test_figure_6_decay_times(grid):
    """Elbow 130 deg, 15 deg forward sweep, dihedral 20 deg: 'For stable configurations with
    lower wrist angles, the time to half the amplitude varied between 2.22 and 2.52 s for the
    phugoid mode and 0.05 and 0.12 s for the short-period mode.' Time to half = 0.69/|Re|
    (analyze_data.R). The wrist angle of 156 deg 'had a trim glide angle of 61 deg'."""
    md, tr = grid["modes"], grid["trim"]
    st = gl.paper_selection(grid) & grid["static_stable"]
    k = np.flatnonzero((grid["elbow"] == 130) & (grid["sweep"] == -15) & (grid["dihedral"] == 20)
                       & np.isin(grid["wrist"], [106, 116, 126, 136, 146]))
    assert st[k].all()
    t_sp = 0.69 / np.abs(md["sp"][k].real)
    t_ph = 0.69 / np.abs(md["ph"][k].real)
    assert round(t_sp.min(), 2) == 0.05 and round(t_sp.max(), 2) == 0.12
    assert round(t_ph.min(), 2) == 2.22 and round(t_ph.max(), 2) == 2.52
    j = np.flatnonzero((grid["elbow"] == 130) & (grid["sweep"] == -15) & (grid["dihedral"] == 20)
                       & (grid["wrist"] == 156))[0]
    assert round(np.degrees(-tr.gamma[j])) == 61
    assert np.any(md["eig"][j].real > 0)                      # the unstable side of Fig. 6


def test_trim_balances_forces_and_moment():
    coef = gl.Coefficients()
    tr = gl.trim(coef, 130, 126, -15, 20)
    q = 0.5 * gl.RHO * tr.U0 ** 2 * gl.S_MAX
    W = gl.MASS * gl.G
    np.testing.assert_allclose(q * tr.CL, W * np.cos(tr.gamma), rtol=1e-12)
    np.testing.assert_allclose(q * tr.CD, -W * np.sin(tr.gamma), rtol=1e-12)
    assert abs(coef.Cm(130, 126, -15, 20, tr.CL)) < 1e-12


def test_streamwise_gust_settles_at_the_new_speed():
    """The paper's discrete gust: a 1 - cos rise of 2% of U0 over 5 s, then held (Methods;
    MIL-F-8785C). For a stable configuration the bird returns to trim relative to the air:
    u -> the gust step, as the A matrix and B column of dyn_functions.py imply."""
    import scipy.signal as sig
    b = make("gull_glider", elbow=130, wrist=126, sweep=-15, dihedral=20)
    t = np.arange(0, 60, 0.002)
    ug = np.where(t < 5, 0.5 * 0.02 * (1 - np.cos(np.pi * t / 5)), 0.02)
    sysd = sig.StateSpace(b.A, b.B("streamwise_gust"), np.eye(4), np.zeros((4, 1)))
    _, y, _ = sig.lsim(sysd, ug, t)
    np.testing.assert_allclose(y[-1], np.linalg.solve(b.A, -b.B("streamwise_gust")[:, 0] * 0.02),
                               atol=1e-6)
    assert np.all(np.isfinite(y)) and np.abs(y[-1, 2]) < 1e-6
