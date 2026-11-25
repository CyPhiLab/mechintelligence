"""Item 4: the checks owed under thread T11 (exact finite windows, band growth, chains)."""
import numpy as np
import pytest

from mibook.bench import t11, linear as L


def test_exact_window_matches_the_stacked_computation():
    for key, (new, old) in t11.check_against_stacked().items():
        assert new == pytest.approx(old, rel=2e-4), key


def test_window_slope_matches_the_spectral_rate_on_the_leg():
    for key, r in t11.window_rates().items():
        assert r["slope"] == pytest.approx(r["whittle"], rel=2e-3), key


def test_several_coordinate_growth_matches_the_leading_coefficient():
    b = t11.chain_body()
    fs_list = (2000.0, 4000.0)
    for rows in ((1,), (1, 2)):
        C = t11.positions(b, rows)
        for i in range(3):
            rates = []
            for fs in fs_list:
                I1 = L.window_fisher(b, f"m{i}", C, int(0.5 * fs), fs, 0.0, rel_step=1e-2)
                I2 = L.window_fisher(b, f"m{i}", C, int(1.0 * fs), fs, 0.0, rel_step=1e-2)
                rates.append((I2 - I1) / 0.5)
            growth = t11.growth_per_hz(np.array(fs_list), np.array(rates))
            assert growth == pytest.approx(t11.predicted_growth(b, f"m{i}", rows), abs=0.02), (rows, i)
