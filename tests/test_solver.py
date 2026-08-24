"""Solver output: mode counts, ordering, calibration, determinism."""

import numpy as np
import pytest

from chladni.solver import _rayleigh_max_residual, _simpson_axis, solve_plate


def test_simpson_axis_integrates_constant_to_one():
    weights = np.ones(5)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2
    scale = (1.0 / 4) / 3.0
    ones = np.ones(5)
    assert _simpson_axis(ones, ones, weights, scale) == pytest.approx(1.0, abs=1e-12)


def test_rayleigh_residual_accepts_true_pairs_and_rejects_swapped():
    d = np.array([2.0, 5.0, 11.0])
    q = np.linalg.qr(np.vander(np.linspace(-0.9, 0.9, 3), 3) + np.eye(3))[0]
    k = (q * d) @ q.T
    m = q @ q.T
    assert _rayleigh_max_residual(k, m, d, q) < 1e-10

    scrambled = np.roll(d, 1)
    assert _rayleigh_max_residual(k, m, scrambled, q) > 0.1


def test_solve_rejects_unknown_bc():
    with pytest.raises(ValueError):
        solve_plate("hinged")


def test_free_mode_count_and_range(results):
    res = results["free"]
    assert res.bc == "free"
    assert len(res.modes) >= 88
    assert len(res.modes) <= 98
    assert res.freq_min == pytest.approx(23.0, abs=1e-6)
    assert res.freq_max < 2000.01


def test_clamped_mode_count_and_range(results):
    res = results["clamped"]
    # The converged clean-basis count at n_elastic=30 is 217; the floor sits
    # well below it to catch spectral collapse without pinning noise.
    assert len(res.modes) >= 200
    assert len(res.modes) <= 260
    assert res.freq_min == pytest.approx(23.0, abs=1e-6)
    assert res.freq_max < 2000.01


@pytest.mark.parametrize("bc", ["free", "clamped"])
def test_frequencies_ascending_and_positive(results, bc):
    freqs = [m.frequency for m in results[bc].modes]
    assert freqs == sorted(freqs)
    assert all(f > 0 for f in freqs)


def test_free_second_mode_is_cross(results):
    # The free-edge second mode is the anti-symmetric cross at ~27.7 Hz.
    f2 = results["free"].modes[1].frequency
    assert 26.0 < f2 < 29.5


def test_clamped_second_mode_plausible(results):
    f2 = results["clamped"].modes[1].frequency
    assert 40.0 < f2 < 52.0


def test_coefficient_shapes(results):
    assert results["free"].n_beams == 32
    assert results["clamped"].n_beams == 30
    assert results["free"].modes[0].coefficients.shape == (1024,)
    assert results["clamped"].modes[0].coefficients.shape == (900,)


def test_deterministic(results):
    again = solve_plate("free", quiet=True)
    a = [m.frequency for m in results["free"].modes]
    b = [m.frequency for m in again.modes]
    assert np.allclose(a, b, rtol=1e-12)


def test_convergence_small_basis(results):
    # The 20-beam basis is a subset of the 30-beam basis. Mode ordering can
    # drift between basis sizes, so compare by frequency proximity rather than
    # by index: every small-basis frequency must have a close partner in the
    # full-basis spectrum.
    small = solve_plate("free", n_elastic=20, quiet=True)
    full = [m.frequency for m in results["free"].modes]
    for sm in small.modes[:30]:
        nearest = min(abs(sm.frequency - f) for f in full)
        assert nearest <= 0.025 * sm.frequency
