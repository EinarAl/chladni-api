"""Beam eigenfunction properties: boundary conditions and orthogonality."""

import numpy as np

from chladni.beams import ClampedBeams, FreeFreeBeams

XS = np.linspace(0.0, 1.0, 512)


def test_clamped_beams_vanish_at_ends():
    beams = ClampedBeams()
    for n in range(3):
        assert np.allclose(beams.value(n, XS)[[0, -1]], 0.0, atol=1e-9)


def test_clamped_beams_zero_slope_at_ends():
    beams = ClampedBeams()
    for n in range(3):
        assert np.allclose(beams.deriv(n, XS)[[0, -1]], 0.0, atol=1e-7)


def test_free_beam_rigid_body_modes():
    beams = FreeFreeBeams()
    assert np.allclose(beams.value(0, XS), 1.0)
    assert np.allclose(beams.deriv(0, XS), 0.0)
    assert np.allclose(beams.value(1, XS), np.sqrt(3) * (2 * XS - 1))
    assert np.allclose(beams.deriv2(1, XS), 0.0, atol=1e-9)


def test_clamped_beam_orthogonality():
    beams = ClampedBeams()
    n = 6
    phi = np.array([beams.value(i, XS) for i in range(n)])
    gram = phi @ phi.T * (XS[1] - XS[0])
    off = gram - np.diag(np.diag(gram))
    assert np.all(np.abs(off) < 1e-6)
    assert np.all(np.diag(gram) > 0.5)


def test_free_beam_near_orthogonal():
    # The free-free convention (shared with the chladni-tuner reference) is
    # only approximately orthogonal; cross terms stay under 1e-2.
    beams = FreeFreeBeams()
    n = 6
    phi = np.array([beams.value(i, XS) for i in range(n)])
    gram = phi @ phi.T * (XS[1] - XS[0])
    off = gram - np.diag(np.diag(gram))
    assert np.all(np.abs(off) < 1e-2)
    assert np.all(np.diag(gram) > 0.5)
