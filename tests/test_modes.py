"""ModeSet cataloging, (m, n) resolution, and field evaluation."""

import numpy as np
import pytest

from chladni.beams import ClampedBeams
from chladni.modes import evaluate_field


def test_free_fundamental_label(catalogs):
    assert catalogs["free"].get(1).label == (1, 1)


def test_clamped_fundamental_label(catalogs):
    assert catalogs["clamped"].get(1).label == (1, 1)


def test_free_resolve_missing_pair(catalogs):
    # The free-edge plate has no single nodal line mode (2, 1).
    assert catalogs["free"].resolve(2, 1) is None


def test_free_resolve_curved_pair(catalogs):
    mode = catalogs["free"].resolve(2, 3)
    assert mode is not None
    assert mode.label == (2, 3)


def test_clamped_resolve_symmetric_twins(catalogs):
    a = catalogs["clamped"].resolve(2, 1)
    b = catalogs["clamped"].resolve(1, 2)
    assert a is not None and b is not None
    assert a.index == b.index


def test_labels_are_sorted_canonical(catalogs):
    labels = catalogs["clamped"].labels()
    assert labels == sorted(labels)
    for m, n in labels:
        assert m <= n


def test_get_missing_returns_none(catalogs):
    assert catalogs["free"].get(9999) is None


def test_field_missing_raises(catalogs):
    with pytest.raises(KeyError):
        catalogs["clamped"].field(9999, 64)


def test_field_shape(catalogs):
    assert catalogs["clamped"].field(1, 64).shape == (64, 64)


def test_fundamental_field_is_single_lobe(catalogs):
    field = catalogs["clamped"].field(1, 64)
    peak = float(np.max(field))
    # The truncated Ritz expansion leaves sub-1e-4 boundary ripples; the mode
    # is a single positive lobe apart from those.
    assert peak > 0
    assert np.min(field) > -0.01 * peak


def test_clamped_second_mode_has_nodal_line(catalogs):
    field = catalogs["clamped"].field(2, 64)
    assert np.any(field < 0) and np.any(field > 0)


def test_evaluate_field_direct():
    beams = ClampedBeams()
    coeffs = np.zeros(900)
    coeffs[0] = 1.0
    field = evaluate_field(beams, 30, coeffs, 32)
    assert field.shape == (32, 32)
    assert np.all(field > 0)
