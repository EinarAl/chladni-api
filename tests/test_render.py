"""Rendering: marching-squares contours, SVG, PNG, and JSON grid."""

import numpy as np

from chladni import render


def test_constant_field_has_no_segments():
    field = np.full((8, 8), 0.7)
    assert render._marching_squares_segments(field) == []


def test_vertical_zero_crossing():
    xs = (np.arange(8) + 0.5) / 8 - 0.5
    field = np.broadcast_to(xs, (8, 8)).copy()
    segs = render._marching_squares_segments(field)
    assert len(segs) == 8
    for x0, y0, x1, y1 in segs:
        # Cell-center sampling places the contour on the shared edge at
        # i + tfrac, so the exact crossing lands at 3.5 (half-cell offset).
        assert abs(x0 - 3.5) < 1e-6 and abs(x1 - 3.5) < 1e-6
        assert y1 > y0


def test_fundamental_has_no_sand(catalogs):
    assert render._marching_squares_segments(catalogs["clamped"].field(1, 64)) == []
    assert render._marching_squares_segments(catalogs["free"].field(1, 64)) == []


def test_free_cross_has_contour(catalogs):
    segs = render._marching_squares_segments(catalogs["free"].field(2, 64))
    assert len(segs) > 0


def test_render_svg_sand(catalogs):
    svg = render.render_svg(catalogs["clamped"], 2, 128, "sand")
    assert svg.startswith("<svg")
    assert 'viewBox="0 0 128 128"' in svg
    assert "#e0cf9f" in svg


def test_render_svg_field(catalogs):
    svg = render.render_svg(catalogs["free"], 2, 128, "field")
    assert "#f4f1ea" in svg
    assert "#1b1c20" in svg


def test_render_png_magic(catalogs):
    png = render.render_png(catalogs["free"], 2, 128, "sand")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_png_field_magic(catalogs):
    png = render.render_png(catalogs["clamped"], 9, 128, "field")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_grid(catalogs):
    data = render.render_grid(catalogs["free"], 1, 32)
    assert data["resolution"] == 32
    assert len(data["values"]) == 32
    assert all(len(row) == 32 for row in data["values"])
    peak = max(abs(v) for row in data["values"] for v in row)
    assert peak <= 1.0
