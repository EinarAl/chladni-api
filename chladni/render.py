"""Chladni pattern rendering: SVG (vector nodal lines) and PNG (raster).

Two visual styles:
  sand  - the classic experiment: sand gathered on the nodal lines against a
          dark plate. Nodal lines are the zero contour of the mode shape,
          traced with marching squares, so they stay crisp at any zoom.
  field - the signed displacement field, positive lobes light and negative
          lobes dark.

Both styles share the same zero-contour extraction: sand draws it as lines,
field encodes the full scalar field.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np

from .modes import ModeSet

if TYPE_CHECKING:
    from PIL import Image

MAX_RESOLUTION = 1024
MIN_RESOLUTION = 32

_PLATE = "#17171a"
_SAND = "#e0cf9f"
_BORDER = "#4a4d55"
_LIGHT = "#f4f1ea"
_LINE = "#1b1c20"


def _normalized(field: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(field)))
    if peak == 0.0:
        return np.zeros_like(field)
    return field / peak


_TINY = 1e-3  # relative ripple floor, as a fraction of the peak field value


def _marching_squares_segments(field: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Trace zero crossings of the field into line segments.

    Field values sit at cell centers; the lattice is extended by one point
    along each axis. For every cell that changes sign across exactly two
    edges, the crossing points are interpolated and a segment is emitted.
    Edge math is vectorized; only the segment assembly loops per cell.

    The truncated Ritz expansion leaves sub-percent boundary ripples on
    otherwise single-lobe modes (e.g. the clamped fundamental). Values below
    the ripple floor are clamped to zero so they do not render as spurious
    nodal speckle. Nodal lines land with half-cell accuracy because the field
    is sampled at cell centers.
    """
    f = field
    peak = float(np.max(np.abs(f)))
    if peak > 0.0:
        f = np.where(np.abs(f) < _TINY * peak, 0.0, f)
    g = f.shape[0]
    ext = np.zeros((g + 1, g + 1))
    ext[:g, :g] = f
    ext[:g, g] = f[:, -1]
    ext[g, :g] = f[-1, :]
    ext[g, g] = f[-1, -1]

    v00 = ext[:g, :g]
    v10 = ext[:g, 1:]
    v01 = ext[1:, :g]
    v11 = ext[1:, 1:]

    def tfrac(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        denom = a - b
        return np.where(denom != 0.0, a / np.where(denom != 0.0, denom, 1.0), 0.5)

    xs = np.arange(g)[None, :]
    ys = np.arange(g)[:, None]
    xx = np.broadcast_to(xs, (g, g))
    yy = np.broadcast_to(ys, (g, g))

    pb = (xx + tfrac(v00, v10), yy)              # bottom edge, x varies
    pr = (xx + 1.0, yy + tfrac(v10, v11))        # right edge, y varies
    pt = (xx + 1.0 - tfrac(v11, v01), yy + 1.0)  # top edge, x varies
    pl = (xx, yy + 1.0 - tfrac(v01, v00))        # left edge, y varies

    present = np.stack(
        [
            (v00 < 0) != (v10 < 0),
            (v10 < 0) != (v11 < 0),
            (v11 < 0) != (v01 < 0),
            (v01 < 0) != (v00 < 0),
        ],
        axis=0,
    )
    count = present.sum(axis=0)
    seg: list[tuple[float, float, float, float]] = []
    for j, i in np.argwhere(count == 2):
        pts: list[tuple[float, float]] = []
        for k, (px, py) in enumerate((pb, pr, pt, pl)):
            if present[k, j, i]:
                pts.append((float(px[j, i]), float(py[j, i])))
        if len(pts) == 2:
            seg.append((pts[0][0], pts[0][1], pts[1][0], pts[1][1]))
    return seg


def _svg(segments: list[tuple[float, float, float, float]], size: int, sand: bool) -> str:
    bg, line, border = (_PLATE, _SAND, _BORDER) if sand else (_LIGHT, _LINE, "#c8c4b8")
    parts = [f"M{x0:.3f} {y0:.3f}L{x1:.3f} {y1:.3f}" for x0, y0, x1, y1 in segments]
    path = (
        f'<path d="{" ".join(parts)}" fill="none" stroke="{line}" '
        f'stroke-width="1.3" stroke-linecap="round"/>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
        f'width="{size}" height="{size}">'
        f'<rect width="{size}" height="{size}" fill="{bg}"/>'
        f'{path}'
        f'<rect x="0.5" y="0.5" width="{size - 1}" height="{size - 1}" '
        f'fill="none" stroke="{border}" stroke-width="2"/>'
        f'</svg>'
    )


def render_svg(mode_set: ModeSet, index: int, resolution: int, style: str) -> str:
    field = mode_set.field(index, resolution)
    segments = _marching_squares_segments(field)
    return _svg(segments, resolution, style == "sand")


def _to_png_image(field: np.ndarray, style: str, resolution: int) -> "Image":
    from PIL import Image

    segments = _marching_squares_segments(field)
    if style == "sand":
        img = Image.new("L", (resolution, resolution), 0)
        img = _draw_segments_l(img, segments, field.shape[0])
        return img
    gray = ((_normalized(field) + 1.0) * 0.5 * 200).astype(np.uint8)
    img = Image.fromarray(gray, mode="L").resize((resolution, resolution))
    return _draw_segments_l(img, segments, field.shape[0], width=2, color=0)


def _draw_segments_l(img, segments, g, width=2, color=255):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    # Scale the lattice segments to pixel space with a margin for the border.
    scale = (img.width - 2) / g
    for x0, y0, x1, y1 in segments:
        draw.line(
            (1 + x0 * scale, 1 + y0 * scale, 1 + x1 * scale, 1 + y1 * scale),
            fill=color,
            width=width,
        )
    return img


def render_png(mode_set: ModeSet, index: int, resolution: int, style: str) -> bytes:
    field = mode_set.field(index, resolution)
    img = _to_png_image(field, style, resolution)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_grid(mode_set: ModeSet, index: int, resolution: int) -> dict:
    """Raw field data as JSON, for callers who want to do their own work."""
    field = _normalized(mode_set.field(index, resolution))
    return {
        "resolution": resolution,
        "values": np.round(field, 6).tolist(),
    }
