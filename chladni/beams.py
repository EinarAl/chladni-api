"""Beam eigenfunctions used as the Rayleigh-Ritz basis.

Both boundary conditions share the same characteristic roots: cosh(b) cos(b) = 1.
Clamped-clamped beams have only elastic modes. Free-free beams add two rigid
body modes (translation and rotation) before the elastic sequence.
"""

from __future__ import annotations

import numpy as np

# First roots of cosh(b)cos(b) = 1 (clamped and free beam elastic modes).
_BETA = [
    4.730040744862704,
    7.853204624095838,
    10.995607838001670,
    14.137165491257348,
    17.278759657399481,
    20.420352245626061,
    23.561944902040455,
    26.703537555508186,
    29.845130209103254,
    32.986722862692819,
    36.128315516282622,
    39.269908169872416,
    42.411500823462205,
    45.553093477051995,
    48.694686130641785,
    51.836278784231588,
    54.977871437821380,
    58.119464091411174,
    61.261056745000967,
    64.402649398590773,
    67.544242052180566,
]


def _beta_for(n: int) -> float:
    """Root index for elastic beam mode n (0-based). Uses the asymptotic
    formula (n + 1.5) * pi beyond the tabulated values."""
    if n < len(_BETA):
        return _BETA[n]
    return (n + 1.5) * np.pi


def _sigma(b: float) -> float:
    ch, sh = np.cosh(b), np.sinh(b)
    c, s = np.cos(b), np.sin(b)
    return (ch - c) / (sh - s)


class ClampedBeams:
    """Clamped-clamped beam eigenfunctions. Mode n has n interior nodes."""

    def value(self, n: int, x: float | np.ndarray) -> float | np.ndarray:
        b = _beta_for(n)
        s = _sigma(b)
        bx = b * x
        return np.cosh(bx) - np.cos(bx) - s * (np.sinh(bx) - np.sin(bx))

    def deriv(self, n: int, x: float | np.ndarray) -> float | np.ndarray:
        b = _beta_for(n)
        s = _sigma(b)
        bx = b * x
        return b * ((np.sinh(bx) + np.sin(bx)) - s * (np.cosh(bx) - np.cos(bx)))

    def deriv2(self, n: int, x: float | np.ndarray) -> float | np.ndarray:
        b = _beta_for(n)
        s = _sigma(b)
        bx = b * x
        return b * b * ((np.cosh(bx) + np.cos(bx)) - s * (np.sinh(bx) - np.sin(bx)))


class FreeFreeBeams:
    """Free-free beam eigenfunctions.

    Mode 0 is rigid-body translation (phi = 1), mode 1 is rigid-body
    rotation (phi = sqrt(3)(2x - 1)); modes n >= 2 are elastic.
    """

    def value(self, n: int, x: float | np.ndarray) -> float | np.ndarray:
        if n == 0:
            return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0
        if n == 1:
            return np.sqrt(3) * (2 * x - 1)
        b = _beta_for(n - 2)
        s = _sigma(b)
        bx = b * x
        return np.cosh(bx) + np.cos(bx) - s * (np.sinh(bx) + np.sin(bx))

    def deriv(self, n: int, x: float | np.ndarray) -> float | np.ndarray:
        if n == 0:
            return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        if n == 1:
            return 2 * np.sqrt(3) * (np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)
        b = _beta_for(n - 2)
        s = _sigma(b)
        bx = b * x
        return b * ((np.sinh(bx) - np.sin(bx)) - s * (np.cosh(bx) + np.cos(bx)))

    def deriv2(self, n: int, x: float | np.ndarray) -> float | np.ndarray:
        if n < 2:
            return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        b = _beta_for(n - 2)
        s = _sigma(b)
        bx = b * x
        return b * b * ((np.cosh(bx) - np.cos(bx)) - s * (np.sinh(bx) - np.sin(bx)))


def make_beams(bc: str) -> ClampedBeams | FreeFreeBeams:
    if bc == "clamped":
        return ClampedBeams()
    if bc == "free":
        return FreeFreeBeams()
    raise ValueError(f"unknown boundary condition: {bc!r}")
