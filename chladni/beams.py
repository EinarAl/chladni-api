"""Beam eigenfunctions used as the Rayleigh-Ritz basis.

Both boundary conditions share the same characteristic roots: cosh(b) cos(b) = 1.
Clamped-clamped beams have only elastic modes. Free-free beams add two rigid
body modes (translation and rotation) before the elastic sequence.

The elastic modes are evaluated through the cancellation-free combinations

    T = cosh(bx) - sigma*sinh(bx)
    U = sinh(bx) - sigma*cosh(bx),   sigma = (cosh(b)-cos(b))/(sinh(b)-sin(b))

rather than raw hyperbolics. For the highest roots (b up to ~95) raw cosh(bx)
reaches ~1e40 while T stays O(1); whether those lost digits survive depends on
the platform math library, and Linux runners with different libm versions have
silently produced garbage stiffness matrices here (caught by the mode-count
guards in test_solver). Every factor below is O(1) by construction.
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


def _beam_params(b: float) -> tuple[float, float, float]:
    """Return (q, sigma, d_den) for characteristic root b without overflow.

    With q = e^{-b}, dividing numerator and denominator of
    sigma = (cosh(b)-cos(b))/(sinh(b)-sin(b)) by cosh(b) gives

        sigma = (1 - X) / D,   X = 2q cos(b)/(1+q^2),
                                D = tanh(b) - 2q sin(b)/(1+q^2).

    Every piece is O(1) and nothing cancels catastrophically. d_den is
    returned because (1 - sigma) feeds the scaled basis combinations:

        1 - sigma = q (cos(b) - sin(b) - q) / ((1+q^2) D)
    """
    q = np.exp(-b)
    two_q = 2.0 * q
    x_ratio = two_q * np.cos(b) / (1.0 + q * q)
    d_den = (1.0 - q * q) / (1.0 + q * q) - two_q * np.sin(b) / (1.0 + q * q)
    return q, (1.0 - x_ratio) / d_den, d_den


class ClampedBeams:
    """Clamped-clamped beam eigenfunctions. Mode n has n interior nodes."""

    def _parts(self, n: int, x: float | np.ndarray):
        """Return (T, U, sigma, b, bx) at x for elastic mode n."""
        b = _beta_for(n)
        q, s, d_den = _beam_params(b)
        bx = b * x
        # P = e^(bx) * (1-sigma)/2, with 1-sigma assembled from exact ratio
        # differences so the e^(bx) factor meets an O(q) term instead of
        # cancelling against a near-equal giant:
        #   P = e^(b(x-1)) * [cos(b)(1-q^2) - sin(b)(1+q^2)]
        #       / ((1+q^2)(1 - q^2 - 2q sin(b)))
        head = (
            np.exp(b * (x - 1.0))
            * (np.cos(b) - np.sin(b) - q)
            / ((1.0 + q * q) * d_den)
        )
        tail = np.exp(-bx) * (1.0 + s) / 2.0
        return head + tail, head - tail, s, b, bx

    def value(self, n: int, x: float | np.ndarray) -> float | np.ndarray:
        big_t, _, s, _, bx = self._parts(n, x)
        return big_t - np.cos(bx) + s * np.sin(bx)

    def deriv(self, n: int, x: float | np.ndarray) -> float | np.ndarray:
        _, big_u, s, b, bx = self._parts(n, x)
        return b * (big_u + np.sin(bx) + s * np.cos(bx))

    def deriv2(self, n: int, x: float | np.ndarray) -> float | np.ndarray:
        big_t, _, s, b, bx = self._parts(n, x)
        return b * b * (big_t + np.cos(bx) + s * np.sin(bx))


class FreeFreeBeams:
    """Free-free beam eigenfunctions.

    Mode 0 is rigid-body translation (phi = 1), mode 1 is rigid-body
    rotation (phi = sqrt(3)(2x - 1)); modes n >= 2 are elastic and share
    the clamped stable evaluation with index shift n - 2.
    """

    def _parts(self, n: int, x: float | np.ndarray):
        """Return (T, U, sigma, b, bx) at x for elastic mode n - 2."""
        b = _beta_for(n - 2)
        q, s, d_den = _beam_params(b)
        bx = b * x
        head = (
            np.exp(b * (x - 1.0))
            * (np.cos(b) - np.sin(b) - q)
            / ((1.0 + q * q) * d_den)
        )
        tail = np.exp(-bx) * (1.0 + s) / 2.0
        return head + tail, head - tail, s, b, bx

    def value(self, n: int, x: float | np.ndarray) -> float | np.ndarray:
        if n == 0:
            return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0
        if n == 1:
            return np.sqrt(3) * (2 * x - 1)
        big_t, _, s, _, bx = self._parts(n, x)
        return big_t + np.cos(bx) - s * np.sin(bx)

    def deriv(self, n: int, x: float | np.ndarray) -> float | np.ndarray:
        if n == 0:
            return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        if n == 1:
            return 2 * np.sqrt(3) * (np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)
        _, big_u, s, b, bx = self._parts(n, x)
        return b * (big_u - np.sin(bx) - s * np.cos(bx))

    def deriv2(self, n: int, x: float | np.ndarray) -> float | np.ndarray:
        if n < 2:
            return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        big_t, _, s, b, bx = self._parts(n, x)
        return b * b * (big_t - np.cos(bx) + s * np.sin(bx))


def make_beams(bc: str) -> ClampedBeams | FreeFreeBeams:
    if bc == "clamped":
        return ClampedBeams()
    if bc == "free":
        return FreeFreeBeams()
    raise ValueError(f"unknown boundary condition: {bc!r}")
