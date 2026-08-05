"""Rayleigh-Ritz eigenmode solver for thin square plates.

Computes Kirchhoff-Love plate eigenmodes under either free-edge or
clamped boundary conditions using beam eigenfunction products as a basis.
This is the same numerical approach Ritz published in 1909: assemble the
stiffness and mass matrices from the biharmonic operator, then solve the
generalized eigenvalue problem K v = lambda M v.
"""

from .beams import ClampedBeams, FreeFreeBeams
from .modes import ModeSet, evaluate_field
from .solver import PlateSolver, solve_plate

__all__ = [
    "ClampedBeams",
    "FreeFreeBeams",
    "PlateSolver",
    "solve_plate",
    "ModeSet",
    "evaluate_field",
]
