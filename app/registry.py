"""Shared solver state.

The Rayleigh-Ritz solves are cached in memory. They are not cheap (roughly
1-2 seconds per boundary condition for a 1024x1024 eigenvalue problem), so
they run once at startup and are reused for the life of the process.
"""

from __future__ import annotations

from chladni.modes import ModeSet
from chladni.solver import PlateSolver, SolverResult

_solver: PlateSolver | None = None
_catalogs: dict[str, ModeSet] = {}


def init() -> None:
    global _solver
    _solver = PlateSolver(quiet=True)
    for bc, result in _solver_cached().items():
        _catalogs[bc] = ModeSet.build(result)


def _solver_cached() -> dict[str, SolverResult]:
    assert _solver is not None, "registry not initialized; run init() first"
    return {bc: _solver.get(bc) for bc in ("free", "clamped")}


def catalog(bc: str) -> ModeSet:
    if bc not in _catalogs:
        raise KeyError(f"catalog for {bc!r} not initialized")
    return _catalogs[bc]


def counts() -> dict[str, int]:
    return {bc: len(cat.modes) for bc, cat in _catalogs.items()}
