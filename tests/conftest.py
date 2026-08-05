"""Shared fixtures for the chladni-api test suite.

The Rayleigh-Ritz solves are cached for the session: each boundary condition
is solved once (~1-2 s) and reused by every test. The FastAPI TestClient
starts the real app, whose lifespan runs the registry solve at startup.
"""

import pytest
from fastapi.testclient import TestClient

from chladni.modes import ModeSet
from chladni.solver import PlateSolver, SolverResult


@pytest.fixture(scope="session")
def solver() -> PlateSolver:
    return PlateSolver(quiet=True)


@pytest.fixture(scope="session")
def results(solver: PlateSolver) -> dict[str, SolverResult]:
    return {bc: solver.get(bc) for bc in ("free", "clamped")}


@pytest.fixture(scope="session")
def catalogs(results: dict[str, SolverResult]) -> dict[str, ModeSet]:
    return {bc: ModeSet.build(res) for bc, res in results.items()}


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c
