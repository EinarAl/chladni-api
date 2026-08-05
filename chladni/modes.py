"""Mode catalog, field evaluation, and (m, n) labeling.

The Ritz solver returns ranked eigenmodes. Plate-theory notation labels a
mode by the number of interior nodal lines along each axis: mode (m, n) is
dominated by the beam-product basis function X_{m-1}(x) X_{n-1}(y), where
beam mode X_i has i interior nodes.

We therefore label each eigenmode by its dominant basis pair (i, j), mapped
to plate notation (m, n) = (i + 1, j + 1), and keep a canonical form
(sorted ascending) so the degenerate twins (m, n) and (n, m) of a symmetric
square plate resolve to the same label. Free-edge eigenmodes have curved
nodal lines, so their labels are approximate; this is documented in the
README.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .beams import ClampedBeams, FreeFreeBeams
from .solver import SolverResult


def evaluate_field(
    beams: ClampedBeams | FreeFreeBeams,
    n_beams: int,
    coefficients: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Evaluate the mode shape on an n x n grid of cell centers in [0, 1]^2."""
    a = coefficients.reshape(n_beams, n_beams, order="F")
    x = (np.arange(resolution) + 0.5) / resolution
    xg = np.array([beams.value(i, x) for i in range(n_beams)])
    return xg.T @ a @ xg


def _dominant_pair(
    coefficients: np.ndarray, n_beams: int
) -> tuple[int, int]:
    a = np.abs(coefficients.reshape(n_beams, n_beams, order="F"))
    i, j = np.unravel_index(int(np.argmax(a)), a.shape)
    return (i, j)


@dataclass
class CatalogedMode:
    index: int
    frequency: float
    eigenvalue: float
    m: int  # dominant basis row index + 1
    n: int  # dominant basis column index + 1
    coefficients: np.ndarray

    @property
    def label(self) -> tuple[int, int]:
        """Canonical plate label, sorted so (m, n) and (n, m) agree."""
        return (min(self.m, self.n), max(self.m, self.n))


@dataclass
class ModeSet:
    bc: str
    beams: ClampedBeams | FreeFreeBeams
    n_beams: int
    modes: list[CatalogedMode] = field(default_factory=list)

    @classmethod
    def build(cls, result: SolverResult) -> "ModeSet":
        beams = FreeFreeBeams() if result.bc == "free" else ClampedBeams()
        catalog: list[CatalogedMode] = []
        for mode in result.modes:
            i, j = _dominant_pair(mode.coefficients, result.n_beams)
            catalog.append(
                CatalogedMode(
                    index=mode.index,
                    frequency=mode.frequency,
                    eigenvalue=mode.eigenvalue,
                    m=i + 1,
                    n=j + 1,
                    coefficients=mode.coefficients,
                )
            )
        return cls(bc=result.bc, beams=beams, n_beams=result.n_beams, modes=catalog)

    def get(self, index: int) -> CatalogedMode | None:
        for mode in self.modes:
            if mode.index == index:
                return mode
        return None

    def resolve(self, m: int, n: int) -> CatalogedMode | None:
        """Return the lowest-frequency mode whose canonical label matches
        (m, n) (the degenerate swap (n, m) matches the same label)."""
        want = (min(m, n), max(m, n))
        for mode in self.modes:
            if mode.label == want:
                return mode
        return None

    def labels(self) -> list[tuple[int, int]]:
        seen: list[tuple[int, int]] = []
        for mode in self.modes:
            if mode.label not in seen:
                seen.append(mode.label)
        return sorted(seen)

    def field(self, index: int, resolution: int) -> np.ndarray:
        mode = self.get(index)
        if mode is None:
            raise KeyError(f"no mode {index} for {self.bc}")
        return evaluate_field(self.beams, self.n_beams, mode.coefficients, resolution)
