"""Rayleigh-Ritz plate solver.

Solves the Kirchhoff-Love biharmonic eigenvalue problem for a square plate,

    D * laplacian^2(W) = rho * h * omega^2 * W,

using products of beam eigenfunctions as basis functions. The mode shape is

    W(x, y) = sum_ij a_ij X_i(x) X_j(y),

and the generalized eigenvalue problem K v = lambda M v is assembled from
the bending energy of the plate. Free-edge boundary conditions are satisfied
by free-free beam functions plus a rigid center constraint W(1/2, 1/2) = 0.
Clamped boundary conditions are satisfied by clamped-clamped beam functions.

Coefficient vectors are stored flat with index p = i + j * N, matching the
binary format produced by the chladni-tuner precompute scripts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from .beams import make_beams

NU = 0.3          # Poisson's ratio (typical metals)
F11 = 23.0        # fundamental frequency calibration, Hz
N_INT = 4000      # Simpson quadrature points per axis
GRID_SIZE = 128   # default evaluation grid
F_MIN = 15.0      # valid-mode frequency floor, Hz
F_MAX = 2000.0    # valid-mode frequency ceiling, Hz

BOUNDARY_CONDITIONS = ("free", "clamped")


@dataclass
class Mode:
    index: int
    frequency: float
    eigenvalue: float
    coefficients: np.ndarray  # flat, index = i + j * N


@dataclass
class SolverResult:
    bc: str
    n_beams: int
    freq_min: float
    freq_max: float
    modes: list[Mode] = field(default_factory=list)


def _simpson_axis(fx: np.ndarray, fy: np.ndarray, weights: np.ndarray, scale: float) -> float:
    """Simpson rule applied over one axis: int fx(s) fy(s) ds on [0, 1]."""
    return float(np.dot(fx * fy, weights) * scale)


def solve_plate(bc: str, n_elastic: int = 30, quiet: bool = False) -> SolverResult:
    """Run the Rayleigh-Ritz solve for one boundary condition."""
    if bc not in BOUNDARY_CONDITIONS:
        raise ValueError(f"bc must be one of {BOUNDARY_CONDITIONS}, got {bc!r}")

    t0 = time.perf_counter()
    beams = make_beams(bc)
    n_rigid = 0 if bc == "clamped" else 2
    n_beams = n_elastic + n_rigid

    # Beam functions sampled for quadrature.
    xs = np.linspace(0.0, 1.0, N_INT + 1)
    phi = np.array([beams.value(i, xs) for i in range(n_beams)])
    phi2 = np.array([beams.deriv2(i, xs) for i in range(n_beams)])
    phi1 = np.array([beams.deriv(i, xs) for i in range(n_beams)])

    weights = np.ones(N_INT + 1)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2
    scale = (1.0 / N_INT) / 3.0

    M = np.zeros((n_beams, n_beams))
    A = np.zeros((n_beams, n_beams))
    B = np.zeros((n_beams, n_beams))
    Ip = np.zeros((n_beams, n_beams))
    for i in range(n_beams):
        for k in range(n_beams):
            M[i, k] = _simpson_axis(phi[i], phi[k], weights, scale)
            A[i, k] = _simpson_axis(phi2[i], phi2[k], weights, scale)
            B[i, k] = _simpson_axis(phi2[i], phi[k], weights, scale)
            Ip[i, k] = _simpson_axis(phi1[i], phi1[k], weights, scale)

    # Assemble the 4th-order stiffness tensor
    #   K_ijkl = A_ik M_jl + M_ik A_jl
    #          + nu (B_ik B_lj + B_ki B_jl)
    #          + 2(1 - nu) Ip_ik Ip_jl
    # with basis index pairs p = i + j*N, q = k + l*N.
    n2 = n_beams * n_beams
    K4 = (
        np.einsum("ik,jl->ijkl", A, M, optimize=True)
        + np.einsum("ik,jl->ijkl", M, A, optimize=True)
        + NU * (np.einsum("ik,jl->ijkl", B, B.T, optimize=True)
                + np.einsum("ik,jl->ijkl", B.T, B, optimize=True))
        + 2.0 * (1.0 - NU) * np.einsum("ik,jl->ijkl", Ip, Ip, optimize=True)
    )
    M4 = np.einsum("ik,jl->ijkl", M, M, optimize=True)

    K = K4.transpose(1, 0, 3, 2).reshape(n2, n2)
    Mbig = M4.transpose(1, 0, 3, 2).reshape(n2, n2)

    # Center constraint W(1/2, 1/2) = 0 for the free-edge plate.
    if bc == "free":
        cent = np.array([beams.value(i, 0.5) for i in range(n_beams)])
        cvec = np.einsum("i,j->ij", cent, cent).ravel(order="F")  # index i + j*N
        pivot = int(np.argmax(np.abs(cvec)))
        keep = np.ones(n2, dtype=bool)
        keep[pivot] = False
        full_idx = np.arange(n2)
        red_idx = full_idx[keep]
        s = -cvec / cvec[pivot]
        k_red = (
            K[np.ix_(red_idx, red_idx)]
            + np.outer(K[red_idx, pivot], s[red_idx])
            + np.outer(s[red_idx], K[pivot, red_idx])
            + K[pivot, pivot] * np.outer(s[red_idx], s[red_idx])
        )
        m_red = (
            Mbig[np.ix_(red_idx, red_idx)]
            + np.outer(Mbig[red_idx, pivot], s[red_idx])
            + np.outer(s[red_idx], Mbig[pivot, red_idx])
            + Mbig[pivot, pivot] * np.outer(s[red_idx], s[red_idx])
        )
    else:
        red_idx = np.arange(n2)
        k_red = K
        m_red = Mbig
        pivot = -1

    eigvals, eigvecs = linalg.eigh(k_red, m_red)
    eigvals = np.maximum(eigvals, 0.0)

    # Frequencies calibrated so the fundamental lands on F11.
    first = int(np.argmax(eigvals > 1e-6))
    f0 = eigvals[first]
    freqs = F11 * np.sqrt(eigvals / f0)

    valid = np.where((freqs >= F_MIN) & (freqs <= F_MAX))[0]

    def expand(v_red: np.ndarray) -> np.ndarray:
        v = np.zeros(n2)
        v[red_idx] = v_red
        if pivot >= 0:
            v[pivot] = -float(np.dot(cvec, v)) / cvec[pivot]
        if np.sum(v) < 0:
            v = -v
        return v

    modes: list[Mode] = []
    for rank, k in enumerate(valid):
        modes.append(
            Mode(
                index=rank + 1,
                frequency=float(freqs[k]),
                eigenvalue=float(eigvals[k]),
                coefficients=expand(eigvecs[:, k]),
            )
        )

    elapsed = time.perf_counter() - t0
    if not quiet:
        print(
            f"  [{bc}] {n2}x{n2} solve in {elapsed:.1f}s, "
            f"{len(modes)} modes in {F_MIN:.0f}-{F_MAX:.0f} Hz"
        )
    return SolverResult(
        bc=bc,
        n_beams=n_beams,
        freq_min=float(freqs[valid[0]]) if len(valid) else 0.0,
        freq_max=float(freqs[valid[-1]]) if len(valid) else 0.0,
        modes=modes,
    )


class PlateSolver:
    """Solver with cached results per boundary condition."""

    def __init__(self, n_elastic: int = 30, quiet: bool = False):
        self.n_elastic = n_elastic
        self._cache: dict[str, SolverResult] = {}
        self._quiet = quiet

    def get(self, bc: str) -> SolverResult:
        if bc not in self._cache:
            self._cache[bc] = solve_plate(bc, n_elastic=self.n_elastic, quiet=self._quiet)
        return self._cache[bc]

    def precompute(self) -> None:
        for bc in BOUNDARY_CONDITIONS:
            self.get(bc)
