"""FastAPI application for the Chladni mode API.

Endpoints:
  GET /                  service metadata
  GET /modes             list modes for a boundary condition
  GET /modes/{index}     single mode, including its coefficient vector
  GET /render            a mode rendered as SVG, PNG, or raw JSON grid

Modes are addressed either by their ranked index (honest solver output) or
by the plate-theory pair (m, n), which resolves to the mode whose dominant
basis character matches. Boundary condition selects free-edge or clamped
simulation.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import app.registry as registry
from chladni import render
from chladni.solver import BOUNDARY_CONDITIONS

BC = Literal["free", "clamped"]
FORMAT = Literal["svg", "png", "json"]
STYLE = Literal["sand", "field"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    registry.init()
    yield


app = FastAPI(
    title="Chladni API",
    version="0.1.0",
    description=(
        "Rayleigh-Ritz eigenmodes of a thin square plate under free-edge or "
        "clamped boundary conditions, rendered as SVG or PNG."
    ),
    lifespan=lifespan,
)


# ── Response models ────────────────────────────────────────────────────────


class ModeSummary(BaseModel):
    index: int = Field(description="Ranked mode index (1-based, ascending frequency)")
    frequency: float = Field(description="Eigenfrequency in Hz (F11=23 Hz reference)")
    label: tuple[int, int] = Field(description="Canonical (m, n) label used for resolution")
    dominant: tuple[int, int] = Field(description="Raw dominant basis pair (m, n)")
    eigenvalue: float


class ModesResponse(BaseModel):
    bc: BC
    count: int
    freq_min: float
    freq_max: float
    modes: list[ModeSummary]


class ModeDetail(ModeSummary):
    coefficients: list[float] = Field(
        description="Flat coefficient vector, index = i + j * N"
    )
    n_beams: int


class ServiceInfo(BaseModel):
    name: str = "Chladni API"
    version: str = "0.1.0"
    boundary_conditions: list[str] = list(BOUNDARY_CONDITIONS)
    formats: list[str] = ["svg", "png", "json"]
    styles: list[str] = ["sand", "field"]
    resolution: tuple[int, int] = (render.MIN_RESOLUTION, render.MAX_RESOLUTION)
    mode_counts: dict[str, int] = {}


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/", response_model=ServiceInfo)
def root() -> ServiceInfo:
    return ServiceInfo(mode_counts=registry.counts())


@app.get("/modes", response_model=ModesResponse)
def list_modes(bc: BC = Query(default="free")) -> ModesResponse:
    cat = registry.catalog(bc)
    return ModesResponse(
        bc=bc,
        count=len(cat.modes),
        freq_min=cat.modes[0].frequency,
        freq_max=cat.modes[-1].frequency,
        modes=[
            ModeSummary(
                index=mode.index,
                frequency=mode.frequency,
                label=mode.label,
                dominant=(mode.m, mode.n),
                eigenvalue=mode.eigenvalue,
            )
            for mode in cat.modes
        ],
    )


@app.get("/modes/{index}", response_model=ModeDetail)
def get_mode(index: int, bc: BC = Query(default="free")) -> ModeDetail:
    cat = registry.catalog(bc)
    mode = cat.get(index)
    if mode is None:
        raise HTTPException(
            status_code=404,
            detail=f"no mode {index} for {bc}; available 1..{len(cat.modes)}",
        )
    return ModeDetail(
        index=mode.index,
        frequency=mode.frequency,
        label=mode.label,
        dominant=(mode.m, mode.n),
        eigenvalue=mode.eigenvalue,
        coefficients=[round(float(c), 8) for c in mode.coefficients],
        n_beams=cat.n_beams,
    )


def _resolve_mode(cat, index: int | None, m: int | None, n: int | None):
    if index is not None:
        mode = cat.get(index)
        if mode is None:
            raise HTTPException(
                status_code=404,
                detail=f"no mode {index} for {cat.bc}; available 1..{len(cat.modes)}",
            )
        return mode
    if m is not None and n is not None:
        mode = cat.resolve(m, n)
        if mode is None:
            labels = ", ".join(f"({a},{b})" for a, b in cat.labels())
            raise HTTPException(
                status_code=404,
                detail=f"no mode matches (m,n)=({m},{n}) for {cat.bc}; "
                f"available labels: {labels}",
            )
        return mode
    raise HTTPException(
        status_code=422,
        detail="provide exactly one of: index, or both m and n",
    )


@app.get("/render")
def render_mode(
    bc: BC = Query(default="free"),
    index: Annotated[int | None, Query(ge=1)] = None,
    m: Annotated[int | None, Query(ge=1, description="plate mode index m")] = None,
    n: Annotated[int | None, Query(ge=1, description="plate mode index n")] = None,
    format: FORMAT = Query(default="svg"),
    style: STYLE = Query(default="sand"),
    resolution: Annotated[int, Query(ge=render.MIN_RESOLUTION, le=render.MAX_RESOLUTION)] = 512,
):
    cat = registry.catalog(bc)
    mode = _resolve_mode(cat, index, m, n)

    if format == "json":
        return JSONResponse(render.render_grid(cat, mode.index, resolution))

    if format == "svg":
        return Response(
            content=render.render_svg(cat, mode.index, resolution, style),
            media_type="image/svg+xml",
        )

    return Response(
        content=render.render_png(cat, mode.index, resolution, style),
        media_type="image/png",
    )
