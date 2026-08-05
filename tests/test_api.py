"""HTTP contract for the FastAPI endpoints."""

import pytest

from chladni.beams import FreeFreeBeams
from chladni.modes import ModeSet

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_root_metadata(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "Chladni API"
    assert set(data["boundary_conditions"]) == {"free", "clamped"}
    assert "svg" in data["formats"] and "png" in data["formats"]
    assert "sand" in data["styles"] and "field" in data["styles"]
    assert data["mode_counts"]["free"] > 0
    assert data["mode_counts"]["clamped"] > 0


def test_list_free_modes(client):
    r = client.get("/modes", params={"bc": "free"})
    assert r.status_code == 200
    data = r.json()
    assert data["bc"] == "free"
    assert data["count"] == len(data["modes"])
    assert data["freq_min"] == pytest.approx(23.0, abs=1e-6)
    assert data["modes"][0]["frequency"] == pytest.approx(23.0, abs=1e-6)
    assert data["modes"][0]["label"] == [1, 1]


def test_list_clamped_modes(client):
    r = client.get("/modes", params={"bc": "clamped"})
    assert r.status_code == 200
    assert r.json()["bc"] == "clamped"
    assert len(r.json()["modes"]) > 0


def test_list_modes_bad_bc(client):
    assert client.get("/modes", params={"bc": "hinged"}).status_code == 422


def test_single_mode(client):
    r = client.get("/modes/1", params={"bc": "clamped"})
    assert r.status_code == 200
    data = r.json()
    assert data["n_beams"] == 30
    assert len(data["coefficients"]) == 900


def test_single_mode_missing(client):
    assert client.get("/modes/9999", params={"bc": "free"}).status_code == 404


@pytest.mark.parametrize("bad", [0, -1])
def test_single_mode_index_must_be_positive(client, bad):
    assert client.get(f"/modes/{bad}", params={"bc": "free"}).status_code == 422


def test_list_modes_empty_catalog(client, monkeypatch):
    import app.registry as registry

    monkeypatch.setattr(
        registry,
        "_catalogs",
        {"free": ModeSet(bc="free", beams=FreeFreeBeams(), n_beams=32)},
    )
    r = client.get("/modes", params={"bc": "free"})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 0
    assert data["modes"] == []
    assert data["freq_min"] is None
    assert data["freq_max"] is None


def test_render_svg(client):
    r = client.get(
        "/render",
        params={"bc": "clamped", "m": 2, "n": 1, "format": "svg", "resolution": 128},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/svg+xml"
    assert r.text.startswith("<svg")


def test_render_png(client):
    r = client.get(
        "/render",
        params={"bc": "free", "index": 2, "format": "png", "resolution": 128},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == PNG_MAGIC


def test_render_json(client):
    r = client.get(
        "/render",
        params={"bc": "free", "index": 1, "format": "json", "resolution": 64},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["resolution"] == 64
    assert len(data["values"]) == 64


def test_render_requires_address(client):
    assert client.get("/render", params={"bc": "free"}).status_code == 422


def test_render_rejects_ambiguous_address(client):
    r = client.get("/render", params={"bc": "free", "index": 1, "m": 2, "n": 1})
    assert r.status_code == 422
    assert "not both" in r.json()["detail"]


def test_render_partial_pair_with_index_uses_index(client):
    # A lone m (no n) next to an index is not an (m, n) pair: index wins.
    r = client.get("/render", params={"bc": "free", "index": 1, "m": 2})
    assert r.status_code == 200


def test_render_unresolvable_pair(client):
    r = client.get("/render", params={"bc": "free", "m": 2, "n": 1, "format": "svg"})
    assert r.status_code == 404
    assert "available labels" in r.json()["detail"]


def test_render_missing_index(client):
    assert client.get("/render", params={"bc": "clamped", "index": 9999}).status_code == 404


def test_render_resolution_bounds(client):
    assert (
        client.get("/render", params={"bc": "free", "index": 1, "resolution": 8}).status_code
        == 422
    )
    assert (
        client.get("/render", params={"bc": "free", "index": 1, "resolution": 99999}).status_code
        == 422
    )


def test_render_bad_format(client):
    assert (
        client.get("/render", params={"bc": "free", "index": 1, "format": "tiff"}).status_code
        == 422
    )
