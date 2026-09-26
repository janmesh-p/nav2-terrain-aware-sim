import math

import numpy as np
import pytest

from terrain_gen.mesh import heightfield_mesh
from terrain_gen.metrics import roughness_m, slope_deg
from terrain_gen.surfaces import (
    PatchGrid,
    build_heightfield,
    iso8608_band,
    iso8608_band_rms,
    iso8608_surface,
    ramp,
)

RES = 0.05


def test_ramp_grade_and_rise():
    g = PatchGrid(6.0, 2.0, RES)
    h = ramp(g, grade_deg=7.0, plateau_length=1.5)
    run = 0.5 * (6.0 - 1.5)
    assert h.max() == pytest.approx(run * math.tan(math.radians(7.0)), rel=1e-6)
    s = slope_deg(h, RES)
    x, _ = g.axes()
    incline = (np.abs(x) > 0.75 + 2 * RES) & (np.abs(x) < 3.0 - 2 * RES)
    assert np.allclose(s[:, incline], 7.0, atol=0.05)


def test_iso8608_rms_matches_class():
    g = PatchGrid(4.0, 3.0, RES)
    rng = np.random.default_rng(0)
    h = iso8608_surface(g, rng, road_class="F")
    target = iso8608_band_rms("F", *iso8608_band(g))
    assert h.std() == pytest.approx(target, rel=1e-6)


def test_iso8608_class_ordering():
    g = PatchGrid(4.0, 4.0, RES)
    r = [
        roughness_m(iso8608_surface(g, np.random.default_rng(1), c), RES, 0.4).mean()
        for c in "CDEF"
    ]
    assert all(a < b for a, b in zip(r, r[1:]))


def test_plane_has_no_roughness():
    g = PatchGrid(3.0, 3.0, RES)
    X, Y = g.mesh()
    plane = 0.1 * X + 0.05 * Y
    assert roughness_m(plane, RES, 0.4).max() < 1e-6


def test_tapered_edges_are_at_floor():
    g = PatchGrid(3.0, 3.0, RES)
    rng = np.random.default_rng(3)
    h = build_heightfield("iso8608", g, {"road_class": "E", "edge_taper": 0.3}, rng)
    border = np.concatenate([h[0], h[-1], h[:, 0], h[:, -1]])
    assert np.abs(border).max() < 1e-9
    assert h.min() >= 0.0


def test_mesh_topology_and_winding():
    g = PatchGrid(1.0, 0.5, 0.1)
    h = np.full(g.shape, 0.2)
    x, y = g.axes()
    pts, counts, idx = heightfield_mesh(h, x, y)
    ny, nx = g.shape
    n_top_faces = (ny - 1) * (nx - 1)
    n_loop = 2 * (nx + ny) - 4
    assert len(counts) == n_top_faces + n_loop
    quad = pts[idx[:4]]
    normal = np.cross(quad[1] - quad[0], quad[2] - quad[1])
    assert normal[2] > 0
    side = pts[idx[4 * n_top_faces: 4 * n_top_faces + 4]]
    side_normal = np.cross(side[1] - side[0], side[2] - side[1])
    assert side_normal[1] < 0  # first skirt face is on the -y edge, facing out


def test_same_seed_same_surface():
    g = PatchGrid(2.0, 2.0, RES)
    a = iso8608_surface(g, np.random.default_rng(42), "D")
    b = iso8608_surface(g, np.random.default_rng(42), "D")
    assert np.array_equal(a, b)
