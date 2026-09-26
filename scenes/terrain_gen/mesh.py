"""Heightfield to polygon mesh conversion (no USD dependency)."""

from __future__ import annotations

import numpy as np


def heightfield_mesh(
    h: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    skirt: bool = True,
    min_skirt: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quad mesh of the top surface plus optional vertical skirt to z = 0.

    The skirt closes the sides so lidar and wheels see a solid edge instead
    of a floating sheet. Returns points (N, 3) float32, face vertex counts and
    face vertex indices as int32, with counter-clockwise winding from outside.
    """
    ny, nx = h.shape
    X, Y = np.meshgrid(x, y, indexing="xy")
    top = np.stack([X.ravel(), Y.ravel(), h.ravel()], axis=1)
    idx = np.arange(ny * nx).reshape(ny, nx)
    quads = np.stack(
        [idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], axis=-1
    ).reshape(-1, 4)
    points, faces = [top], [quads]

    if skirt:
        loop = np.concatenate(
            [idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]]
        )
        nxt = np.roll(loop, -1)
        bottom = top[loop].copy()
        bottom[:, 2] = 0.0
        b_idx = len(top) + np.arange(len(loop))
        b_nxt = np.roll(b_idx, -1)
        hf = h.ravel()
        tall = np.maximum(hf[loop], hf[nxt]) > min_skirt
        side = np.stack([b_idx, b_nxt, nxt, loop], axis=1)[tall]
        points.append(bottom)
        faces.append(side)

    pts = np.vstack(points).astype(np.float32)
    f = np.vstack(faces)
    counts = np.full(len(f), 4, dtype=np.int32)
    return pts, counts, f.ravel().astype(np.int32)
