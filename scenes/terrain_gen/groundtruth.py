"""World-frame ground truth rasters for evaluating terrain perception."""

from __future__ import annotations

import math

import numpy as np


def _corners(pose: dict, length: float, width: float) -> np.ndarray:
    yaw = math.radians(pose.get("yaw_deg", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array(
        [[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float
    ) * [length / 2, width / 2]
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + [pose["x"], pose["y"]]


def rasterize(patches: list, resolution: float) -> tuple[dict, int]:
    """Nearest-neighbour resample of every patch into one world grid.

    Row 0 is the minimum y. Cells outside all patches hold NaN (patch_id -1).
    Returns the raster dict and the number of cells claimed by two patches.
    """
    corners = np.vstack([_corners(p.pose, p.length, p.width) for p in patches])
    xmin, ymin = np.floor(corners.min(0) / resolution) * resolution
    xmax, ymax = np.ceil(corners.max(0) / resolution) * resolution
    xs = np.arange(xmin, xmax + 0.5 * resolution, resolution)
    ys = np.arange(ymin, ymax + 0.5 * resolution, resolution)
    WX, WY = np.meshgrid(xs, ys, indexing="xy")

    height = np.full(WX.shape, np.nan)
    slope = np.full(WX.shape, np.nan)
    rough = np.full(WX.shape, np.nan)
    pid = np.full(WX.shape, -1, dtype=np.int16)
    overlaps = 0

    for i, p in enumerate(patches):
        yaw = math.radians(p.pose.get("yaw_deg", 0.0))
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = WX - p.pose["x"], WY - p.pose["y"]
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        inside = (np.abs(lx) <= p.length / 2) & (np.abs(ly) <= p.width / 2)
        col = np.rint((lx - p.x[0]) / (p.x[1] - p.x[0])).astype(int)
        row = np.rint((ly - p.y[0]) / (p.y[1] - p.y[0])).astype(int)
        col = np.clip(col, 0, len(p.x) - 1)
        row = np.clip(row, 0, len(p.y) - 1)
        overlaps += int((inside & (pid >= 0)).sum())
        r, cc = row[inside], col[inside]
        height[inside] = p.height[r, cc]
        slope[inside] = p.slope[r, cc]
        rough[inside] = p.roughness[r, cc]
        pid[inside] = i

    raster = {
        "origin_xy": np.array([xs[0], ys[0]]),
        "resolution": np.float64(resolution),
        "height_m": height,
        "slope_deg": slope,
        "roughness_m": rough,
        "patch_id": pid,
    }
    return raster, overlaps
