"""Check terrain patch footprints against a Nav2 occupancy map."""

from __future__ import annotations

import math
import os

import numpy as np
import yaml


class OccupancyMap:
    def __init__(self, map_yaml: str):
        from PIL import Image

        with open(map_yaml) as f:
            meta = yaml.safe_load(f)
        image = meta["image"]
        if not os.path.isabs(image):
            image = os.path.join(os.path.dirname(map_yaml), image)
        img = np.asarray(Image.open(image).convert("L"), dtype=np.float64)
        if meta.get("negate", 0):
            img = 255.0 - img
        occ = (255.0 - img) / 255.0
        self.free = occ < float(meta.get("free_thresh", 0.196))
        self.resolution = float(meta["resolution"])
        self.origin = (float(meta["origin"][0]), float(meta["origin"][1]))

    def is_free(self, wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
        h, w = self.free.shape
        col = np.floor((wx - self.origin[0]) / self.resolution).astype(int)
        row = h - 1 - np.floor((wy - self.origin[1]) / self.resolution).astype(int)
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        out = np.zeros(wx.shape, dtype=bool)
        out[inside] = self.free[row[inside], col[inside]]
        return out


def footprint_free_fraction(
    occ: OccupancyMap, pose: dict, length: float, width: float, clearance: float
) -> float:
    """Fraction of the footprint (grown by clearance) that is free space.

    Unknown and out-of-map cells count as blocked.
    """
    step = occ.resolution
    hl, hw = length / 2 + clearance, width / 2 + clearance
    lx, ly = np.meshgrid(
        np.arange(-hl, hl + 1e-9, step), np.arange(-hw, hw + 1e-9, step)
    )
    yaw = math.radians(pose.get("yaw_deg", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    wx = pose["x"] + c * lx - s * ly
    wy = pose["y"] + s * lx + c * ly
    return float(occ.is_free(wx, wy).mean())
