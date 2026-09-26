"""Traversability metrics computed from a heightfield."""

from __future__ import annotations

import numpy as np


def slope_deg(h: np.ndarray, resolution: float) -> np.ndarray:
    gy, gx = np.gradient(h, resolution)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def _box_mean(a: np.ndarray, k: int) -> np.ndarray:
    """k x k moving average via an integral image, edge padded."""
    r = k // 2
    p = np.pad(a, r, mode="edge")
    s = np.zeros((p.shape[0] + 1, p.shape[1] + 1))
    s[1:, 1:] = p.cumsum(0).cumsum(1)
    return (s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]) / (k * k)


def roughness_m(h: np.ndarray, resolution: float, window: float) -> np.ndarray:
    """Local RMS residual after a least-squares plane fit in each window.

    Removing the plane keeps a steady slope from reading as roughness.
    """
    k = max(3, int(round(window / resolution)) | 1)
    ny, nx = h.shape
    X, Y = np.meshgrid(
        np.arange(nx) * resolution, np.arange(ny) * resolution, indexing="xy"
    )
    mh, mx, my = _box_mean(h, k), _box_mean(X, k), _box_mean(Y, k)
    var_h = _box_mean(h * h, k) - mh**2
    var_x = _box_mean(X * X, k) - mx**2
    var_y = _box_mean(Y * Y, k) - my**2
    cov_hx = _box_mean(h * X, k) - mh * mx
    cov_hy = _box_mean(h * Y, k) - mh * my
    eps = 1e-12
    resid = (
        var_h
        - cov_hx**2 / np.maximum(var_x, eps)
        - cov_hy**2 / np.maximum(var_y, eps)
    )
    return np.sqrt(np.clip(resid, 0.0, None))


def summarize(h: np.ndarray, slope: np.ndarray, rough: np.ndarray) -> dict:
    return {
        "max_height_m": float(h.max()),
        "rms_height_m": float(h.std()),
        "max_slope_deg": float(slope.max()),
        "p95_slope_deg": float(np.percentile(slope, 95)),
        "mean_roughness_m": float(rough.mean()),
        "p95_roughness_m": float(np.percentile(rough, 95)),
    }
