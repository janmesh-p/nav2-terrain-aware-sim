"""Heightfield primitives for navigation test terrain.

Every surface is sampled on a regular grid in a patch-local frame.
x runs along the patch length, y across it, z is up. Units are meters.
Arrays are indexed [row, col] = [y, x].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ISO 8608 road roughness classes. Geometric mean of the displacement PSD
# Gd(n0) at reference spatial frequency n0 = 0.1 cycles/m, in m^3.
ISO8608_GD_N0 = {
    "A": 16e-6,
    "B": 64e-6,
    "C": 256e-6,
    "D": 1024e-6,
    "E": 4096e-6,
    "F": 16384e-6,
    "G": 65536e-6,
    "H": 262144e-6,
}
ISO8608_N0 = 0.1


@dataclass(frozen=True)
class PatchGrid:
    length: float
    width: float
    resolution: float

    def __post_init__(self):
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if min(self.length, self.width) <= 2 * self.resolution:
            raise ValueError("patch must span more than two grid cells")

    @property
    def shape(self) -> tuple[int, int]:
        nx = int(round(self.length / self.resolution)) + 1
        ny = int(round(self.width / self.resolution)) + 1
        return ny, nx

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = self.shape
        x = np.linspace(-self.length / 2, self.length / 2, nx)
        y = np.linspace(-self.width / 2, self.width / 2, ny)
        return x, y

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.axes()
        return np.meshgrid(x, y, indexing="xy")


def ramp(grid: PatchGrid, grade_deg: float, plateau_length: float) -> np.ndarray:
    """Symmetric up-plateau-down ramp along x with a fixed grade."""
    if not 0.0 < grade_deg < 45.0:
        raise ValueError("grade_deg must be in (0, 45)")
    run = 0.5 * (grid.length - plateau_length)
    if run <= grid.resolution:
        raise ValueError("plateau leaves no room for the incline")
    X, _ = grid.mesh()
    rise_per_m = np.tan(np.radians(grade_deg))
    dist_from_end = grid.length / 2 - np.abs(X)
    return np.minimum(dist_from_end * rise_per_m, run * rise_per_m)


def gaussian_bumps(
    grid: PatchGrid,
    rng: np.random.Generator,
    count: int,
    height_range: tuple[float, float],
    sigma_range: tuple[float, float],
) -> np.ndarray:
    """Superposition of isotropic Gaussian bumps with random size and height."""
    h_lo, h_hi = height_range
    s_lo, s_hi = sigma_range
    margin = 2.0 * s_hi
    if 2 * margin >= min(grid.length, grid.width):
        raise ValueError("sigma_range too large for patch size")
    X, Y = grid.mesh()
    cx = rng.uniform(-grid.length / 2 + margin, grid.length / 2 - margin, count)
    cy = rng.uniform(-grid.width / 2 + margin, grid.width / 2 - margin, count)
    amp = rng.uniform(h_lo, h_hi, count)
    sig = rng.uniform(s_lo, s_hi, count)
    h = np.zeros_like(X)
    for a, s, x0, y0 in zip(amp, sig, cx, cy):
        h += a * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * s * s))
    return h


def iso8608_band_rms(
    road_class: str, n_min: float, n_max: float, waviness: float = 2.0
) -> float:
    """RMS height of an ISO 8608 profile integrated over [n_min, n_max] cycles/m."""
    gd = ISO8608_GD_N0[road_class.upper()]
    w = waviness
    if abs(w - 1.0) < 1e-9:
        var = gd * ISO8608_N0 * np.log(n_max / n_min)
    else:
        var = gd * ISO8608_N0**w * (n_max ** (1 - w) - n_min ** (1 - w)) / (1 - w)
    return float(np.sqrt(var))


def iso8608_band(grid: PatchGrid) -> tuple[float, float]:
    """Spatial frequency band the grid can resolve: patch size to Nyquist."""
    return 1.0 / max(grid.length, grid.width), 0.5 / grid.resolution


def iso8608_surface(
    grid: PatchGrid,
    rng: np.random.Generator,
    road_class: str,
    waviness: float = 2.0,
) -> np.ndarray:
    """Isotropic random surface with an ISO 8608 spectral shape.

    White noise is filtered in the Fourier domain so the 2D PSD falls off as
    n^-(w+1), which gives 1D profiles with the ISO exponent w. Amplitude is
    calibrated to the class RMS over the band the grid resolves.
    """
    ny, nx = grid.shape
    n_min, n_max = iso8608_band(grid)
    kx = np.fft.fftfreq(nx, d=grid.resolution)
    ky = np.fft.fftfreq(ny, d=grid.resolution)
    n = np.hypot(*np.meshgrid(kx, ky, indexing="xy"))
    band = (n >= n_min) & (n <= n_max)
    shaping = np.zeros_like(n)
    shaping[band] = n[band] ** (-(waviness + 1.0) / 2.0)
    white = rng.standard_normal((ny, nx))
    h = np.fft.ifft2(np.fft.fft2(white) * shaping).real
    h -= h.mean()
    std = h.std()
    if std == 0.0:
        raise ValueError("degenerate spectrum, check resolution and patch size")
    return h * (iso8608_band_rms(road_class, n_min, n_max, waviness) / std)


def edge_taper(grid: PatchGrid, width: float) -> np.ndarray:
    """Raised-cosine window that brings heights to zero at the patch border."""
    X, Y = grid.mesh()
    if width <= 0:
        return np.ones_like(X)
    d = np.minimum(grid.length / 2 - np.abs(X), grid.width / 2 - np.abs(Y))
    s = np.clip(d / width, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * s)


def build_heightfield(
    patch_type: str, grid: PatchGrid, params: dict, rng: np.random.Generator
) -> np.ndarray:
    params = dict(params)
    taper = float(params.pop("edge_taper", 0.0))
    if patch_type == "ramp":
        if taper:
            raise ValueError("ramp ends are already at floor height, drop edge_taper")
        return ramp(grid, **params)
    if patch_type == "bumps":
        h = gaussian_bumps(grid, rng, **params)
    elif patch_type == "iso8608":
        h = iso8608_surface(grid, rng, **params)
        h = h - h.min()  # keep the surface above the floor collider
    else:
        raise ValueError(f"unknown patch type '{patch_type}'")
    return h * edge_taper(grid, taper)
