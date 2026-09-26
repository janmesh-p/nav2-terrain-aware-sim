"""Config loading, terrain generation and ground truth export."""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import zlib
from dataclasses import dataclass, field

import numpy as np
import yaml

from . import __version__
from .groundtruth import rasterize
from .metrics import roughness_m, slope_deg, summarize
from .placement import OccupancyMap, footprint_free_fraction
from .surfaces import PatchGrid, build_heightfield, iso8608_band, iso8608_band_rms

PATCH_TYPES = {"ramp", "bumps", "iso8608"}


@dataclass
class PatchResult:
    name: str
    type: str
    material: str
    pose: dict
    length: float
    width: float
    x: np.ndarray
    y: np.ndarray
    height: np.ndarray
    slope: np.ndarray
    roughness: np.ndarray
    summary: dict


@dataclass
class TerrainResult:
    config: dict
    patches: list
    raster: dict
    warnings: list = field(default_factory=list)


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    if cfg.get("nav2_map"):
        cfg["nav2_map"] = os.path.expanduser(cfg["nav2_map"])
    return cfg


def _validate(cfg: dict) -> None:
    for key in ("seed", "resolution", "root_prim", "materials", "patches"):
        if key not in cfg:
            raise ValueError(f"config missing '{key}'")
    names = set()
    for p in cfg["patches"]:
        for key in ("name", "type", "pose", "size", "material"):
            if key not in p:
                raise ValueError(f"patch missing '{key}': {p}")
        if p["name"] in names:
            raise ValueError(f"duplicate patch name '{p['name']}'")
        names.add(p["name"])
        if p["type"] not in PATCH_TYPES:
            raise ValueError(f"patch '{p['name']}': unknown type '{p['type']}'")
        if p["material"] not in cfg["materials"]:
            raise ValueError(f"patch '{p['name']}': unknown material '{p['material']}'")


def _patch_rng(seed: int, name: str) -> np.random.Generator:
    # Seeded per patch name so adding or reordering patches does not
    # change the others.
    return np.random.default_rng([int(seed), zlib.crc32(name.encode())])


def generate(cfg: dict) -> TerrainResult:
    res = float(cfg["resolution"])
    window = float(cfg.get("roughness_window", 0.4))
    warnings = []

    occ = None
    map_yaml = cfg.get("nav2_map")
    if map_yaml and os.path.exists(map_yaml):
        occ = OccupancyMap(map_yaml)
    elif map_yaml:
        warnings.append(f"nav2_map not found, placement unchecked: {map_yaml}")

    patches = []
    for spec in cfg["patches"]:
        grid = PatchGrid(float(spec["size"]["length"]), float(spec["size"]["width"]), res)
        params = dict(spec.get("params", {}))
        h = build_heightfield(spec["type"], grid, params, _patch_rng(cfg["seed"], spec["name"]))
        s = slope_deg(h, res)
        r = roughness_m(h, res, window)
        x, y = grid.axes()
        summary = summarize(h, s, r)

        if spec["type"] == "iso8608":
            n_min, n_max = iso8608_band(grid)
            summary["iso8608_class"] = params["road_class"]
            summary["iso8608_band_rms_m"] = iso8608_band_rms(
                params["road_class"], n_min, n_max, params.get("waviness", 2.0)
            )

        if occ is not None:
            frac = footprint_free_fraction(
                occ, spec["pose"], grid.length, grid.width,
                float(cfg.get("placement_clearance", 0.5)),
            )
            summary["map_free_fraction"] = frac
            if frac < 1.0:
                warnings.append(
                    f"{spec['name']}: {100 * (1 - frac):.1f}% of footprint plus "
                    f"clearance hits occupied or unknown map cells"
                )

        patches.append(PatchResult(
            name=spec["name"], type=spec["type"], material=spec["material"],
            pose=dict(spec["pose"]), length=grid.length, width=grid.width,
            x=x, y=y, height=h, slope=s, roughness=r, summary=summary,
        ))

    raster, overlaps = rasterize(patches, float(cfg.get("ground_truth_resolution", res)))
    if overlaps:
        warnings.append(f"{overlaps} ground truth cells covered by more than one patch")

    return TerrainResult(config=cfg, patches=patches, raster=raster, warnings=warnings)


def _git_commit(path: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def save_ground_truth(result: TerrainResult, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "terrain_ground_truth.npz")
    np.savez_compressed(npz_path, **result.raster)

    manifest = {
        "generator_version": __version__,
        "git_commit": _git_commit(os.path.dirname(os.path.abspath(__file__))),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "frame": "Isaac Sim world, assumed equal to Nav2 map frame",
        "raster": {
            "file": os.path.basename(npz_path),
            "origin_xy": result.raster["origin_xy"].tolist(),
            "resolution": float(result.raster["resolution"]),
            "shape": list(result.raster["height_m"].shape),
            "row0": "minimum y",
        },
        "patches": [
            {
                "id": i, "name": p.name, "type": p.type, "material": p.material,
                "pose": p.pose, "size": {"length": p.length, "width": p.width},
                **p.summary,
            }
            for i, p in enumerate(result.patches)
        ],
        "warnings": result.warnings,
        "config": result.config,
    }
    json_path = os.path.join(out_dir, "terrain_manifest.json")
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return npz_path, json_path


def format_report(result: TerrainResult) -> str:
    head = f"{'patch':<18}{'type':<10}{'max h':>8}{'max slope':>11}{'p95 rough':>11}{'map free':>10}"
    lines = [head, "-" * len(head)]
    for p in result.patches:
        s = p.summary
        free = s.get("map_free_fraction")
        free_txt = f"{100 * free:.0f}%" if free is not None else "n/a"
        lines.append(
            f"{p.name:<18}{p.type:<10}{s['max_height_m']:>7.3f}m"
            f"{s['max_slope_deg']:>9.1f}deg{s['p95_roughness_m'] * 1000:>9.1f}mm{free_txt:>10}"
        )
    for w in result.warnings:
        lines.append(f"WARNING: {w}")
    return "\n".join(lines)
