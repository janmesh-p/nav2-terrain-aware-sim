#!/usr/bin/env python3
"""Check that every route goal is reachable in principle before driving.

A goal passes when it is
  * at least --clearance metres from every obstacle: occupied Nav2 map cells,
    scene objects exported from Isaac Sim, and (optionally) lethal cells the
    terrain layer actually produced in an accuracy capture;
  * outside every terrain patch (entry and exit points sit beside a patch).

For failing goals, --suggest searches nearby for the closest point that passes.

    python3 tools/validate_route.py
    python3 tools/validate_route.py --suggest
    python3 tools/validate_route.py --lethal-capture results/accuracy/<run>/capture.npz
"""

import argparse
import json
import math
import os
import pathlib
import sys

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scenes"))
from terrain_gen.placement import OccupancyMap  # noqa: E402


def rect_distance(px, py, cx, cy, length, width, yaw_deg):
    """Signed distance from points to a rotated rectangle (negative inside)."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = px - cx, py - cy
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    ox, oy = np.abs(lx) - length / 2, np.abs(ly) - width / 2
    outside = np.hypot(np.maximum(ox, 0), np.maximum(oy, 0))
    inside = np.minimum(np.maximum(ox, oy), 0)
    return outside + inside


class Obstacles:
    def __init__(self, map_yaml, scene_json, lethal_npz, terrain_cfg):
        self.sources = []
        pts = []
        if map_yaml and os.path.exists(map_yaml):
            occ = OccupancyMap(map_yaml)
            h, w = occ.free.shape
            rows, cols = np.nonzero(~occ.free)
            x = occ.origin[0] + (cols + 0.5) * occ.resolution
            y = occ.origin[1] + (h - 1 - rows + 0.5) * occ.resolution
            pts.append(np.c_[x, y])
            self.sources.append(f"nav2 map ({len(x)} occupied cells)")
        self.map_pts = np.vstack(pts) if pts else np.zeros((0, 2))

        self.boxes = []
        if scene_json and os.path.exists(scene_json):
            self.boxes = json.loads(pathlib.Path(scene_json).read_text())["boxes"]
            self.sources.append(f"scene objects ({len(self.boxes)} boxes)")

        self.lethal_pts = np.zeros((0, 2))
        if lethal_npz and os.path.exists(lethal_npz):
            cap = np.load(lethal_npz)
            keep = cap["cost_pct"] >= 99
            px, py = cap["x"][keep], cap["y"][keep]
            # terrain patches are allowed to be lethal inside; only keep cells off-patch
            off = np.ones(px.shape, bool)
            for p in terrain_cfg["patches"]:
                d = rect_distance(px, py, p["pose"]["x"], p["pose"]["y"], p["size"]["length"],
                                  p["size"]["width"], p["pose"].get("yaw_deg", 0.0))
                off &= d > 0.2
            self.lethal_pts = np.c_[px[off], py[off]]
            self.sources.append(f"layer lethal cells ({off.sum()} off-patch)")

        self.patches = terrain_cfg["patches"]

    def nearest(self, x, y):
        """Nearest obstacle distance and its label."""
        best, label = np.inf, "none"
        for pts, name in ((self.map_pts, "nav2 map"), (self.lethal_pts, "layer lethal")):
            if len(pts):
                d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
                i = int(d.argmin())
                if d[i] < best:
                    best, label = float(d[i]), f"{name} at ({pts[i, 0]:.1f}, {pts[i, 1]:.1f})"
        for b in self.boxes:
            dx = max(b["min"][0] - x, 0.0, x - b["max"][0])
            dy = max(b["min"][1] - y, 0.0, y - b["max"][1])
            d = math.hypot(dx, dy)
            if d < best:
                best, label = d, b["path"].split("/")[-1]
        return best, label

    def patch_problem(self, x, y, own_prefix, own_margin, other_margin):
        """Name of a patch the goal is too close to, or None.

        A goal may sit beside its own patch (entry or exit point) but must keep
        a wider margin from every other patch.
        """
        for p in self.patches:
            d = rect_distance(np.array([x]), np.array([y]), p["pose"]["x"], p["pose"]["y"],
                              p["size"]["length"], p["size"]["width"], p["pose"].get("yaw_deg", 0.0))[0]
            own = own_prefix is not None and p["name"].startswith(own_prefix)
            need = own_margin if own else other_margin
            if d < need:
                return f"{p['name']} ({d:+.2f} m, needs {need:.1f})"
        return None


OWN_MARGIN = 0.5
OTHER_MARGIN = 1.0


def check(obs, x, y, clearance, own_prefix=None):
    d, label = obs.nearest(x, y)
    patch = obs.patch_problem(x, y, own_prefix, OWN_MARGIN, OTHER_MARGIN)
    ok = d >= clearance and patch is None
    return ok, d, label, patch


def suggest(obs, x, y, clearance, own_prefix=None, radius=2.5, step=0.1):
    best = None
    for r in np.arange(step, radius + 1e-9, step):
        n = max(8, int(2 * math.pi * r / step))
        for a in np.linspace(0, 2 * math.pi, n, endpoint=False):
            cx, cy = x + r * math.cos(a), y + r * math.sin(a)
            ok, d, _, _ = check(obs, cx, cy, clearance, own_prefix)
            if ok and (best is None or r < best[0] - 1e-9):
                best = (r, cx, cy, d)
        if best:
            return best
    return None


def validate(route_path, terrain_path, clearance, scene_json, lethal_npz, do_suggest=False,
             quiet=False):
    route = yaml.safe_load(open(route_path))
    terrain = yaml.safe_load(open(terrain_path))
    map_yaml = os.path.expanduser(terrain.get("nav2_map", ""))
    obs = Obstacles(map_yaml, scene_json, lethal_npz, terrain)
    out = print if not quiet else (lambda *a, **k: None)
    out("obstacle sources: " + ("; ".join(obs.sources) or "none"))
    if not obs.boxes:
        out("WARNING: no scene objects loaded. Run scenes/export_scene_obstacles.py in Isaac Sim.")
    out(f"required clearance: obstacles {clearance:.1f} m, own patch {OWN_MARGIN:.1f} m, "
        f"other patches {OTHER_MARGIN:.1f} m\n")
    out(f"{'goal':<13}{'x':>7}{'y':>7}{'clear':>8}  nearest obstacle")
    all_ok = True
    points = [("start", route["start"]["x"], route["start"]["y"])] + \
             [(w["name"], w["x"], w["y"]) for w in route["waypoints"]]
    for name, x, y in points:
        prefix = name.split("_")[0] if "_" in name else None
        ok, d, label, patch = check(obs, x, y, clearance, prefix)
        flag = "ok" if ok else "FAIL"
        extra = f"  too close to {patch}" if patch else ""
        out(f"{name:<13}{x:>7.2f}{y:>7.2f}{d:>7.2f}m  {label}{extra}  [{flag}]")
        if not ok and name != "start":
            all_ok = False
            if do_suggest:
                s = suggest(obs, x, y, clearance, prefix)
                if s:
                    out(f"{'':<13}suggest ({s[1]:.2f}, {s[2]:.2f}), moved {s[0]:.2f} m, clearance {s[3]:.2f} m")
                else:
                    out(f"{'':<13}no valid point within 2.5 m")
    out("\nroute " + ("OK" if all_ok else "INVALID"))
    return all_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--route", default=str(REPO / "bringup" / "routes" / "terrain_tour.yaml"))
    ap.add_argument("--terrain", default=str(REPO / "scenes" / "configs" / "warehouse_terrain.yaml"))
    ap.add_argument("--scene", default=str(REPO / "scenes" / "scene_obstacles.json"))
    ap.add_argument("--lethal-capture", help="capture.npz from tools/layer_accuracy.py")
    ap.add_argument("--clearance", type=float, default=1.0)
    ap.add_argument("--suggest", action="store_true")
    args = ap.parse_args()
    ok = validate(args.route, args.terrain, args.clearance, args.scene, args.lethal_capture, args.suggest)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
