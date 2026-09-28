#!/usr/bin/env python3
"""Search a terrain layout that is clear of obstacles and well spaced.

Constraints (all hard):
  * every patch footprint is free space, grown by --clearance, in both the
    Nav2 map and the exported scene objects;
  * patches are at least --gap metres apart, edge to edge;
  * no patch within --spawn-clearance of the robot spawn;
  * each patch gets an entry and an exit goal just beyond its two ends,
    and those goals are clear too, so the patch can be crossed lengthwise.

Among valid layouts, the most compact one near the spawn wins. Writes the
terrain config and a matching route; run tools/validate_route.py after.

    python3 tools/plan_layout.py                 # dry run, prints the layout
    python3 tools/plan_layout.py --write         # update config and route
"""

import argparse
import copy
import itertools
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

GOAL_OFFSET = 0.6   # entry and exit goals sit this far beyond a patch end
STEP = 0.5          # candidate grid spacing [m]


class Blocked:
    """Boolean grid of cells a footprint may not touch, with O(1) box queries."""

    def __init__(self, occ: OccupancyMap, boxes: list, grow: float):
        self.res = occ.resolution
        self.ox, self.oy = occ.origin
        blocked = ~occ.free[::-1]  # flip so row 0 is minimum y
        h, w = blocked.shape
        for b in boxes:
            i0 = int(math.floor((b["min"][0] - self.ox) / self.res))
            i1 = int(math.ceil((b["max"][0] - self.ox) / self.res))
            j0 = int(math.floor((b["min"][1] - self.oy) / self.res))
            j1 = int(math.ceil((b["max"][1] - self.oy) / self.res))
            blocked[max(0, j0):min(h, j1), max(0, i0):min(w, i1)] = True
        self.raw = blocked
        self.grown = self._grow(blocked, grow)
        self.w, self.h = w, h
        self.width_m, self.height_m = w * self.res, h * self.res
        s = np.zeros((h + 1, w + 1), np.int64)
        s[1:, 1:] = self.grown.astype(np.int64).cumsum(0).cumsum(1)
        self.integral = s

    def _grow(self, grid, r):
        k = int(math.ceil(r / self.res))
        if k <= 0:
            return grid.copy()
        p = np.pad(grid.astype(np.int64), k)
        s = np.zeros((p.shape[0] + 1, p.shape[1] + 1), np.int64)
        s[1:, 1:] = p.cumsum(0).cumsum(1)
        n = 2 * k + 1
        tot = s[n:, n:] - s[:-n, n:] - s[n:, :-n] + s[:-n, :-n]
        return tot > 0

    def rect_clear(self, xmin, ymin, xmax, ymax) -> bool:
        i0 = int(math.floor((xmin - self.ox) / self.res))
        i1 = int(math.ceil((xmax - self.ox) / self.res))
        j0 = int(math.floor((ymin - self.oy) / self.res))
        j1 = int(math.ceil((ymax - self.oy) / self.res))
        if i0 < 0 or j0 < 0 or i1 > self.w or j1 > self.h:
            return False
        s = self.integral
        return (s[j1, i1] - s[j0, i1] - s[j1, i0] + s[j0, i0]) == 0

    def point_clear(self, x, y) -> bool:
        return self.rect_clear(x - self.res / 2, y - self.res / 2, x + self.res / 2, y + self.res / 2)


def footprint(cx, cy, length, width, yaw):
    lx, ly = (length, width) if yaw == 0 else (width, length)
    return (cx - lx / 2, cy - ly / 2, cx + lx / 2, cy + ly / 2)


def rect_gap(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)


def ends(cx, cy, length, yaw):
    d = length / 2 + GOAL_OFFSET
    if yaw == 0:
        return (cx - d, cy), (cx + d, cy)
    return (cx, cy - d), (cx, cy + d)


def candidates(spec, blocked, spawn, args, target=None):
    """Valid poses for one patch, sorted by distance to target (spawn if None)."""
    tx, ty = target if target is not None else spawn
    L, W = spec["size"]["length"], spec["size"]["width"]
    out = []
    xs = np.arange(blocked.ox + 1, blocked.ox + blocked.width_m - 1, STEP)
    ys = np.arange(blocked.oy + 1, blocked.oy + blocked.height_m - 1, STEP)
    for yaw in (0, 90):
        for cx in xs:
            for cy in ys:
                fp = footprint(cx, cy, L, W, yaw)
                if not blocked.rect_clear(*fp):
                    continue
                if rect_gap(fp, (spawn[0], spawn[1], spawn[0], spawn[1])) < args.spawn_clearance:
                    continue
                e1, e2 = ends(cx, cy, L, yaw)
                if not (blocked.point_clear(*e1) and blocked.point_clear(*e2)):
                    continue
                out.append((math.hypot(cx - tx, cy - ty), float(cx), float(cy), yaw, fp))
    out.sort(key=lambda c: c[0])
    return out


def search(specs, cands, args):
    """Best combination by total distance to spawn, pruned by the gap rule."""
    best = [math.inf, None]
    top = [c[: args.beam] for c in cands]

    def rec(i, chosen, cost):
        if cost >= best[0]:
            return
        if i == len(specs):
            best[0], best[1] = cost, list(chosen)
            return
        for c in top[i]:
            if cost + c[0] >= best[0]:
                break  # sorted by distance, nothing cheaper follows
            if all(rect_gap(c[4], o[4]) >= args.gap for o in chosen):
                chosen.append(c)
                rec(i + 1, chosen, cost + c[0])
                chosen.pop()

    rec(0, [], 0.0)
    return best[1]


def build_route(specs, layout, spawn):
    """Visit patches nearest first, crossing each lengthwise toward the far end."""
    remaining = list(range(len(specs)))
    pos, wps = spawn, []
    while remaining:
        k = min(remaining, key=lambda i: math.hypot(layout[i][1] - pos[0], layout[i][2] - pos[1]))
        remaining.remove(k)
        _, cx, cy, yaw, _ = layout[k]
        e1, e2 = ends(cx, cy, specs[k]["size"]["length"], yaw)
        if math.hypot(e2[0] - pos[0], e2[1] - pos[1]) < math.hypot(e1[0] - pos[0], e1[1] - pos[1]):
            e1, e2 = e2, e1
        prefix = specs[k]["name"].split("_")[0]
        wps.append({"name": f"{prefix}_entry", "x": round(e1[0], 2), "y": round(e1[1], 2)})
        wps.append({"name": f"{prefix}_exit", "x": round(e2[0], 2), "y": round(e2[1], 2)})
        pos = e2
    return wps


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--terrain", default=str(REPO / "scenes" / "configs" / "warehouse_terrain.yaml"))
    ap.add_argument("--route", default=str(REPO / "bringup" / "routes" / "terrain_tour.yaml"))
    ap.add_argument("--scene", default=str(REPO / "scenes" / "scene_obstacles.json"))
    ap.add_argument("--spawn", nargs=2, type=float, default=[-6.0, -1.0])
    ap.add_argument("--gap", type=float, default=5.0, help="min edge-to-edge patch spacing [m]")
    ap.add_argument("--clearance", type=float, default=1.0, help="free margin around footprints [m]")
    ap.add_argument("--spawn-clearance", type=float, default=1.5)
    ap.add_argument("--beam", type=int, default=400, help="candidates kept per patch")
    ap.add_argument("--anchor", nargs=2, action="append", default=[], metavar=("PATCH", "CORNER"),
                    help="pull a patch toward a map corner: ne, nw, se, sw (repeatable)")
    ap.add_argument("--patch-clearance", nargs=2, action="append", default=[], metavar=("PATCH", "M"),
                    help="clearance for one patch, overriding --clearance (repeatable)")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.terrain))
    occ = OccupancyMap(os.path.expanduser(cfg["nav2_map"]))
    boxes = json.loads(pathlib.Path(args.scene).read_text())["boxes"] if os.path.exists(args.scene) else []
    if not boxes:
        print("WARNING: no scene objects. Run scenes/export_scene_obstacles.py in Isaac Sim first.")
    spawn = tuple(args.spawn)
    per_clear = {name: float(m) for name, m in args.patch_clearance}
    grids = {}

    def blocked_for(name):
        c = per_clear.get(name, args.clearance)
        if c not in grids:
            grids[c] = Blocked(occ, boxes, c)
        return grids[c]

    base = blocked_for(None)
    corners = {
        "sw": (base.ox, base.oy), "se": (base.ox + base.width_m, base.oy),
        "nw": (base.ox, base.oy + base.height_m), "ne": (base.ox + base.width_m, base.oy + base.height_m),
    }
    anchors = {}
    for name, corner in args.anchor:
        if corner.lower() not in corners:
            sys.exit(f"unknown corner '{corner}', use ne, nw, se or sw")
        anchors[name] = corners[corner.lower()]

    # largest patch first: fewest positions, prunes the search fastest
    order = sorted(range(len(cfg["patches"])),
                   key=lambda i: -cfg["patches"][i]["size"]["length"] * cfg["patches"][i]["size"]["width"])
    specs = [cfg["patches"][i] for i in order]
    cands = [candidates(s, blocked_for(s["name"]), spawn, args, anchors.get(s["name"])) for s in specs]
    for s, c in zip(specs, cands):
        print(f"{s['name']:<14} {len(c):>5} valid positions")
    if any(not c for c in cands):
        sys.exit("no valid position for at least one patch; relax --clearance or --gap")

    layout = search(specs, cands, args)
    if layout is None:
        sys.exit(f"no layout with {args.gap} m spacing in the top {args.beam} positions; raise --beam")

    print(f"\nlayout (gap >= {args.gap} m edge to edge, clearance {args.clearance} m):")
    for s, c in zip(specs, layout):
        clear = per_clear.get(s["name"], args.clearance)
        where = f"anchored {dict(args.anchor)[s['name']]}" if s["name"] in anchors else "near spawn"
        print(f"  {s['name']:<14} center ({c[1]:6.2f}, {c[2]:6.2f})  yaw {c[3]:>2}  "
              f"{math.hypot(c[1] - spawn[0], c[2] - spawn[1]):.1f} m from spawn  "
              f"clearance {clear:.1f} m  {where}")
    for (i, a), (j, b) in itertools.combinations(enumerate(layout), 2):
        print(f"  gap {specs[i]['name']} - {specs[j]['name']}: {rect_gap(a[4], b[4]):.2f} m")

    route = {"name": "terrain_tour_v3", "start": {"x": spawn[0], "y": spawn[1]},
             "waypoints": build_route(specs, layout, spawn)}
    print("\nroute:")
    for w in route["waypoints"]:
        print(f"  {w['name']:<12} ({w['x']:6.2f}, {w['y']:6.2f})")

    if not args.write:
        print("\ndry run. Add --write to update the terrain config and route.")
        return
    new = copy.deepcopy(cfg)
    by_name = {s["name"]: c for s, c in zip(specs, layout)}
    for p in new["patches"]:
        c = by_name[p["name"]]
        p["pose"] = {"x": round(c[1], 2), "y": round(c[2], 2), "yaw_deg": float(c[3])}
    new["min_patch_gap"] = args.gap
    with open(args.terrain, "w") as f:
        f.write("# Layout from tools/plan_layout.py. Do not hand edit poses.\n")
        yaml.safe_dump(new, f, sort_keys=False)
    with open(args.route, "w") as f:
        f.write("# Route from tools/plan_layout.py. Validate with tools/validate_route.py.\n")
        yaml.safe_dump(route, f, sort_keys=False)
    print(f"\nwrote {args.terrain}\nwrote {args.route}")


if __name__ == "__main__":
    main()
