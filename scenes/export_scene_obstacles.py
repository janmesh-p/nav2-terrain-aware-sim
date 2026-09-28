"""Paste into the Isaac Sim Script Editor to export floor-level obstacles.

Writes scenes/scene_obstacles.json: world axis-aligned boxes of meshes that
occupy the robot's height band. Used by tools/validate_route.py.

Kept: meshes whose box starts below MAX_BOTTOM and rises above MIN_TOP.
Dropped: floors, decals, ground planes, the generated terrain, the robot,
and building-scale meshes (taller than MAX_HEIGHT), whose boxes are too
coarse to be useful and which the Nav2 map already covers.
"""

import json
import os

import omni.usd
from pxr import Usd, UsdGeom

REPO = os.path.expanduser("~/nav2-terrain-aware-sim")
MIN_TOP = 0.03
MAX_BOTTOM = 0.5
MAX_HEIGHT = 3.0
SKIP = ("floor", "Floor", "Decal", "GroundPlane", "/World/Terrain", "Nova_Carter", "GroundTruthGraph")

stage = omni.usd.get_context().get_stage()
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
boxes, dropped_tall = [], 0
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if not prim.IsA(UsdGeom.Mesh) or any(s in path for s in SKIP):
        continue
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = r.GetMin(), r.GetMax()
    if lo[2] > MAX_BOTTOM or hi[2] < MIN_TOP:
        continue
    if hi[2] - lo[2] > MAX_HEIGHT:
        dropped_tall += 1
        continue
    boxes.append({"path": path, "min": [round(lo[0], 3), round(lo[1], 3)],
                  "max": [round(hi[0], 3), round(hi[1], 3)], "top": round(hi[2], 3)})

out = os.path.join(REPO, "scenes", "scene_obstacles.json")
with open(out, "w") as f:
    json.dump({"min_top": MIN_TOP, "max_bottom": MAX_BOTTOM, "max_height": MAX_HEIGHT,
               "dropped_building_scale": dropped_tall, "boxes": boxes}, f, indent=1)
print(f"{len(boxes)} obstacle boxes written to {out} ({dropped_tall} building-scale meshes dropped)")
