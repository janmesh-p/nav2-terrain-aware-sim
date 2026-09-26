"""Paste into the Isaac Sim Script Editor to rebuild terrain in the open stage.

Edit REPO if the repo lives elsewhere. Re-running replaces /World/Terrain.
"""

import importlib
import os
import sys

REPO = os.path.expanduser("~/nav2-terrain-aware-sim")
sys.path.insert(0, os.path.join(REPO, "scenes"))

import omni.usd
import terrain_gen.builder as builder
import terrain_gen.usd_writer as usd_writer

for mod in list(sys.modules):
    if mod.startswith("terrain_gen"):
        importlib.reload(sys.modules[mod])

cfg = builder.load_config(os.path.join(REPO, "scenes", "configs", "warehouse_terrain.yaml"))
result = builder.generate(cfg)
print(builder.format_report(result))
usd_writer.write_terrain(omni.usd.get_context().get_stage(), cfg, result)
