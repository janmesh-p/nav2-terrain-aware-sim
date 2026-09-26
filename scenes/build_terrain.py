#!/usr/bin/env python3
"""Build navigation test terrain into an Isaac Sim USD stage.

Dry run (no Isaac Sim, fast): heightfields, metrics, placement check,
ground truth export.

    ~/IsaacLab/.venv/bin/python scenes/build_terrain.py --dry-run

Full build (headless Isaac Sim, writes a new USD):

    ~/IsaacLab/.venv/bin/python scenes/build_terrain.py \
        --stage scenes/warehouse_terrain.usd \
        --out scenes/warehouse_terrain_built.usd
"""

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "configs" / "warehouse_terrain.yaml"))
    ap.add_argument("--stage", default=str(HERE / "warehouse_terrain.usd"))
    ap.add_argument("--out", default=str(HERE / "warehouse_terrain_built.usd"))
    ap.add_argument("--gt-dir", default=str(HERE / "ground_truth"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strict", action="store_true", help="fail on placement warnings")
    ap.add_argument("--gui", action="store_true", help="run Isaac Sim with a window")
    args, _ = ap.parse_known_args()

    from terrain_gen.builder import format_report, generate, load_config, save_ground_truth

    cfg = load_config(args.config)
    result = generate(cfg)
    print(format_report(result))
    npz, manifest = save_ground_truth(result, args.gt_dir)
    print(f"ground truth: {npz}\nmanifest:     {manifest}")

    if args.strict and result.warnings:
        print("strict mode: aborting on warnings")
        return 2
    if args.dry_run:
        return 0

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": not args.gui})
    try:
        import omni.usd

        from terrain_gen.usd_writer import write_terrain

        ctx = omni.usd.get_context()
        if not ctx.open_stage(args.stage):
            raise RuntimeError(f"could not open stage {args.stage}")
        for _ in range(5):
            app.update()
        written = write_terrain(ctx.get_stage(), cfg, result)
        for path in written:
            print(f"wrote {path}")
        if not ctx.save_as_stage(args.out):
            ctx.get_stage().GetRootLayer().Export(args.out)
        print(f"saved {args.out}")
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
