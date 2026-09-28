#!/usr/bin/env python3
"""Score the terrain costmap layer against generator ground truth.

Perception is isolated from localization: the layer's local costmap grids
(odom frame) are placed in the map frame through odometry anchored to the
true start pose. In this simulator odometry is exact, so any error left is
the layer's own.

Ground truth is recomputed at the layer's scale (same plane-fit window), so
sub-window ripples the layer cannot resolve by design are not counted.

Start before the robot moves, drive (or run the route), then Ctrl+C:

    source /opt/ros/jazzy/setup.bash
    python3 tools/layer_accuracy.py --label terrain_acc1

Offline re-score of a saved capture:

    python3 tools/layer_accuracy.py --score results/accuracy/<run>/capture.npz
"""

import argparse
import datetime as dt
import json
import math
import pathlib
import sys

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scenes"))
from terrain_gen.metrics import _box_mean  # noqa: E402

CELL = 0.10          # capture grid, matches the layer resolution
FLOOR_BUFFER = 0.30  # floor cells this close to a patch are excluded
EDGE_BAND = 0.25     # ramp side band used for the edge statistic
LETHAL_PCT = 99      # published cost value (0..100) treated as lethal


# ---------------------------------------------------------------- ground truth

def plane_fit(h: np.ndarray, res: float, window: float):
    """Windowed least-squares plane: slope [deg] and residual RMS [m].

    Same model as the layer, so ground truth and estimate share a scale.
    """
    k = max(3, int(round(window / res)) | 1)
    ny, nx = h.shape
    X, Y = np.meshgrid(np.arange(nx) * res, np.arange(ny) * res, indexing="xy")
    m = lambda a: _box_mean(a, k)
    mh, mx, my = m(h), m(X), m(Y)
    cxx, cyy = m(X * X) - mx**2, m(Y * Y) - my**2
    cxy = m(X * Y) - mx * my
    cxh, cyh = m(X * h) - mx * mh, m(Y * h) - my * mh
    chh = m(h * h) - mh**2
    det = np.maximum(cxx * cyy - cxy**2, 1e-12)
    gx = (cyy * cxh - cxy * cyh) / det
    gy = (cxx * cyh - cxy * cxh) / det
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    rough = np.sqrt(np.clip(chh - gx * cxh - gy * cyh, 0, None))
    return slope, rough


class GroundTruth:
    def __init__(self, gt_dir: pathlib.Path, window: float, pad: float = 1.5):
        d = np.load(gt_dir / "terrain_ground_truth.npz")
        manifest = json.loads((gt_dir / "terrain_manifest.json").read_text())
        self.names = {p["id"]: p["name"] for p in manifest["patches"]}
        res = float(d["resolution"])
        n = int(round(pad / res))
        h = np.nan_to_num(d["height_m"], nan=0.0)          # floor is z = 0
        pid = d["patch_id"].astype(int)
        self.height = np.pad(h, n)
        self.pid = np.pad(pid, n, constant_values=-1)
        self.res = res
        self.origin = np.asarray(d["origin_xy"], float) - n * res
        self.slope, self.rough = plane_fit(self.height, res, window)
        k = int(round(FLOOR_BUFFER / res))
        patch = (self.pid >= 0).astype(float)
        near = _box_mean(patch, 2 * k + 1) > 0 if k > 0 else patch > 0
        self.region = np.full(self.pid.shape, "floor", dtype=object)
        self.region[near & (self.pid < 0)] = "buffer"
        for i, name in self.names.items():
            self.region[self.pid == i] = name
        # ramp side band, for the edge statistic
        self.ramp_edge = np.zeros(self.pid.shape, bool)
        for i, p in enumerate(manifest["patches"]):
            if p["type"] != "ramp":
                continue
            yaw = math.radians(p["pose"].get("yaw_deg", 0.0))
            ny, nx = self.pid.shape
            X, Y = np.meshgrid(self.origin[0] + (np.arange(nx) + 0.5) * res,
                               self.origin[1] + (np.arange(ny) + 0.5) * res, indexing="xy")
            dx, dy = X - p["pose"]["x"], Y - p["pose"]["y"]
            ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
            self.ramp_edge |= (self.pid == p["id"]) & (np.abs(ly) > p["size"]["width"] / 2 - EDGE_BAND)

    def lookup(self, wx: np.ndarray, wy: np.ndarray):
        i = np.floor((wx - self.origin[0]) / self.res).astype(int)
        j = np.floor((wy - self.origin[1]) / self.res).astype(int)
        inside = (i >= 0) & (j >= 0) & (i < self.pid.shape[1]) & (j < self.pid.shape[0])
        i, j = np.clip(i, 0, self.pid.shape[1] - 1), np.clip(j, 0, self.pid.shape[0] - 1)
        region = np.where(inside, self.region[j, i], "floor")
        slope = np.where(inside, self.slope[j, i], 0.0)
        rough = np.where(inside, self.rough[j, i], 0.0)
        edge = np.where(inside, self.ramp_edge[j, i], False)
        return region, slope, rough, edge


# ---------------------------------------------------------------- scoring

def load_layer_params(params_file: pathlib.Path) -> dict:
    cfg = yaml.safe_load(params_file.read_text())
    node = cfg["local_costmap"]
    node = node.get("local_costmap", node)["ros__parameters"]
    return node["terrain_layer"]


def score(cap: dict, gt: GroundTruth, lp: dict) -> dict:
    """cap: dict of arrays x, y (map frame cell centres), slope_pct, rough_pct, cost_pct."""
    x, y = cap["x"], cap["y"]
    s_lim, r_lim = float(lp["slope_lethal_deg"]), float(lp["roughness_max_cost"])
    est_s = np.where(cap["slope_pct"] >= 0, cap["slope_pct"] / 100.0 * s_lim, np.nan)
    est_r = np.where(cap["rough_pct"] >= 0, cap["rough_pct"] / 100.0 * r_lim, np.nan)
    cost = cap["cost_pct"]
    region, gt_s, gt_r, edge = gt.lookup(x, y)
    # Published slope saturates at the lethal limit; compare within that range.
    gt_s_c = np.minimum(gt_s, s_lim)
    gt_r_c = np.minimum(gt_r, r_lim)

    out = {}
    order = ["floor"] + sorted(set(region) - {"floor", "buffer"})
    for reg in order:
        m = (region == reg) & np.isfinite(est_s)
        if not m.any():
            continue
        e = {
            "cells": int(m.sum()),
            "gt_slope_median_deg": float(np.median(gt_s_c[m])),
            "est_slope_median_deg": float(np.median(est_s[m])),
            "slope_mae_deg": float(np.mean(np.abs(est_s[m] - gt_s_c[m]))),
            "slope_bias_deg": float(np.mean(est_s[m] - gt_s_c[m])),
            "gt_rough_median_mm": float(np.median(gt_r_c[m]) * 1000),
            "est_rough_median_mm": float(np.nanmedian(est_r[m]) * 1000),
            "rough_mae_mm": float(np.nanmean(np.abs(est_r[m] - gt_r_c[m])) * 1000),
            "cost_median_pct": float(np.median(cost[m])),
            "lethal_share": float((cost[m] >= LETHAL_PCT).mean()),
            "nonzero_cost_share": float((cost[m] > 0).mean()),
        }
        if reg.startswith("ramp"):
            inc = m & ~edge & (gt_s > 3.0)
            if inc.any():
                e["incline_est_slope_median_deg"] = float(np.median(est_s[inc]))
                e["incline_gt_slope_median_deg"] = float(np.median(gt_s[inc]))
                e["incline_lethal_share"] = float((cost[inc] >= LETHAL_PCT).mean())
            me = m & edge
            if me.any():
                e["edge_lethal_share"] = float((cost[me] >= LETHAL_PCT).mean())
        out[reg] = e
    return out


def format_report(res: dict) -> str:
    head = (f"{'region':<14}{'cells':>7}{'gt slope':>10}{'est slope':>11}{'MAE':>7}{'bias':>7}"
            f"{'gt rough':>10}{'est rough':>11}{'cost>0':>8}{'lethal':>8}")
    L = [head, "-" * len(head)]
    for reg, e in res.items():
        L.append(f"{reg:<14}{e['cells']:>7}{e['gt_slope_median_deg']:>8.1f}d{e['est_slope_median_deg']:>9.1f}d"
                 f"{e['slope_mae_deg']:>6.1f}d{e['slope_bias_deg']:>+6.1f}d{e['gt_rough_median_mm']:>8.1f}mm"
                 f"{e['est_rough_median_mm']:>9.1f}mm{100 * e['nonzero_cost_share']:>7.0f}%"
                 f"{100 * e['lethal_share']:>7.0f}%")
    for reg, e in res.items():
        if "incline_est_slope_median_deg" in e:
            L += ["", f"{reg}: incline reads {e['incline_est_slope_median_deg']:.1f} deg "
                      f"(truth {e['incline_gt_slope_median_deg']:.1f} deg), "
                      f"lethal on incline {100 * e['incline_lethal_share']:.0f}%"
                      + (f", lethal on side band {100 * e['edge_lethal_share']:.0f}%"
                         if "edge_lethal_share" in e else "")]
    return "\n".join(L)


def figure(cap: dict, gt: GroundTruth, lp: dict, path: pathlib.Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping figure")
        return
    s_lim = float(lp["slope_lethal_deg"])
    ny, nx = gt.pid.shape
    ext = [gt.origin[0], gt.origin[0] + nx * gt.res, gt.origin[1], gt.origin[1] + ny * gt.res]
    est = np.full(gt.pid.shape, np.nan)
    cost = np.full(gt.pid.shape, np.nan)
    i = np.floor((cap["x"] - gt.origin[0]) / gt.res).astype(int)
    j = np.floor((cap["y"] - gt.origin[1]) / gt.res).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < nx) & (j < ny) & (cap["slope_pct"] >= 0)
    # capture cells are 0.1 m, raster 0.05 m: paint the 2x2 block
    for di in (0, 1):
        for dj in (0, 1):
            ii, jj = np.clip(i[ok] + di, 0, nx - 1), np.clip(j[ok] + dj, 0, ny - 1)
            est[jj, ii] = cap["slope_pct"][ok] / 100.0 * s_lim
            cost[jj, ii] = cap["cost_pct"][ok]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    for a, img, title, cmap, vmax in (
        (ax[0], np.minimum(gt.slope, s_lim), "Ground truth slope (layer scale)", "viridis", s_lim),
        (ax[1], est, "Layer slope estimate", "viridis", s_lim),
        (ax[2], cost, "Layer cost (0-100, lethal >= 99)", "magma", 100),
    ):
        im = a.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=vmax)
        a.contour(np.linspace(ext[0], ext[1], nx), np.linspace(ext[2], ext[3], ny),
                  (gt.pid >= 0).astype(float), levels=[0.5], colors="w", linewidths=0.8)
        a.set_title(title)
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure: {path}")


def finish(cap: dict, out: pathlib.Path, gt: GroundTruth, lp: dict) -> None:
    res = score(cap, gt, lp)
    text = format_report(res)
    (out / "accuracy.txt").write_text(text + "\n")
    (out / "accuracy.json").write_text(json.dumps(res, indent=2))
    print(text)
    figure(cap, gt, lp, out / "accuracy.png")


# ---------------------------------------------------------------- live capture

def run_live(args, gt: GroundTruth, lp: dict) -> None:
    import rclpy
    import tf2_ros
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rclpy.time import Time
    from tf2_msgs.msg import TFMessage

    def yaw_of(q):
        return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    class Capture(Node):
        def __init__(self):
            super().__init__("terrain_layer_accuracy")
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
            self.anchor = None  # map <- odom, (x, y, yaw)
            self.cells = {}     # (ix, iy) -> [slope, rough, cost]
            self.msgs = 0
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            base = args.topic_prefix
            for k, name in enumerate(("slope", "roughness", "cost")):
                self.create_subscription(OccupancyGrid, base + name,
                                         lambda m, k=k: self.on_grid(k, m), qos)
            self.create_subscription(TFMessage, args.gt_topic, self.on_truth, 10)
            self.create_timer(5.0, lambda: self.get_logger().info(
                f"grids {self.msgs}  cells {len(self.cells)}  anchored {self.anchor is not None}"),
                clock=rclpy.clock.Clock())

        def on_truth(self, msg):
            if self.anchor is not None:
                return
            tf = next((t for t in msg.transforms if t.child_frame_id == args.gt_child), None)
            if tf is None:
                return
            try:
                ob = self.tf_buffer.lookup_transform("odom", args.gt_child, Time())
            except tf2_ros.TransformException:
                return
            tx, ty = tf.transform.translation.x, tf.transform.translation.y
            tyaw = yaw_of(tf.transform.rotation)
            ox, oy = ob.transform.translation.x, ob.transform.translation.y
            oyaw = yaw_of(ob.transform.rotation)
            ayaw = tyaw - oyaw
            c, s = math.cos(ayaw), math.sin(ayaw)
            self.anchor = (tx - (c * ox - s * oy), ty - (s * ox + c * oy), ayaw)
            self.get_logger().info(
                "anchored odom to truth: map <- odom = (%.2f, %.2f, %.0f deg). Start driving."
                % (self.anchor[0], self.anchor[1], math.degrees(ayaw)))

        def on_grid(self, k, msg):
            if self.anchor is None or msg.header.frame_id != "odom":
                return
            self.msgs += 1
            w, h, r = msg.info.width, msg.info.height, msg.info.resolution
            d = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
            jj, ii = np.nonzero(d >= 0)
            ox = msg.info.origin.position.x + (ii + 0.5) * r
            oy = msg.info.origin.position.y + (jj + 0.5) * r
            ax, ay, ayaw = self.anchor
            c, s = math.cos(ayaw), math.sin(ayaw)
            mx, my = ax + c * ox - s * oy, ay + s * ox + c * oy
            ix = np.floor(mx / CELL).astype(int)
            iy = np.floor(my / CELL).astype(int)
            vals = d[jj, ii]
            for a, b, v in zip(ix.tolist(), iy.tolist(), vals.tolist()):
                cell = self.cells.get((a, b))
                if cell is None:
                    cell = self.cells[(a, b)] = [-1, -1, -1]
                cell[k] = v  # latest observation wins

    rclpy.init()
    node = Capture()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    cells = dict(node.cells)
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass
    if not cells:
        print("no cells captured")
        return
    keys = np.array(list(cells.keys()))
    vals = np.array(list(cells.values()))
    keep = (vals >= 0).all(axis=1)
    cap = {"x": (keys[keep, 0] + 0.5) * CELL, "y": (keys[keep, 1] + 0.5) * CELL,
           "slope_pct": vals[keep, 0].astype(float), "rough_pct": vals[keep, 1].astype(float),
           "cost_pct": vals[keep, 2].astype(float)}
    out = pathlib.Path(args.out) / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{args.label}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "capture.npz", **cap)
    print(f"\n{keep.sum()} cells captured -> {out}")
    finish(cap, out, gt, lp)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default="acc")
    ap.add_argument("--params", default=str(REPO / "bringup" / "params" / "nav2_terrain.yaml"))
    ap.add_argument("--gt-dir", default=str(REPO / "scenes" / "ground_truth"))
    ap.add_argument("--topic-prefix", default="/local_costmap/local_costmap/terrain_layer/")
    ap.add_argument("--gt-topic", default="/ground_truth_tf")
    ap.add_argument("--gt-child", default="nova_carter")
    ap.add_argument("--out", default=str(REPO / "results" / "accuracy"))
    ap.add_argument("--score", metavar="CAPTURE_NPZ", help="re-score a saved capture offline")
    args = ap.parse_args()

    lp = load_layer_params(pathlib.Path(args.params))
    gt = GroundTruth(pathlib.Path(args.gt_dir), window=float(lp["window"]))
    if args.score:
        cap = dict(np.load(args.score))
        finish(cap, pathlib.Path(args.score).parent, gt, lp)
    else:
        run_live(args, gt, lp)


if __name__ == "__main__":
    main()
