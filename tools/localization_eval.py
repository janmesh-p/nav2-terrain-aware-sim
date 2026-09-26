#!/usr/bin/env python3
"""Measure localization error against Isaac Sim ground truth.

Compares three poses at every ground truth sample:
  truth  : robot chassis pose from Isaac Sim (/ground_truth_tf)
  amcl   : map -> robot from the Nav2 TF tree (AMCL plus odometry)
  odom   : odometry alone, anchored to the map at the first sample

Odometry-only error shows raw drift (wheel slip on terrain). AMCL error
shows how well scan matching corrects it. Each sample is tagged with the
terrain patch under the robot using the generator's ground truth raster.

Run while driving, stop with Ctrl+C to write CSV, summary and plots:

    python3 tools/localization_eval.py --label terrain_run1
"""

import argparse
import collections
import csv
import datetime as dt
import json
import math
import os
import pathlib

import numpy as np
import rclpy
import rclpy.clock
import tf2_ros
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from tf2_msgs.msg import TFMessage

REPO = pathlib.Path(__file__).resolve().parents[1]


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def pose2d(tf) -> tuple:
    t = tf.transform.translation
    return (t.x, t.y, yaw_of(tf.transform.rotation))


def compose(a: tuple, b: tuple) -> tuple:
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


class TerrainLookup:
    def __init__(self, gt_dir: pathlib.Path):
        self.names = {}
        self.pid = None
        npz = gt_dir / "terrain_ground_truth.npz"
        manifest = gt_dir / "terrain_manifest.json"
        if not npz.exists():
            return
        d = np.load(npz)
        self.pid = d["patch_id"]
        self.origin = d["origin_xy"]
        self.res = float(d["resolution"])
        if manifest.exists():
            for p in json.loads(manifest.read_text())["patches"]:
                self.names[p["id"]] = p["name"]

    def region(self, x: float, y: float) -> str:
        if self.pid is None:
            return "unknown"
        i = int(math.floor((x - self.origin[0]) / self.res))
        j = int(math.floor((y - self.origin[1]) / self.res))
        if 0 <= j < self.pid.shape[0] and 0 <= i < self.pid.shape[1] and self.pid[j, i] >= 0:
            return self.names.get(int(self.pid[j, i]), f"patch_{self.pid[j, i]}")
        return "floor"


class LocalizationEval(Node):
    def __init__(self, args):
        super().__init__("localization_eval")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.args = args
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30))
        # Own thread, so TF keeps arriving while this node processes samples.
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        self.terrain = TerrainLookup(REPO / "scenes" / "ground_truth")
        self.rows = []
        self.map_odom0 = None
        self.robot_frame = None
        self.last_ns = None
        self.misses = 0
        self.gt_count = 0
        self.last_error = ""
        # Truth arrives before the matching TF, so evaluate each sample after
        # a short delay instead of blocking on a lookup.
        self.pending = collections.deque(maxlen=2000)
        self.latest_ns = 0
        self.create_subscription(TFMessage, args.gt_topic, self.on_truth, 50)
        self.create_timer(0.05, self.process)
        self.create_timer(5.0, self.status, clock=rclpy.clock.Clock())
        self.get_logger().info(f"waiting for {args.gt_topic} ...")

    def _lookup(self, parent: str, child: str, stamp: Time):
        return self.tf_buffer.lookup_transform(parent, child, stamp)

    def status(self):
        self.get_logger().info(
            f"truth msgs {self.gt_count}  samples {len(self.rows)}  tf misses {self.misses}"
            + (f"  last error: {self.last_error}" if self.last_error else ""))

    def _pick_robot_frame(self, stamp: Time) -> bool:
        for frame in (self.args.gt_child, "base_link"):
            try:
                self._lookup(self.args.map_frame, frame, stamp)
                self.robot_frame = frame
                self.get_logger().info(f"estimate frame: {self.args.map_frame} -> {frame}")
                return True
            except tf2_ros.TransformException as e:
                self.last_error = str(e).splitlines()[0][:120]
        return False

    def on_truth(self, msg: TFMessage):
        tf = next((t for t in msg.transforms if t.child_frame_id == self.args.gt_child), None)
        if tf is None:
            return
        self.gt_count += 1
        stamp_ns = Time.from_msg(tf.header.stamp).nanoseconds
        self.latest_ns = max(self.latest_ns, stamp_ns)
        if self.last_ns is not None and stamp_ns - self.last_ns < 1e9 / self.args.rate:
            return
        self.last_ns = stamp_ns
        self.pending.append((stamp_ns, pose2d(tf)))

    def process(self):
        ready_before = self.latest_ns - int(self.args.delay * 1e9)
        while self.pending and self.pending[0][0] <= ready_before:
            stamp_ns, truth = self.pending.popleft()
            self.evaluate(Time(nanoseconds=stamp_ns), truth)

    def evaluate(self, stamp: Time, truth: tuple):
        if self.robot_frame is None and not self._pick_robot_frame(stamp):
            self.misses += 1
            return
        try:
            amcl = pose2d(self._lookup(self.args.map_frame, self.robot_frame, stamp))
            odom_base = pose2d(self._lookup(self.args.odom_frame, self.robot_frame, stamp))
            if self.map_odom0 is None:
                self.map_odom0 = pose2d(self._lookup(self.args.map_frame, self.args.odom_frame, stamp))
        except tf2_ros.TransformException as e:
            self.misses += 1
            self.last_error = str(e).splitlines()[0][:120]
            return

        odom = compose(self.map_odom0, odom_base)
        if not self.rows:
            self.get_logger().info(
                "first sample  truth (%.2f, %.2f, %.0f deg)  amcl (%.2f, %.2f, %.0f deg)"
                % (truth[0], truth[1], math.degrees(truth[2]),
                   amcl[0], amcl[1], math.degrees(amcl[2])))

        self.rows.append({
            "t": stamp.nanoseconds * 1e-9,
            "truth_x": truth[0], "truth_y": truth[1], "truth_yaw": truth[2],
            "amcl_x": amcl[0], "amcl_y": amcl[1], "amcl_yaw": amcl[2],
            "odom_x": odom[0], "odom_y": odom[1], "odom_yaw": odom[2],
            "amcl_pos_err": math.hypot(amcl[0] - truth[0], amcl[1] - truth[1]),
            "amcl_yaw_err": abs(wrap(amcl[2] - truth[2])),
            "odom_pos_err": math.hypot(odom[0] - truth[0], odom[1] - truth[1]),
            "odom_yaw_err": abs(wrap(odom[2] - truth[2])),
            "region": self.terrain.region(truth[0], truth[1]),
        })


def summarize(rows: list) -> list:
    out = []
    for region in ["all"] + sorted({r["region"] for r in rows}):
        sel = rows if region == "all" else [r for r in rows if r["region"] == region]
        if not sel:
            continue
        a = np.array([r["amcl_pos_err"] for r in sel]) * 100
        ay = np.degrees([r["amcl_yaw_err"] for r in sel])
        o = np.array([r["odom_pos_err"] for r in sel]) * 100
        out.append({
            "region": region, "samples": len(sel),
            "amcl_mean_cm": a.mean(), "amcl_p95_cm": np.percentile(a, 95), "amcl_max_cm": a.max(),
            "amcl_p95_yaw_deg": np.percentile(ay, 95),
            "odom_mean_cm": o.mean(), "odom_max_cm": o.max(),
        })
    return out


def write_outputs(rows: list, label: str, out_root: pathlib.Path):
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_root / f"{stamp}_{label}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    head = (f"{'region':<14}{'n':>6}{'amcl mean':>11}{'amcl p95':>10}{'amcl max':>10}"
            f"{'yaw p95':>9}{'odom mean':>11}{'odom max':>10}")
    lines = [head, "-" * len(head)]
    for s in summary:
        lines.append(
            f"{s['region']:<14}{s['samples']:>6}{s['amcl_mean_cm']:>9.1f}cm{s['amcl_p95_cm']:>8.1f}cm"
            f"{s['amcl_max_cm']:>8.1f}cm{s['amcl_p95_yaw_deg']:>7.1f}d{s['odom_mean_cm']:>9.1f}cm"
            f"{s['odom_max_cm']:>8.1f}cm")
    table = "\n".join(lines)
    (out / "summary.txt").write_text(table + "\n")
    print(table)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return out

    t = np.array([r["t"] for r in rows]) - rows[0]["t"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(t, [r["amcl_pos_err"] * 100 for r in rows], label="AMCL")
    ax1.plot(t, [r["odom_pos_err"] * 100 for r in rows], label="odometry only", alpha=0.7)
    on_terrain = np.array([r["region"] not in ("floor", "unknown") for r in rows])
    ymax = max(1.0, max(r["odom_pos_err"] for r in rows) * 100)
    ax1.fill_between(t, 0, ymax, where=on_terrain, color="orange", alpha=0.15,
                     step="mid", label="on terrain patch")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("position error [cm]")
    ax1.set_title("Localization error vs ground truth")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot([r["truth_x"] for r in rows], [r["truth_y"] for r in rows], "k-", lw=2, label="truth")
    ax2.plot([r["amcl_x"] for r in rows], [r["amcl_y"] for r in rows], "--", label="AMCL")
    ax2.plot([r["odom_x"] for r in rows], [r["odom_y"] for r in rows], ":", label="odometry only")
    ax2.set_aspect("equal")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Trajectory (map frame)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "localization.png", dpi=130)
    print(f"plot: {out / 'localization.png'}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default="run")
    ap.add_argument("--gt-topic", default="/ground_truth_tf")
    ap.add_argument("--gt-child", default="nova_carter")
    ap.add_argument("--map-frame", default="map")
    ap.add_argument("--odom-frame", default="odom")
    ap.add_argument("--rate", type=float, default=10.0, help="samples per second")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds to wait for TF")
    ap.add_argument("--out", default=str(REPO / "results" / "localization"))
    args = ap.parse_args()

    rclpy.init()
    node = LocalizationEval(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rows, misses = node.rows, node.misses
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    print(f"\n{len(rows)} samples, {misses} skipped for missing TF")
    if rows:
        out = write_outputs(rows, args.label, pathlib.Path(args.out))
        print(f"results: {out}")


if __name__ == "__main__":
    main()
