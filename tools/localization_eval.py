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
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage

REPO = pathlib.Path(__file__).resolve().parents[1]


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def roll_pitch_deg(q) -> tuple:
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    return math.degrees(roll), math.degrees(pitch)


def tilt_deg(roll_deg: float, pitch_deg: float) -> float:
    """Angle between the body z axis and world vertical."""
    c = math.cos(math.radians(roll_deg)) * math.cos(math.radians(pitch_deg))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def pose2d(tf) -> tuple:
    t = tf.transform.translation
    return (t.x, t.y, yaw_of(tf.transform.rotation))


def compose(a: tuple, b: tuple) -> tuple:
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def inverse(a: tuple) -> tuple:
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], wrap(-a[2]))


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
        # Truth stamps can come from a different sim clock than /clock (for
        # example a time node that does not reset on Stop). Measure the offset
        # against /clock and remove it when it is clearly not just latency.
        self.clock_ns = None
        self.offsets = collections.deque(maxlen=50)
        self.offset_ns = 0
        self.offset_reported = False
        self.create_subscription(Clock, "/clock", self.on_clock, qos_profile_sensor_data)
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

    def on_clock(self, msg: Clock):
        self.clock_ns = Time.from_msg(msg.clock).nanoseconds

    def _correct(self, raw_ns: int) -> int:
        if self.clock_ns is None:
            return raw_ns
        self.offsets.append(raw_ns - self.clock_ns)
        median = int(np.median(self.offsets))
        if abs(median) > 1e9:
            self.offset_ns = median
            if not self.offset_reported and len(self.offsets) >= 10:
                self.get_logger().warn(
                    f"truth clock is {median * 1e-9:+.2f} s off /clock, correcting")
                self.offset_reported = True
        return raw_ns - self.offset_ns

    def on_truth(self, msg: TFMessage):
        tf = next((t for t in msg.transforms if t.child_frame_id == self.args.gt_child), None)
        if tf is None:
            return
        self.gt_count += 1
        raw_ns = Time.from_msg(tf.header.stamp).nanoseconds
        stamp_ns = self._correct(raw_ns)
        if self.clock_ns is None or (len(self.offsets) < 10 and abs(raw_ns - self.clock_ns) > 1e9):
            return  # wait until the offset estimate is stable
        self.latest_ns = max(self.latest_ns, stamp_ns)
        if self.last_ns is not None and stamp_ns - self.last_ns < 1e9 / self.args.rate:
            return
        self.last_ns = stamp_ns
        self.pending.append((stamp_ns, pose2d(tf), roll_pitch_deg(tf.transform.rotation)))

    def process(self):
        ready_before = self.latest_ns - int(self.args.delay * 1e9)
        while self.pending and self.pending[0][0] <= ready_before:
            stamp_ns, truth, rp = self.pending.popleft()
            self.evaluate(Time(nanoseconds=stamp_ns), truth, rp)

    def evaluate(self, stamp: Time, truth: tuple, rp: tuple):
        if self.robot_frame is None and not self._pick_robot_frame(stamp):
            self.misses += 1
            return
        try:
            amcl = pose2d(self._lookup(self.args.map_frame, self.robot_frame, stamp))
            odom_base = pose2d(self._lookup(self.args.odom_frame, self.robot_frame, stamp))
            if self.map_odom0 is None:
                # Anchor odometry to the true start pose, so the odometry-only
                # curve shows pure odometry drift, free of AMCL's initial error.
                self.map_odom0 = compose(truth, inverse(odom_base))
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
            "roll_deg": rp[0], "pitch_deg": rp[1], "tilt_deg": tilt_deg(*rp),
            "region": self.terrain.region(truth[0], truth[1]),
        })


TILT_BANDS = [(0.0, 1.0), (1.0, 3.0), (3.0, 90.0)]
EPISODE_TILT_DEG = 1.5    # tilt above this starts an episode
FLAT_TILT_DEG = 0.5       # control windows must stay below this
EPISODE_MERGE_S = 1.0     # join episodes separated by shorter gaps
EPISODE_MIN_S = 0.5       # ignore blips shorter than this
WINDOW_S = 2.0            # error is measured this long before and after
CONTROLS_PER_EPISODE = 20


def _f(rows: list, key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows])


def summarize(rows: list) -> list:
    out = []
    for region in ["all"] + sorted({r["region"] for r in rows}):
        sel = rows if region == "all" else [r for r in rows if r["region"] == region]
        if not sel:
            continue
        a = _f(sel, "amcl_pos_err") * 100
        ay = np.degrees(_f(sel, "amcl_yaw_err"))
        o = _f(sel, "odom_pos_err") * 100
        entry = {
            "region": region, "samples": len(sel),
            "amcl_mean_cm": a.mean(), "amcl_p95_cm": np.percentile(a, 95), "amcl_max_cm": a.max(),
            "amcl_p95_yaw_deg": np.percentile(ay, 95),
            "odom_mean_cm": o.mean(), "odom_max_cm": o.max(),
        }
        if "tilt_deg" in sel[0]:
            entry["tilt_p95_deg"] = float(np.percentile(_f(sel, "tilt_deg"), 95))
        out.append(entry)
    return out


def _has_tilt(rows: list) -> bool:
    return bool(rows) and "tilt_deg" in rows[0] and rows[0]["tilt_deg"] not in ("", None)


def find_episodes(t: np.ndarray, tilt: np.ndarray) -> list:
    """Contiguous stretches with tilt above EPISODE_TILT_DEG, as (t_start, t_end)."""
    above = tilt > EPISODE_TILT_DEG
    raw, i = [], 0
    while i < len(t):
        if above[i]:
            j = i
            while j + 1 < len(t) and above[j + 1]:
                j += 1
            raw.append([t[i], t[j]])
            i = j + 1
        else:
            i += 1
    merged = []
    for ep in raw:
        if merged and ep[0] - merged[-1][1] < EPISODE_MERGE_S:
            merged[-1][1] = ep[1]
        else:
            merged.append(ep)
    return [(a, b) for a, b in merged if b - a >= EPISODE_MIN_S]


def _window_delta(t: np.ndarray, err: np.ndarray, t0: float, t1: float):
    """Median error after the window minus median error before it."""
    before = err[(t >= t0 - WINDOW_S) & (t < t0)]
    after = err[(t > t1) & (t <= t1 + WINDOW_S)]
    if len(before) < 3 or len(after) < 3:
        return None
    return float(np.median(after) - np.median(before))


def episode_analysis(rows: list, seed: int = 0) -> dict | None:
    """Error change across tilt episodes versus matched flat-floor windows.

    AMCL corrects in discrete jumps, so per-sample error rates are dominated
    by those jumps. Comparing error just before and just after each tilt
    episode, against identical windows on flat floor, isolates what the tilt
    itself did.
    """
    if not _has_tilt(rows):
        return None
    t = _f(rows, "t")
    tilt = _f(rows, "tilt_deg")
    err = _f(rows, "amcl_pos_err") * 100
    regions = np.array([r["region"] for r in rows])
    rng = np.random.default_rng(seed)

    episodes = []
    for t0, t1 in find_episodes(t, tilt):
        d = _window_delta(t, err, t0, t1)
        if d is None:
            continue
        m = (t >= t0) & (t <= t1)
        vals, counts = np.unique(regions[m], return_counts=True)
        episodes.append({
            "t_start": float(t0), "duration_s": float(t1 - t0),
            "peak_tilt_deg": float(tilt[m].max()),
            "region": str(vals[counts.argmax()]),
            "error_before_cm": float(np.median(err[(t >= t0 - WINDOW_S) & (t < t0)])),
            "delta_cm": d,
        })

    flat_ok = (tilt < FLAT_TILT_DEG) & (regions == "floor")
    controls = []
    for ep in episodes:
        dur = ep["duration_s"]
        lo, hi = t[0] + WINDOW_S, t[-1] - dur - WINDOW_S
        if hi <= lo:
            continue
        for c0 in rng.uniform(lo, hi, 200):
            c1 = c0 + dur
            span = (t >= c0 - WINDOW_S) & (t <= c1 + WINDOW_S)
            if span.sum() < 6 or not flat_ok[span].all():
                continue
            d = _window_delta(t, err, c0, c1)
            if d is not None:
                controls.append(d)
            if len(controls) >= CONTROLS_PER_EPISODE * (episodes.index(ep) + 1):
                break

    bands = []
    for lo_b, hi_b in TILT_BANDS:
        m = (tilt >= lo_b) & (tilt < hi_b)
        if m.any():
            bands.append({
                "band": f"{lo_b:g}-{hi_b:g} deg" if hi_b < 90 else f">{lo_b:g} deg",
                "samples": int(m.sum()), "time_share": float(m.mean()),
                "amcl_mean_cm": float(err[m].mean()),
            })
    level_r = float(np.corrcoef(tilt, err)[0, 1]) if tilt.std() > 0 and err.std() > 0 else float("nan")
    return {"episodes": episodes, "controls": controls, "bands": bands,
            "pearson_tilt_vs_error": level_r}


def compare(ep_deltas: list, ctrl_deltas: list, seed: int = 0, n_perm: int = 5000,
            n_boot: int = 5000) -> dict:
    """Median difference, episodes minus control, with a one-sided permutation
    p-value and a bootstrap 95% interval.

    Medians, because flat-floor windows occasionally catch AMCL snapping back
    from a large failure (hundreds of cm), which would dominate a mean.
    """
    a, b = np.asarray(ep_deltas, float), np.asarray(ctrl_deltas, float)
    out = {"episodes": len(a), "controls": len(b)}
    if len(a) == 0 or len(b) == 0:
        return out
    rng = np.random.default_rng(seed)
    diff = float(np.median(a) - np.median(b))

    pooled = np.concatenate([a, b])
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if np.median(pooled[:len(a)]) - np.median(pooled[len(a):]) >= diff:
            count += 1

    boots = np.array([
        np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b)))
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    out.update({
        "episode_median_delta_cm": float(np.median(a)),
        "episode_mean_delta_cm": float(a.mean()),
        "episode_share_worse": float((a > 0).mean()),
        "control_median_delta_cm": float(np.median(b)),
        "control_mean_delta_cm": float(b.mean()),
        "control_share_worse": float((b > 0).mean()),
        "median_difference_cm": diff,
        "ci95_cm": [float(lo), float(hi)],
        "p_value_one_sided": (count + 1) / (n_perm + 1),
    })
    return out


def format_tables(summary: list, epi: dict | None, stats: dict | None) -> str:
    head = (f"{'region':<14}{'n':>6}{'amcl mean':>11}{'amcl p95':>10}{'amcl max':>10}"
            f"{'yaw p95':>9}{'odom mean':>11}{'tilt p95':>10}")
    lines = [head, "-" * len(head)] if summary else []
    for s in summary:
        tp = f"{s['tilt_p95_deg']:>8.1f}d" if "tilt_p95_deg" in s else f"{'n/a':>9}"
        lines.append(
            f"{s['region']:<14}{s['samples']:>6}{s['amcl_mean_cm']:>9.1f}cm{s['amcl_p95_cm']:>8.1f}cm"
            f"{s['amcl_max_cm']:>8.1f}cm{s['amcl_p95_yaw_deg']:>7.1f}d{s['odom_mean_cm']:>9.1f}cm {tp}")
    if epi:
        lines += ["", f"{'tilt band':<12}{'n':>6}{'time':>7}{'amcl mean':>11}", "-" * 36]
        for b in epi["bands"]:
            lines.append(f"{b['band']:<12}{b['samples']:>6}{100 * b['time_share']:>6.0f}%"
                         f"{b['amcl_mean_cm']:>9.1f}cm")
        lines.append(f"correlation tilt vs error level {epi['pearson_tilt_vs_error']:+.2f}")
        lines += ["", f"{'episode at':<12}{'dur':>6}{'peak tilt':>11}{'region':>15}{'before':>9}{'change':>10}",
                  "-" * 63]
        for e in epi["episodes"]:
            lines.append(f"{e['t_start']:>8.1f} s {e['duration_s']:>5.1f}s{e['peak_tilt_deg']:>9.1f}d"
                         f"{e['region']:>15}{e['error_before_cm']:>7.1f}cm{e['delta_cm']:>+8.1f}cm")
    if stats and "median_difference_cm" in stats:
        lo, hi = stats["ci95_cm"]
        lines += [
            "",
            f"tilt episodes  n={stats['episodes']:<4} median change {stats['episode_median_delta_cm']:+6.1f} cm "
            f"(mean {stats['episode_mean_delta_cm']:+.1f}), worse in {100 * stats['episode_share_worse']:.0f}%",
            f"flat control   n={stats['controls']:<4} median change {stats['control_median_delta_cm']:+6.1f} cm "
            f"(mean {stats['control_mean_delta_cm']:+.1f}), worse in {100 * stats['control_share_worse']:.0f}%",
            f"median difference {stats['median_difference_cm']:+.1f} cm, 95% CI [{lo:+.1f}, {hi:+.1f}], "
            f"one-sided permutation p = {stats['p_value_one_sided']:.3f}",
        ]
    return "\n".join(lines)


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return None


def _episode_panels(ax_bar, ax_box, episodes: list, controls: list, stats: dict | None):
    colors = {"floor": "tab:gray", "bump_field": "tab:green",
              "rough_iso_f": "tab:red", "ramp_7deg": "tab:purple"}
    ax_bar.bar(range(len(episodes)), [e["delta_cm"] for e in episodes],
               color=[colors.get(e["region"], "tab:blue") for e in episodes])
    ax_bar.axhline(0, color="k", lw=0.6)
    ax_bar.set_xticks(range(len(episodes)))
    ax_bar.set_xticklabels([f"{e['peak_tilt_deg']:.0f}d" for e in episodes], fontsize=8)
    ax_bar.set_xlabel("tilt episode (peak tilt)")
    ax_bar.set_ylabel("AMCL error change [cm]")
    ax_bar.set_title("Error change across each tilt episode")
    handles = [plt_patch(c, r) for r, c in colors.items() if any(e["region"] == r for e in episodes)]
    if handles:
        ax_bar.legend(handles=handles, fontsize=8)
    ax_bar.grid(alpha=0.3, axis="y")

    data = [d for d in (np.asarray([e["delta_cm"] for e in episodes]), np.asarray(controls)) if True]
    ax_box.boxplot(data, showfliers=True)
    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels([f"tilt episodes\nn={len(episodes)}", f"flat control\nn={len(controls)}"])
    ax_box.axhline(0, color="k", lw=0.6)
    ax_box.set_ylabel("AMCL error change [cm]")
    title = "Tilt vs flat floor"
    if stats and "p_value_one_sided" in stats:
        title += f" (median diff {stats['median_difference_cm']:+.1f} cm, p={stats['p_value_one_sided']:.3f})"
    ax_box.set_title(title)
    ax_box.grid(alpha=0.3, axis="y")


def plt_patch(color, label):
    from matplotlib.patches import Patch
    return Patch(color=color, label=label)


def plot(rows: list, epi: dict | None, stats: dict | None, path: pathlib.Path) -> None:
    plt = _mpl()
    if plt is None:
        return
    t = _f(rows, "t") - float(rows[0]["t"])
    amcl = _f(rows, "amcl_pos_err") * 100
    odom = _f(rows, "odom_pos_err") * 100
    on_terrain = np.array([r["region"] not in ("floor", "unknown") for r in rows])

    if epi:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes = np.atleast_1d(axes).ravel()

    ax = axes[0]
    ax.plot(t, amcl, label="AMCL")
    ax.plot(t, odom, label="odometry only", alpha=0.7)
    ymax = max(1.0, amcl.max(), odom.max())
    ax.fill_between(t, 0, ymax, where=on_terrain, color="orange", alpha=0.15, step="mid",
                    label="on terrain patch")
    ax.set_xlabel("sim time [s]")
    ax.set_ylabel("position error [cm]")
    ax.set_title("Localization error vs ground truth")
    ax.grid(alpha=0.3)
    if epi:
        ax2 = ax.twinx()
        ax2.plot(t, _f(rows, "tilt_deg"), color="gray", lw=0.8, alpha=0.8, label="tilt")
        ax2.axhline(EPISODE_TILT_DEG, color="gray", ls=":", lw=0.8)
        ax2.set_ylabel("tilt [deg]")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left")
    else:
        ax.legend()

    ax = axes[1]
    ax.plot(_f(rows, "truth_x"), _f(rows, "truth_y"), "k-", lw=2, label="truth")
    ax.plot(_f(rows, "amcl_x"), _f(rows, "amcl_y"), "--", label="AMCL")
    ax.plot(_f(rows, "odom_x"), _f(rows, "odom_y"), ":", label="odometry only")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory (map frame)")
    ax.legend()
    ax.grid(alpha=0.3)

    if epi:
        _episode_panels(axes[2], axes[3], epi["episodes"], epi["controls"], stats)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"plot: {path}")


def analyze(rows: list, out: pathlib.Path) -> dict | None:
    summary = summarize(rows)
    epi = episode_analysis(rows)
    stats = compare([e["delta_cm"] for e in epi["episodes"]], epi["controls"]) if epi else None
    with open(out / "summary.json", "w") as f:
        json.dump({"regions": summary, "episodes": epi, "tilt_vs_flat": stats}, f, indent=2, default=float)
    table = format_tables(summary, epi, stats)
    (out / "summary.txt").write_text(table + "\n")
    print(table)
    plot(rows, epi, stats, out / "localization.png")
    return epi


def write_outputs(rows: list, label: str, out_root: pathlib.Path) -> pathlib.Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_root / f"{stamp}_{label}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    analyze(rows, out)
    return out


def _load_csv(path: str) -> list:
    with open(path) as f:
        return list(csv.DictReader(f))


def reanalyze(csv_paths: list, pooled_out: str | None = None) -> None:
    """Rebuild per-run outputs from saved samples.csv files, and pool episodes
    across runs when more than one is given. No ROS needed."""
    all_eps, all_ctrl, per_run = [], [], []
    for path in csv_paths:
        rows = _load_csv(path)
        if not rows:
            print(f"{path}: no samples")
            continue
        print(f"\n=== {path}")
        epi = analyze(rows, pathlib.Path(path).parent)
        if epi:
            all_eps += epi["episodes"]
            all_ctrl += epi["controls"]
            per_run.append((pathlib.Path(path).parent.name, epi))
        else:
            print("(no tilt data in this run, skipped for pooling)")

    if len(per_run) < 2:
        return
    stats = compare([e["delta_cm"] for e in all_eps], all_ctrl)
    out = pathlib.Path(pooled_out) if pooled_out else REPO / "results" / "localization" / "pooled"
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"pooled over {len(per_run)} runs: " + ", ".join(n for n, _ in per_run), ""]
    by_region = {}
    for e in all_eps:
        by_region.setdefault(e["region"], []).append(e["delta_cm"])
    lines.append(f"{'region':<14}{'episodes':>9}{'median change':>15}{'worse':>8}")
    lines.append("-" * 44)
    for r, d in sorted(by_region.items()):
        d = np.asarray(d)
        lines.append(f"{r:<14}{len(d):>9}{np.median(d):>+13.1f}cm{100 * (d > 0).mean():>7.0f}%")
    lines += ["", format_tables([], None, stats).strip()]
    text = "\n".join(lines)
    (out / "pooled_summary.txt").write_text(text + "\n")
    with open(out / "pooled_summary.json", "w") as f:
        json.dump({"runs": [n for n, _ in per_run], "tilt_vs_flat": stats,
                   "episodes": all_eps}, f, indent=2, default=float)
    print("\n=== pooled\n" + text)

    plt = _mpl()
    if plt is not None:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        _episode_panels(a1, a2, all_eps, all_ctrl, stats)
        fig.tight_layout()
        fig.savefig(out / "pooled_episodes.png", dpi=130)
        plt.close(fig)
        print(f"plot: {out / 'pooled_episodes.png'}")


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
    ap.add_argument("--analyze", nargs="+", metavar="SAMPLES_CSV",
                    help="re-run analysis on saved runs; several files are also pooled")
    ap.add_argument("--pooled-out", help="output folder for pooled results")
    args = ap.parse_args()

    if args.analyze:
        reanalyze(args.analyze, args.pooled_out)
        return

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
