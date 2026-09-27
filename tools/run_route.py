#!/usr/bin/env python3
"""Drive a fixed waypoint route through Nav2 and log the outcome of each leg.

    python3 tools/run_route.py                       # default terrain tour
    python3 tools/run_route.py --route bringup/routes/terrain_tour.yaml --label terrain

Each goal is sent after the previous one finishes, so a failed leg does not
stop the run. Results go to results/routes/<time>_<label>.json.
"""

import argparse
import datetime as dt
import json
import math
import pathlib

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.parameter import Parameter

REPO = pathlib.Path(__file__).resolve().parents[1]


def make_pose(nav: BasicNavigator, x: float, y: float, yaw: float) -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = "map"
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.orientation.z = math.sin(yaw / 2.0)
    p.pose.orientation.w = math.cos(yaw / 2.0)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--route", default=str(REPO / "bringup" / "routes" / "terrain_tour.yaml"))
    ap.add_argument("--label", default="run")
    ap.add_argument("--leg-timeout", type=float, default=180.0, help="seconds per leg")
    args = ap.parse_args()

    route = yaml.safe_load(open(args.route))
    rclpy.init()
    nav = BasicNavigator()
    nav.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
    # waitUntilNav2Active(localizer="amcl") publishes a default (0, 0, 0)
    # initial pose if AMCL has not reported one yet, which teleports the
    # estimate. Wait for the lifecycle states only and never touch the pose.
    nav._waitForNodeToActivate("amcl")
    nav._waitForNodeToActivate("bt_navigator")

    prev = (route["start"]["x"], route["start"]["y"])
    legs = []
    for wp in route["waypoints"]:
        yaw = math.atan2(wp["y"] - prev[1], wp["x"] - prev[0])
        nav.get_logger().info(f"-> {wp['name']} ({wp['x']:.1f}, {wp['y']:.1f})")
        t0 = nav.get_clock().now()
        nav.goToPose(make_pose(nav, wp["x"], wp["y"], yaw))

        timed_out = False
        while not nav.isTaskComplete():
            fb = nav.getFeedback()
            elapsed = (nav.get_clock().now() - t0).nanoseconds * 1e-9
            if elapsed > args.leg_timeout:
                nav.cancelTask()
                timed_out = True
            if fb is not None:
                nav.get_logger().info(
                    f"   {fb.distance_remaining:5.2f} m left, recoveries {fb.number_of_recoveries}",
                    throttle_duration_sec=5.0)

        result = nav.getResult()
        fb = nav.getFeedback()
        leg = {
            "name": wp["name"],
            "goal": [wp["x"], wp["y"]],
            "result": "timeout" if timed_out else result.name.lower(),
            "sim_seconds": round((nav.get_clock().now() - t0).nanoseconds * 1e-9, 2),
            "recoveries": int(fb.number_of_recoveries) if fb is not None else None,
        }
        legs.append(leg)
        nav.get_logger().info(f"   {leg['result']} in {leg['sim_seconds']} s")
        prev = (wp["x"], wp["y"])

    out_dir = REPO / "results" / "routes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{args.label}.json"
    out.write_text(json.dumps({"route": route["name"], "label": args.label, "legs": legs}, indent=2))

    ok = sum(1 for l in legs if l["result"] == "succeeded")
    print(f"\n{ok}/{len(legs)} legs succeeded")
    for l in legs:
        print(f"  {l['name']:<12} {l['result']:<10} {l['sim_seconds']:>7.1f} s  recoveries {l['recoveries']}")
    print(f"saved {out}")

    nav.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
