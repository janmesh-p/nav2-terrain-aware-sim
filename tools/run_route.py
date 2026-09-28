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
import time
import pathlib

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.parameter import Parameter
from tf2_msgs.msg import TFMessage

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
    ap.add_argument("--dwell", type=float, default=7.0,
                    help="seconds to hold at each goal before the next one (5 to 10 recommended)")
    ap.add_argument("--truth-tolerance", type=float, default=0.5,
                    help="max true distance to goal [m] for a leg to count as succeeded")
    ap.add_argument("--gt-topic", default="/ground_truth_tf")
    ap.add_argument("--gt-child", default="nova_carter")
    ap.add_argument("--skip-validate", action="store_true",
                    help="drive even if tools/validate_route.py rejects the route")
    args = ap.parse_args()

    if not args.skip_validate:
        from validate_route import validate
        terrain = str(REPO / "scenes" / "configs" / "warehouse_terrain.yaml")
        scene = str(REPO / "scenes" / "scene_obstacles.json")
        if not validate(args.route, terrain, 1.0, scene, None):
            print("route failed validation; fix the goals or pass --skip-validate")
            return

    route = yaml.safe_load(open(args.route))
    rclpy.init()
    nav = BasicNavigator()
    nav.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
    # waitUntilNav2Active(localizer="amcl") publishes a default (0, 0, 0)
    # initial pose if AMCL has not reported one yet, which teleports the
    # estimate. Wait for the lifecycle states only and never touch the pose.
    nav._waitForNodeToActivate("amcl")
    nav._waitForNodeToActivate("bt_navigator")

    # Nav2 judges arrival by its own (AMCL) pose. A lost robot can report
    # success far from the goal, so every leg is re-judged against the true
    # pose from Isaac Sim.
    truth = {"pose": None}

    def on_truth(msg):
        for t in msg.transforms:
            if t.child_frame_id == args.gt_child:
                q = t.transform.rotation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                truth["pose"] = (t.transform.translation.x, t.transform.translation.y, yaw)

    nav.create_subscription(TFMessage, args.gt_topic, on_truth, 10)
    for _ in range(50):
        rclpy.spin_once(nav, timeout_sec=0.1)
        if truth["pose"] is not None:
            break
    if truth["pose"] is None:
        nav.get_logger().error(f"no ground truth on {args.gt_topic}; cannot judge legs")
        return

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
        for _ in range(5):  # let the latest truth pose arrive
            rclpy.spin_once(nav, timeout_sec=0.05)
        tx, ty, tyaw = truth["pose"]
        dist = math.hypot(tx - wp["x"], ty - wp["y"])
        nav_result = "timeout" if timed_out else result.name.lower()
        if nav_result == "succeeded":
            verdict = "succeeded" if dist <= args.truth_tolerance else "false_success"
        else:
            verdict = nav_result
        leg = {
            "name": wp["name"],
            "goal": [wp["x"], wp["y"]],
            "result": verdict,
            "nav2_result": nav_result,
            "truth_final": [round(tx, 3), round(ty, 3)],
            "truth_distance_m": round(dist, 3),
            "truth_yaw_error_deg": round(math.degrees(abs((tyaw - yaw + math.pi) % (2 * math.pi) - math.pi)), 1),
            "sim_seconds": round((nav.get_clock().now() - t0).nanoseconds * 1e-9, 2),
            "recoveries": int(fb.number_of_recoveries) if fb is not None else None,
        }
        legs.append(leg)

        # Hold still so the robot settles and AMCL can converge before the
        # next goal. Wall clock, so a slow simulator still gets a real pause.
        if args.dwell > 0:
            nav.get_logger().info(f"   holding {args.dwell:.0f} s")
            end = time.monotonic() + args.dwell
            while time.monotonic() < end:
                rclpy.spin_once(nav, timeout_sec=0.1)
        nav.get_logger().info(
            f"   {leg['result']} in {leg['sim_seconds']} s, truly {dist:.2f} m from goal")
        prev = (wp["x"], wp["y"])

    out_dir = REPO / "results" / "routes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{args.label}.json"
    out.write_text(json.dumps({"route": route["name"], "label": args.label,
                               "truth_tolerance_m": args.truth_tolerance, "legs": legs}, indent=2))

    ok = sum(1 for l in legs if l["result"] == "succeeded")
    print(f"\n{ok}/{len(legs)} legs succeeded")
    false_ok = sum(1 for l in legs if l["result"] == "false_success")
    if false_ok:
        print(f"{false_ok} leg(s) reported success by Nav2 but were not at the goal")
    for l in legs:
        print(f"  {l['name']:<12} {l['result']:<14} {l['sim_seconds']:>7.1f} s  "
              f"recoveries {l['recoveries']:<3} true dist {l['truth_distance_m']:.2f} m")
    print(f"saved {out}")

    nav.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
