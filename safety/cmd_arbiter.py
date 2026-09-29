#!/usr/bin/env python3
"""ROS 2 node: arbitrates autonomy, teleop and safe stop into one /cmd_vel.

Inputs
  /cmd_vel_nav              geometry_msgs/Twist   Nav2 output (collision monitor)
  /cmd_vel_teleop           geometry_msgs/Twist   remote operator stream
  /arbiter/request_mode     std_msgs/String       "autonomy" | "teleop" | "stop"
  /arbiter/estop            std_msgs/Bool         true latches the e-stop
  /arbiter/reset            std_msgs/Empty        clears the e-stop, stays stopped
  /chassis/odom             nav_msgs/Odometry     liveness of the drive feedback
  /chassis/imu              sensor_msgs/Imu       body tilt
  /amcl_pose                PoseWithCovarianceStamped   localization confidence
Outputs
  /cmd_vel                  geometry_msgs/Twist   published every cycle, 20 Hz
  /arbiter/state            std_msgs/String       JSON: source, reason, command
  results/arbiter/<time>_<label>.jsonl   every decision change and standstill

    python3 safety/cmd_arbiter.py --label demo1
"""

import argparse
import datetime as dt
import json
import math
import pathlib
import sys

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Empty, String

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arbiter_core import Arbiter, Limits, Mode  # noqa: E402

REPO = HERE.parent


def tilt_from_imu(msg: Imu) -> float | None:
    q = msg.orientation
    n = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
    if msg.orientation_covariance[0] >= 0.0 and n > 0.5:
        c = 1.0 - 2.0 * (q.x * q.x + q.y * q.y) / n
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))
    a = msg.linear_acceleration
    g = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
    if g < 1e-3:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, a.z / g))))


class ArbiterNode(Node):
    def __init__(self, args):
        super().__init__("cmd_arbiter")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.core = Arbiter(limits=Limits(
            max_tilt_deg=args.max_tilt, max_localization_std_m=args.max_loc_std,
            teleop_timeout_s=args.teleop_timeout, max_linear=args.max_linear))
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "/arbiter/state", 10)

        now = lambda: self.get_clock().now().nanoseconds * 1e-9  # noqa: E731
        self.now = now
        self.create_subscription(Twist, "/cmd_vel_nav",
                                 lambda m: self.core.autonomy_cmd(m.linear.x, m.angular.z, now()), 10)
        self.create_subscription(Twist, "/cmd_vel_teleop", self.on_teleop, 10)
        self.create_subscription(String, "/arbiter/request_mode", self.on_request, 10)
        self.create_subscription(Bool, "/arbiter/estop", self.on_estop, 10)
        self.create_subscription(Empty, "/arbiter/reset", self.on_reset, 10)
        self.create_subscription(Odometry, "/chassis/odom", lambda m: self.core.odometry(now()),
                                 qos_profile_sensor_data)
        self.create_subscription(Imu, "/chassis/imu", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.on_amcl, 10)

        out = REPO / "results" / "arbiter"
        out.mkdir(parents=True, exist_ok=True)
        self.log_path = out / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{args.label}.jsonl"
        self.log = open(self.log_path, "w", buffering=1)
        self.last = None
        self.stopping_since = None
        self.create_timer(0.05, self.tick)
        self.get_logger().info(f"arbiter up, logging to {self.log_path}")

    def event(self, kind, **fields):
        rec = {"t": round(self.now(), 3), "event": kind, **fields}
        self.log.write(json.dumps(rec) + "\n")
        self.get_logger().info(json.dumps(rec))

    def on_teleop(self, m):
        before = self.core.requested
        self.core.teleop_cmd(m.linear.x, m.angular.z, self.now())
        if before == Mode.AUTONOMY and self.core.requested == Mode.TELEOP:
            self.event("takeover")

    def on_request(self, m):
        try:
            self.core.request(Mode(m.data.strip().lower()))
            self.event("request", mode=m.data.strip().lower())
        except ValueError:
            self.get_logger().warn(f"unknown mode '{m.data}'")

    def on_estop(self, m):
        if m.data:
            self.core.estop()
            self.event("estop")

    def on_reset(self, _):
        self.core.reset()
        self.event("reset")

    def on_imu(self, m):
        t = tilt_from_imu(m)
        if t is not None:
            self.core.tilt(t)

    def on_amcl(self, m):
        c = m.pose.covariance
        self.core.localization(math.sqrt(max(c[0], c[7], 0.0)))

    def tick(self):
        now = self.now()
        d = self.core.step(now)
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = d.linear, d.angular
        self.pub.publish(cmd)

        key = (d.source, d.reason)
        if key != self.last:
            self.event("decision", source=d.source, reason=d.reason,
                       linear=round(d.linear, 3), angular=round(d.angular, 3))
            self.stopping_since = now if d.source == "stop" and (d.linear or d.angular) else None
            self.last = key
        if self.stopping_since is not None and d.linear == 0.0 and d.angular == 0.0:
            self.event("standstill", after_s=round(now - self.stopping_since, 3))
            self.stopping_since = None

        self.state_pub.publish(String(data=json.dumps(
            {"source": d.source, "reason": d.reason, "linear": round(d.linear, 3),
             "angular": round(d.angular, 3), "requested": self.core.requested.value,
             "estop": self.core.estop_latched})))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default="run")
    ap.add_argument("--max-tilt", type=float, default=15.0)
    ap.add_argument("--max-loc-std", type=float, default=1.0)
    ap.add_argument("--teleop-timeout", type=float, default=0.3)
    ap.add_argument("--max-linear", type=float, default=1.0)
    args, _ = ap.parse_known_args()
    rclpy.init()
    node = ArbiterNode(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        # Last word on the bus is always zero.
        try:
            node.pub.publish(Twist())
            node.log.close()
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
    print(f"events: {node.log_path}")


if __name__ == "__main__":
    main()
