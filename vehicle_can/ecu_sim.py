#!/usr/bin/env python3
"""Simulated drive ECU on a CAN bus. Drives the Isaac Sim robot.

Listens for DRIVE_CMD on the bus, applies the safety rules in ecu_core.py,
and publishes the resulting setpoint to /cmd_vel, which moves the robot in
Isaac Sim. Reports DRIVE_FB (50 Hz) and ECU_HEARTBEAT (10 Hz) back.
Runs on wall-clock time, like real firmware.

    python3 vehicle_can/ecu_sim.py --channel vcan0 --label demo1
"""

import argparse
import datetime as dt
import json
import pathlib
import sys
import threading
import time

import can
import rclpy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as p  # noqa: E402
from ecu_core import EcuCore  # noqa: E402

STATE_NAMES = {0: "disabled", 1: "driving", 2: "safe_stop", 3: "fault"}


class EcuNode(Node):
    def __init__(self, args):
        super().__init__("drive_ecu_sim")
        self.core = EcuCore()
        self.lock = threading.Lock()
        self.bus = can.Bus(channel=args.channel, interface="socketcan",
                           can_filters=[{"can_id": p.DRIVE_CMD.frame_id, "can_mask": 0x7FF}])
        self.t0 = time.monotonic()
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.meas = (0.0, 0.0)
        self.create_subscription(Odometry, "/chassis/odom", self.on_odom, qos_profile_sensor_data)
        self.fb_counter = 0
        self.alive = 0
        out = HERE.parent / "results" / "can"
        out.mkdir(parents=True, exist_ok=True)
        self.log = open(out / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{args.label}_ecu.jsonl", "w", buffering=1)
        self.logged = 0
        self.last_state = None
        self.notifier = can.Notifier(self.bus, [self.on_can])
        # All safety timing runs here, not on the ROS executor: executor
        # stalls (DDS discovery when other nodes start) must not delay the
        # watchdog, the motor output, or the heartbeat.
        self.running = True
        self.loop = threading.Thread(target=self.run_loop, name="ecu_rt", daemon=True)
        self.loop.start()
        self.get_logger().info(f"ECU on {args.channel}, driving /cmd_vel, timing thread at 50 Hz")

    def run_loop(self, period=0.02):
        next_t = time.monotonic()
        n = 0
        while self.running:
            self.tick()
            if n % 5 == 0:
                self.heartbeat()
            n += 1
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()  # overran: resync instead of bursting

    def now(self):
        return time.monotonic() - self.t0

    def on_can(self, msg: can.Message):
        with self.lock:
            self.core.on_frame(bytes(msg.data), self.now())

    def on_odom(self, m):
        self.meas = (m.twist.twist.linear.x, m.twist.twist.angular.z)

    def tick(self):
        with self.lock:
            speed, yaw = self.core.tick(self.now())
            state, faults = self.core.state, dict(self.core.faults)
            new_events = self.core.events[self.logged:]
            self.logged = len(self.core.events)
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = speed, yaw
        self.cmd_pub.publish(cmd)
        self.fb_counter = (self.fb_counter + 1) & 0xF
        self.bus.send(can.Message(arbitration_id=p.DRIVE_FB.frame_id, is_extended_id=False,
                                  data=p.encode_drive_fb(*self.meas, state, faults, self.fb_counter)))
        for e in new_events:
            self.log.write(json.dumps(e) + "\n")
            self.get_logger().info(json.dumps(e))
        if state != self.last_state:
            self.get_logger().info(f"state -> {STATE_NAMES[state]}")
            self.last_state = state

    def heartbeat(self):
        self.alive = (self.alive + 1) & 0xFF
        self.bus.send(can.Message(arbitration_id=p.ECU_HEARTBEAT.frame_id, is_extended_id=False,
                                  data=p.encode_ecu_heartbeat(self.alive, self.now())))

    def shutdown(self):
        self.running = False
        self.loop.join(timeout=1.0)
        for _ in range(3):
            self.cmd_pub.publish(Twist())  # motors off on the way out
        with self.lock:
            stats = dict(self.core.stats)
        self.log.write(json.dumps({"t": round(self.now(), 3), "event": "shutdown", "stats": stats}) + "\n")
        self.log.close()
        self.notifier.stop()
        self.bus.shutdown()
        print(f"ECU frame stats: {stats}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--channel", default="vcan0")
    ap.add_argument("--label", default="run")
    args, _ = ap.parse_known_args()
    # Own the Ctrl+C: rclpy's handler would tear down the context before
    # we could send the final zero command.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = EcuNode(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.shutdown()
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
