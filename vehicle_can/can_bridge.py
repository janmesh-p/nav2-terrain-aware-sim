#!/usr/bin/env python3
"""Computer-side CAN bridge: arbiter output -> DRIVE_CMD, ECU feedback -> ROS.

  /cmd_vel_arbiter   Twist    arbiter's chosen command
  /arbiter/state     String   JSON with the active source and e-stop
  -> vcan0  DRIVE_CMD 50 Hz (fresh counter, CRC), CTRL_HEARTBEAT 10 Hz
  <- vcan0  DRIVE_FB, ECU_HEARTBEAT
  /vehicle/state     String   JSON: ECU state, measured speed, fault flags
  /vehicle/fault     String   "" when healthy, otherwise the reason

Enable is sent high only while the arbiter grants autonomy or teleop and its
output is fresh. If the arbiter goes quiet, the bridge sends enable = 0 and
zero speed rather than repeating a stale command.

    python3 vehicle_can/can_bridge.py --channel vcan0 --label demo1
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
from rclpy.node import Node
from std_msgs.msg import String

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as p  # noqa: E402

SOURCE_MODE = {"autonomy": p.MODE_AUTONOMY, "teleop": p.MODE_TELEOP}
ECU_TIMEOUT_S = 0.3
UPSTREAM_TIMEOUT_S = 0.3
OK_DEBOUNCE_S = 0.5  # a fault is reported for at least this long


class BridgeNode(Node):
    def __init__(self, args):
        super().__init__("can_bridge")
        self.bus = can.Bus(channel=args.channel, interface="socketcan", can_filters=[
            {"can_id": p.DRIVE_FB.frame_id, "can_mask": 0x7FF},
            {"can_id": p.ECU_HEARTBEAT.frame_id, "can_mask": 0x7FF}])
        self.lock = threading.Lock()
        self.cmd = (0.0, 0.0)
        self.cmd_t = None
        self.source, self.estop = "stop", False
        self.counter = 0
        self.alive = 0
        self.ecu_hb_t = None
        self.fb = None
        self.fb_bad_crc = 0
        self.fault = None
        self.fault_seen_t = None
        self.sent = 0
        self.create_subscription(Twist, "/cmd_vel_arbiter", self.on_cmd, 10)
        self.create_subscription(String, "/arbiter/state", self.on_state, 10)
        self.state_pub = self.create_publisher(String, "/vehicle/state", 10)
        self.fault_pub = self.create_publisher(String, "/vehicle/fault", 10)
        out = HERE.parent / "results" / "can"
        out.mkdir(parents=True, exist_ok=True)
        self.log = open(out / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{args.label}_bridge.jsonl", "w", buffering=1)
        self.t0 = time.monotonic()
        self.notifier = can.Notifier(self.bus, [self.on_can])
        self.create_timer(0.02, self.send_cmd)
        self.create_timer(0.1, self.send_heartbeat)
        self.create_timer(0.05, self.check_ecu)
        self.get_logger().info(f"bridge on {args.channel}")

    def now(self):
        return time.monotonic() - self.t0

    def event(self, kind, **kw):
        rec = {"t": round(self.now(), 3), "event": kind, **kw}
        self.log.write(json.dumps(rec) + "\n")
        self.get_logger().info(json.dumps(rec))

    def on_cmd(self, m):
        with self.lock:
            self.cmd, self.cmd_t = (m.linear.x, m.angular.z), time.monotonic()

    def on_state(self, m):
        try:
            s = json.loads(m.data)
        except ValueError:
            return
        with self.lock:
            self.source, self.estop = s.get("source", "stop"), bool(s.get("estop", False))

    def send_cmd(self):
        with self.lock:
            fresh = self.cmd_t is not None and time.monotonic() - self.cmd_t <= UPSTREAM_TIMEOUT_S
            enable = fresh and self.source in SOURCE_MODE and not self.estop
            speed, yaw = self.cmd if enable else (0.0, 0.0)
            mode = SOURCE_MODE.get(self.source, p.MODE_STOP) if enable else p.MODE_STOP
            self.counter = (self.counter + 1) & 0xF
            data = p.encode_drive_cmd(max(-2.0, min(2.0, speed)), max(-3.0, min(3.0, yaw)),
                                      enable, mode, self.estop, self.counter)
        self.bus.send(can.Message(arbitration_id=p.DRIVE_CMD.frame_id, is_extended_id=False, data=data))
        self.sent += 1

    def send_heartbeat(self):
        self.alive = (self.alive + 1) & 0xFF
        src = SOURCE_MODE.get(self.source, 0)
        self.bus.send(can.Message(arbitration_id=p.CTRL_HEARTBEAT.frame_id, is_extended_id=False,
                                  data=p.encode_ctrl_heartbeat(self.alive, src)))

    def on_can(self, msg: can.Message):
        data = bytes(msg.data)
        with self.lock:
            if msg.arbitration_id == p.ECU_HEARTBEAT.frame_id:
                self.ecu_hb_t = time.monotonic()
            elif msg.arbitration_id == p.DRIVE_FB.frame_id:
                if not p.crc_ok(data):
                    self.fb_bad_crc += 1
                    return
                self.fb = p.decode_drive_fb(data)

    def check_ecu(self):
        with self.lock:
            hb_age = None if self.ecu_hb_t is None else time.monotonic() - self.ecu_hb_t
            fb = dict(self.fb) if self.fb else None
        if hb_age is None or hb_age > ECU_TIMEOUT_S:
            fault = "ECU heartbeat lost"
        elif fb and fb["state"] in (p.STATE_SAFE_STOP, p.STATE_FAULT):
            flags = [k[6:] for k in ("fault_timeout", "fault_counter", "fault_crc", "fault_range") if fb[k]]
            fault = f"ECU stopped: {', '.join(flags)}" if flags else None
        else:
            fault = None
        now = time.monotonic()
        if fault:
            self.fault_seen_t = now
        elif self.fault and self.fault_seen_t and now - self.fault_seen_t < OK_DEBOUNCE_S:
            fault = self.fault  # hold the report so the arbiter cannot miss it
        if fault != self.fault:
            self.event("vehicle_fault" if fault else "vehicle_ok", reason=fault)
            self.fault = fault
        self.fault_pub.publish(String(data=fault or ""))
        if fb:
            self.state_pub.publish(String(data=json.dumps({
                "state": fb["state"], "speed": round(fb["speed_meas"], 3),
                "yaw_rate": round(fb["yaw_rate_meas"], 3),
                "faults": {k[6:]: fb[k] for k in fb if k.startswith("fault_")}})))

    def shutdown(self):
        self.log.write(json.dumps({"t": round(self.now(), 3), "event": "shutdown",
                                   "sent": self.sent, "fb_bad_crc": self.fb_bad_crc}) + "\n")
        self.log.close()
        self.notifier.stop()
        self.bus.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--channel", default="vcan0")
    ap.add_argument("--label", default="run")
    args, _ = ap.parse_known_args()
    # Own the Ctrl+C: rclpy's handler would tear down the context before
    # we could send the final zero command.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = BridgeNode(args)
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
