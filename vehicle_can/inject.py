#!/usr/bin/env python3
"""Fault injection on the CAN bus, for testing the ECU's defences.

    # corrupted frames: a burst of DRIVE_CMD with one flipped bit
    python3 vehicle_can/inject.py corrupt --count 10

    # stuck sender: capture one real DRIVE_CMD and replay it unchanged.
    # Freeze the bridge first (kill -STOP) so it stops sending fresh frames.
    python3 vehicle_can/inject.py replay --seconds 2

    # out-of-range command with a valid CRC and a new counter
    python3 vehicle_can/inject.py range --speed 30
"""

import argparse
import pathlib
import sys
import time

import can

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import protocol as p  # noqa: E402


def capture(bus, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        m = bus.recv(timeout=0.1)
        if m and m.arbitration_id == p.DRIVE_CMD.frame_id and p.crc_ok(bytes(m.data)):
            return bytes(m.data)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=["corrupt", "replay", "range"])
    ap.add_argument("--channel", default="vcan0")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--speed", type=float, default=30.0)
    args = ap.parse_args()
    bus = can.Bus(channel=args.channel, interface="socketcan")
    send = lambda d: bus.send(can.Message(arbitration_id=p.DRIVE_CMD.frame_id,  # noqa: E731
                                          is_extended_id=False, data=d))

    if args.kind == "corrupt":
        base = capture(bus) or p.encode_drive_cmd(0.3, 0.0, True, p.MODE_AUTONOMY, False, 1)
        bad = bytearray(base)
        bad[1] ^= 0x10
        for _ in range(args.count):
            send(bytes(bad))
            time.sleep(0.001)  # faster than the bridge, so they arrive back to back
        print(f"sent {args.count} corrupted frames")
    elif args.kind == "replay":
        frame = capture(bus, timeout=0.5)
        if frame is None:
            frame = p.encode_drive_cmd(0.3, 0.0, True, p.MODE_AUTONOMY, False, 5)
            print("no live frame seen (bridge frozen?), replaying a synthetic one")
        n = int(args.seconds / 0.02)
        for _ in range(n):
            send(frame)
            time.sleep(0.02)
        print(f"replayed one frame {n} times, counter {p.decode_drive_cmd(frame)['counter']}")
    else:
        raw = bytearray(p.DRIVE_CMD.encode({"speed_cmd": args.speed, "yaw_rate_cmd": 0, "enable": 1,
                                            "mode": 1, "estop": 0, "counter": 11, "crc8": 0}, strict=False))
        raw[7] = p.crc8_j1850(bytes(raw[:7]))
        send(bytes(raw))
        print(f"sent out-of-range command: {args.speed} m/s")
    bus.shutdown()


if __name__ == "__main__":
    main()
