"""Encode and decode the drive messages defined in drive.dbc.

Frames that carry motion commands are protected two ways:
  * crc8: CRC-8 SAE J1850 over bytes 0-6, in byte 7. Catches corrupted bits.
  * counter: 4-bit rolling counter. Catches a sender that froze while its
    CAN controller keeps repeating the last valid frame.
"""

from __future__ import annotations

import pathlib

import cantools

DBC = cantools.database.load_file(str(pathlib.Path(__file__).with_name("drive.dbc")))
DRIVE_CMD = DBC.get_message_by_name("DRIVE_CMD")
CTRL_HEARTBEAT = DBC.get_message_by_name("CTRL_HEARTBEAT")
DRIVE_FB = DBC.get_message_by_name("DRIVE_FB")
ECU_HEARTBEAT = DBC.get_message_by_name("ECU_HEARTBEAT")

MODE_STOP, MODE_AUTONOMY, MODE_TELEOP = 0, 1, 2
STATE_DISABLED, STATE_DRIVING, STATE_SAFE_STOP, STATE_FAULT = 0, 1, 2, 3


def crc8_j1850(data: bytes) -> int:
    """CRC-8 SAE J1850: poly 0x1D, init 0xFF, final xor 0xFF."""
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc ^ 0xFF


def _seal(message, fields: dict) -> bytes:
    raw = bytearray(message.encode({**fields, "crc8": 0}, scaling=True, strict=True))
    raw[7] = crc8_j1850(bytes(raw[:7]))
    return bytes(raw)


def crc_ok(data: bytes) -> bool:
    return len(data) == 8 and crc8_j1850(data[:7]) == data[7]


def encode_drive_cmd(speed: float, yaw_rate: float, enable: bool, mode: int,
                     estop: bool, counter: int) -> bytes:
    return _seal(DRIVE_CMD, {
        "speed_cmd": speed, "yaw_rate_cmd": yaw_rate, "enable": int(enable),
        "mode": mode, "estop": int(estop), "counter": counter & 0xF,
    })


def decode_drive_cmd(data: bytes) -> dict:
    """Decode without range checks: the ECU decides what is out of range."""
    return DRIVE_CMD.decode(data, scaling=True, decode_choices=False, allow_truncated=False)


def encode_drive_fb(speed: float, yaw_rate: float, state: int, faults: dict, counter: int) -> bytes:
    return _seal(DRIVE_FB, {
        "speed_meas": max(-32.0, min(32.0, speed)), "yaw_rate_meas": max(-32.0, min(32.0, yaw_rate)),
        "state": state, "counter": counter & 0xF,
        "fault_timeout": int(faults.get("timeout", 0)), "fault_counter": int(faults.get("counter", 0)),
        "fault_crc": int(faults.get("crc", 0)), "fault_range": int(faults.get("range", 0)),
    })


def decode_drive_fb(data: bytes) -> dict:
    return DRIVE_FB.decode(data, scaling=True, decode_choices=False)


def encode_ctrl_heartbeat(alive: int, source: int) -> bytes:
    return CTRL_HEARTBEAT.encode({"alive": alive & 0xFF, "source": source})


def encode_ecu_heartbeat(alive: int, uptime: float) -> bytes:
    return ECU_HEARTBEAT.encode({"alive": alive & 0xFF, "uptime": min(uptime, 167772.15)})
