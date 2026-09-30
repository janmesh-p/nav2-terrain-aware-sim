import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import protocol as p  # noqa: E402
from ecu_core import EcuCore  # noqa: E402

DT = 0.02


def test_crc_matches_published_check_value():
    assert p.crc8_j1850(b"123456789") == 0x4B


def test_drive_cmd_round_trip():
    f = p.encode_drive_cmd(0.5, -0.25, True, p.MODE_AUTONOMY, False, 7)
    m = p.decode_drive_cmd(f)
    assert (m["speed_cmd"], m["yaw_rate_cmd"], m["enable"], m["mode"], m["counter"]) == (0.5, -0.25, 1, 1, 7)
    assert p.crc_ok(f)


def test_every_single_bit_flip_is_detected():
    f = p.encode_drive_cmd(0.8, 0.1, True, p.MODE_TELEOP, False, 3)
    for byte in range(8):
        for bit in range(8):
            bad = bytearray(f)
            bad[byte] ^= 1 << bit
            assert not p.crc_ok(bytes(bad)), (byte, bit)


class Sender:
    def __init__(self):
        self.counter = 0

    def frame(self, speed=0.5, yaw=0.0, enable=True, mode=p.MODE_AUTONOMY, estop=False):
        self.counter = (self.counter + 1) & 0xF
        return p.encode_drive_cmd(speed, yaw, enable, mode, estop, self.counter)


def armed_ecu():
    ecu, s, t = EcuCore(), Sender(), 0.0
    ecu.on_frame(s.frame(enable=False, mode=p.MODE_STOP), t)
    t += DT
    ecu.on_frame(s.frame(), t)
    assert ecu.tick(t) == (0.5, 0.0) and ecu.state == p.STATE_DRIVING
    return ecu, s, t


def test_does_not_arm_without_enable_low_first():
    ecu, s = EcuCore(), Sender()
    for i in range(10):
        ecu.on_frame(s.frame(), i * DT)
        assert ecu.tick(i * DT) == (0.0, 0.0)


def test_timeout_stops_and_disarms():
    ecu, s, t = armed_ecu()
    t += 0.15
    assert ecu.tick(t) == (0.0, 0.0)
    assert ecu.state == p.STATE_SAFE_STOP and ecu.faults["timeout"]
    ecu.on_frame(s.frame(), t + DT)          # sender comes back
    assert ecu.tick(t + DT) == (0.0, 0.0), "must re-arm before moving again"


def test_single_bad_crc_is_dropped_three_is_a_fault():
    ecu, s, t = armed_ecu()
    bad = bytearray(s.frame())
    bad[0] ^= 0x01
    ecu.on_frame(bytes(bad), t)
    assert ecu.state == p.STATE_DRIVING
    for _ in range(2):
        ecu.on_frame(bytes(bad), t)
    assert ecu.state == p.STATE_FAULT and ecu.faults["crc"]


def test_frozen_counter_is_a_fault_even_with_valid_crc():
    ecu, s, t = armed_ecu()
    stuck = s.frame()
    ecu.on_frame(stuck, t)
    for i in range(3):
        ecu.on_frame(stuck, t + (i + 1) * DT)
    assert p.crc_ok(stuck)
    assert ecu.state == p.STATE_FAULT and ecu.faults["counter"]


def test_out_of_range_is_a_fault():
    ecu, s, t = armed_ecu()
    raw = bytearray(p.DRIVE_CMD.encode({"speed_cmd": 30.0, "yaw_rate_cmd": 0, "enable": 1, "mode": 1,
                                        "estop": 0, "counter": 9, "crc8": 0}, strict=False))
    raw[7] = p.crc8_j1850(bytes(raw[:7]))
    ecu.on_frame(bytes(raw), t)
    assert ecu.state == p.STATE_FAULT and ecu.faults["range"]


def test_estop_bit_stops_immediately():
    ecu, s, t = armed_ecu()
    ecu.on_frame(s.frame(estop=True), t + DT)
    assert ecu.tick(t + DT) == (0.0, 0.0) and ecu.state == p.STATE_SAFE_STOP


def test_rearm_after_fault_clears_flags():
    ecu, s, t = armed_ecu()
    t += 0.2
    ecu.tick(t)
    for i in range(30):
        ecu.on_frame(s.frame(enable=False, mode=p.MODE_STOP), t + i * DT)
    t += 30 * DT
    ecu.on_frame(s.frame(), t)
    assert ecu.tick(t) == (0.5, 0.0)
    assert ecu.state == p.STATE_DRIVING and not any(ecu.faults.values())


def test_fault_stays_visible_until_enable_held_low():
    ecu, s, t = armed_ecu()
    ecu.tick(t + 0.2)                         # timeout fault
    assert ecu.faults["timeout"]
    t += 0.3
    ecu.on_frame(s.frame(enable=False, mode=p.MODE_STOP), t)
    assert ecu.faults["timeout"], "a single enable-low frame must not clear the fault"
    for i in range(1, 30):                    # 0.58 s of enable low
        ecu.on_frame(s.frame(enable=False, mode=p.MODE_STOP), t + i * DT)
    assert ecu.state == p.STATE_DISABLED and not any(ecu.faults.values())
    assert ecu.tick(t + 0.6) == (0.0, 0.0), "clearing faults must not move the vehicle"


def test_stuck_high_enable_after_fault_never_rearms():
    ecu, s, t = armed_ecu()
    ecu.tick(t + 0.2)
    for i in range(20):                       # sender recovers but never drops enable
        ecu.on_frame(s.frame(), t + 0.3 + i * DT)
        assert ecu.tick(t + 0.3 + i * DT) == (0.0, 0.0)
