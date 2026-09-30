import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from arbiter_core import Arbiter, Limits, Mode  # noqa: E402

DT = 0.05


def ready(a: Arbiter, now: float) -> None:
    a.odometry(now)
    a.localization(0.1)


def run_autonomy(a: Arbiter, t: float, lin=0.8, ang=0.0) -> float:
    a.request(Mode.AUTONOMY)
    for _ in range(5):
        ready(a, t)
        a.autonomy_cmd(lin, ang, t)
        d = a.step(t)
        t += DT
    assert d.source == "autonomy"
    return t


def test_starts_stopped():
    a = Arbiter()
    ready(a, 0.0)
    assert a.step(0.0).source == "stop"


def test_autonomy_drives_when_healthy():
    a = Arbiter()
    t = run_autonomy(a, 0.0)
    ready(a, t)
    a.autonomy_cmd(0.5, 0.2, t)
    d = a.step(t)
    assert (d.source, d.linear, d.angular) == ("autonomy", 0.5, 0.2)


def test_estop_latches_until_reset_and_new_request():
    a = Arbiter()
    t = run_autonomy(a, 0.0)
    a.estop()
    for _ in range(40):
        ready(a, t)
        a.autonomy_cmd(0.8, 0.0, t)
        d = a.step(t)
        t += DT
    assert d.reason == "e-stop latched" and d.linear == 0.0
    a.reset()
    ready(a, t)
    a.autonomy_cmd(0.8, 0.0, t)
    assert a.step(t).source == "stop", "reset alone must not resume motion"


def test_teleop_link_loss_stops_and_never_falls_back_to_autonomy():
    a = Arbiter()
    t = run_autonomy(a, 0.0)
    ready(a, t)
    a.teleop_cmd(0.3, 0.0, t)           # operator takes over
    assert a.step(t).reason == "operator takeover"
    t += 1.0                              # teleop stream stops
    for _ in range(30):
        ready(a, t)
        a.autonomy_cmd(0.8, 0.0, t)       # autonomy still publishing
        d = a.step(t)
        t += DT
    assert d.source == "stop" and d.reason == "teleop link lost"


def test_bad_localization_blocks_autonomy_but_not_teleop():
    a = Arbiter()
    t = run_autonomy(a, 0.0)
    a.localization(1.2)
    a.odometry(t)
    a.autonomy_cmd(0.8, 0.0, t)
    assert a.step(t).reason.startswith("localization uncertain")
    a.request(Mode.TELEOP)
    a.odometry(t)
    a.teleop_cmd(0.2, 0.0, t)
    assert a.step(t).source == "teleop"


def test_tilt_fault_overrides_teleop():
    a = Arbiter()
    a.request(Mode.TELEOP)
    ready(a, 0.0)
    a.teleop_cmd(0.3, 0.0, 0.0)
    a.tilt(20.0)
    assert a.step(0.0).reason.startswith("tilt")


def test_stale_odometry_stops():
    a = Arbiter()
    t = run_autonomy(a, 0.0)
    t += 1.0
    a.autonomy_cmd(0.8, 0.0, t)          # no odometry for 1 s
    assert a.step(t).reason == "odometry stale"


def test_stop_ramps_down_at_decel_limit():
    lim = Limits(decel_linear=1.0)
    a = Arbiter(limits=lim)
    t = run_autonomy(a, 0.0, lin=0.8)
    a.estop()
    speeds = []
    for _ in range(25):
        ready(a, t)
        speeds.append(a.step(t).linear)
        t += DT
    assert speeds[0] == pytest.approx(0.8 - 1.0 * DT)
    assert all(b <= a_ for a_, b in zip(speeds, speeds[1:]))
    assert speeds[-1] == 0.0
    # 0.8 m/s at 1 m/s^2 reaches zero after 0.8 s = 16 cycles
    assert speeds.index(0.0) == 15


def test_output_is_clamped():
    a = Arbiter(limits=Limits(max_linear=1.0, max_angular=1.5))
    t = run_autonomy(a, 0.0)
    ready(a, t)
    a.autonomy_cmd(5.0, -9.0, t)
    d = a.step(t)
    assert (d.linear, d.angular) == (1.0, -1.5)


def test_vehicle_fault_stops_and_requires_new_request():
    a = Arbiter()
    t = run_autonomy(a, 0.0)
    a.vehicle_fault("ECU heartbeat lost")
    ready(a, t)
    a.autonomy_cmd(0.8, 0.0, t)
    assert a.step(t).reason == "vehicle: ECU heartbeat lost"
    a.vehicle_fault(None)                      # ECU recovers
    ready(a, t + DT)
    a.autonomy_cmd(0.8, 0.0, t + DT)
    assert a.step(t + DT).source == "stop", "recovery alone must not resume motion"
    a.request(Mode.AUTONOMY)
    ready(a, t + 2 * DT)
    a.autonomy_cmd(0.8, 0.0, t + 2 * DT)
    assert a.step(t + 2 * DT).source == "autonomy"
