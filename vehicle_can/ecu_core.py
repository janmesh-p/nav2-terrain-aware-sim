"""Drive ECU safety logic, independent of CAN hardware and ROS.

The ECU does not trust the computer. It drives only while commands keep
arriving, intact, fresh, in range, and after a deliberate arming sequence.

  * No valid new command for timeout_s              -> safe stop
  * Bad CRC: frame dropped; crc_fault_after in a row -> fault
  * Same counter repeated counter_fault_after times -> fault (stuck sender)
  * Speed or yaw rate outside limits                -> fault
  * estop bit set                                    -> safe stop
After a stop or fault the ECU is disarmed. It re-arms only after a valid
frame stream with enable = 0 held for clear_after_s (which clears the fault
flags) followed by enable = 1. The human decision to drive again lives upstream, in the arbiter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import protocol as p


@dataclass
class EcuLimits:
    timeout_s: float = 0.1
    crc_fault_after: int = 3
    counter_fault_after: int = 3
    max_speed: float = 2.0
    max_yaw_rate: float = 3.0
    clear_after_s: float = 0.5   # enable must stay low this long to acknowledge a fault


@dataclass
class EcuCore:
    limits: EcuLimits = field(default_factory=EcuLimits)
    state: int = p.STATE_DISABLED
    armed: bool = False
    saw_enable_low: bool = False
    enable_low_since: float | None = None
    faults: dict = field(default_factory=lambda: {"timeout": 0, "counter": 0, "crc": 0, "range": 0})
    speed: float = 0.0
    yaw_rate: float = 0.0
    last_valid_t: float | None = None
    last_counter: int | None = None
    bad_crc_run: int = 0
    repeat_run: int = 0
    stats: dict = field(default_factory=lambda: {"accepted": 0, "bad_crc": 0, "repeated": 0, "range": 0})
    events: list = field(default_factory=list)

    def _event(self, now, kind, **kw):
        self.events.append({"t": round(now, 3), "event": kind, **kw})

    def _stop(self, now, state, reason):
        if self.state != state or self.armed:
            self._event(now, "stop", state=state, reason=reason)
        self.state, self.armed, self.saw_enable_low = state, False, False
        self.enable_low_since = None
        self.speed = self.yaw_rate = 0.0

    def on_frame(self, data: bytes, now: float) -> None:
        if not p.crc_ok(data):
            self.stats["bad_crc"] += 1
            self.bad_crc_run += 1
            if self.bad_crc_run >= self.limits.crc_fault_after and not self.faults["crc"]:
                self.faults["crc"] = 1
                self._stop(now, p.STATE_FAULT, f"{self.bad_crc_run} corrupted frames in a row")
            return
        self.bad_crc_run = 0
        m = p.decode_drive_cmd(data)

        if m["counter"] == self.last_counter:
            self.stats["repeated"] += 1
            self.repeat_run += 1
            if self.repeat_run >= self.limits.counter_fault_after and not self.faults["counter"]:
                self.faults["counter"] = 1
                self._stop(now, p.STATE_FAULT, f"counter stuck at {m['counter']}")
            return  # an old frame is not fresh, whatever its content
        self.repeat_run = 0
        self.last_counter = m["counter"]

        if abs(m["speed_cmd"]) > self.limits.max_speed or abs(m["yaw_rate_cmd"]) > self.limits.max_yaw_rate:
            self.stats["range"] += 1
            self.faults["range"] = 1
            self._stop(now, p.STATE_FAULT, f"out of range {m['speed_cmd']:.2f} m/s {m['yaw_rate_cmd']:.2f} rad/s")
            return

        self.stats["accepted"] += 1
        self.last_valid_t = now

        if m["estop"]:
            self._stop(now, p.STATE_SAFE_STOP, "estop bit")
            return
        if not m["enable"] or m["mode"] == p.MODE_STOP:
            # Enable low: the computer acknowledges the stop. Faults stay
            # visible until enable has been low for clear_after_s, so nobody
            # upstream can miss them.
            self.armed = False
            self.speed = self.yaw_rate = 0.0
            if self.enable_low_since is None:
                self.enable_low_since = now
            if now - self.enable_low_since >= self.limits.clear_after_s or not any(self.faults.values()):
                if not self.saw_enable_low:
                    self._event(now, "disarmed_ready")
                self.saw_enable_low = True
                self.state = p.STATE_DISABLED
                self.faults = {k: 0 for k in self.faults}
            return
        self.enable_low_since = None
        if not self.armed:
            if not self.saw_enable_low:
                return  # enable must be seen low first; a stuck-high enable never arms
            self.armed = True
            self.faults = {k: 0 for k in self.faults}
            self.state = p.STATE_DRIVING
            self._event(now, "armed")
        self.speed, self.yaw_rate = m["speed_cmd"], m["yaw_rate_cmd"]

    def tick(self, now: float) -> tuple[float, float]:
        """Call at the output rate. Returns the setpoint to apply to the motors."""
        if self.armed and (self.last_valid_t is None or now - self.last_valid_t > self.limits.timeout_s):
            self.faults["timeout"] = 1
            self._stop(now, p.STATE_SAFE_STOP, f"no valid command for {self.limits.timeout_s * 1000:.0f} ms")
        return (self.speed, self.yaw_rate) if self.armed else (0.0, 0.0)
