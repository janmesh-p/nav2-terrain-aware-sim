"""Command arbitration and safe-stop policy, independent of ROS.

Decides every cycle who drives the vehicle: the autonomy stack, a remote
operator, or nobody (stop). Priority, highest first:

  1. E-stop. Latched; clearing it needs an explicit reset.
  2. Safety faults: stale odometry, excessive tilt.
  3. Teleop, when requested and its stream is fresh.
  4. Autonomy, when requested, fresh, and localization is trustworthy.
  5. Stop.

Judgment calls encoded here:
  * A lost teleop link stops the vehicle. Control never falls back to
    autonomy on its own; someone has to ask for it.
  * Fresh teleop input during autonomy is an operator takeover.
  * Stopping ramps down at a fixed deceleration, then holds zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class Mode(str, Enum):
    AUTONOMY = "autonomy"
    TELEOP = "teleop"
    STOP = "stop"


@dataclass
class Limits:
    autonomy_timeout_s: float = 0.5
    teleop_timeout_s: float = 0.3
    odom_timeout_s: float = 0.5
    max_tilt_deg: float = 15.0
    max_localization_std_m: float = 0.5
    max_linear: float = 1.0
    max_angular: float = 1.5
    decel_linear: float = 1.0    # m/s^2 while stopping
    decel_angular: float = 2.0   # rad/s^2 while stopping


@dataclass
class Decision:
    source: str
    reason: str
    linear: float
    angular: float


@dataclass
class _Stream:
    linear: float = 0.0
    angular: float = 0.0
    stamp: float | None = None

    def fresh(self, now: float, timeout: float) -> bool:
        return self.stamp is not None and 0.0 <= now - self.stamp <= timeout


@dataclass
class Arbiter:
    limits: Limits = field(default_factory=Limits)
    requested: Mode = Mode.STOP
    estop_latched: bool = False
    tilt_deg: float = 0.0
    localization_std_m: float | None = None
    vehicle_fault_reason: str | None = None
    odom_stamp: float | None = None
    _autonomy: _Stream = field(default_factory=_Stream)
    _teleop: _Stream = field(default_factory=_Stream)
    _out: tuple = (0.0, 0.0)
    _last_step: float | None = None
    takeover: bool = False

    # ----------------------------------------------------------- inputs
    def estop(self) -> None:
        self.estop_latched = True

    def reset(self) -> None:
        """Clear the e-stop. The vehicle stays stopped until a mode is requested."""
        self.estop_latched = False
        self.requested = Mode.STOP

    def request(self, mode: Mode) -> None:
        self.requested = Mode(mode)
        self.takeover = False

    def autonomy_cmd(self, linear: float, angular: float, now: float) -> None:
        self._autonomy = _Stream(linear, angular, now)

    def teleop_cmd(self, linear: float, angular: float, now: float) -> None:
        self._teleop = _Stream(linear, angular, now)
        if self.requested == Mode.AUTONOMY:
            self.requested = Mode.TELEOP  # operator takeover
            self.takeover = True

    def odometry(self, now: float) -> None:
        self.odom_stamp = now

    def vehicle_fault(self, reason: str | None) -> None:
        """Fault reported by the vehicle (ECU). Drops any granted mode, so
        driving again after it clears needs an explicit request."""
        if reason and not self.vehicle_fault_reason:
            self.requested = Mode.STOP
        self.vehicle_fault_reason = reason or None

    def tilt(self, deg: float) -> None:
        self.tilt_deg = deg

    def localization(self, std_m: float) -> None:
        self.localization_std_m = std_m

    # ----------------------------------------------------------- decision
    def _choose(self, now: float) -> tuple[str, str, float, float]:
        L = self.limits
        if self.estop_latched:
            return "stop", "e-stop latched", 0.0, 0.0
        if self.odom_stamp is None or not (0.0 <= now - self.odom_stamp <= L.odom_timeout_s):
            return "stop", "odometry stale", 0.0, 0.0
        if self.vehicle_fault_reason:
            return "stop", f"vehicle: {self.vehicle_fault_reason}", 0.0, 0.0
        if self.tilt_deg > L.max_tilt_deg:
            return "stop", f"tilt {self.tilt_deg:.1f} deg over {L.max_tilt_deg:.0f}", 0.0, 0.0

        if self.requested == Mode.TELEOP:
            if self._teleop.fresh(now, L.teleop_timeout_s):
                why = "operator takeover" if self.takeover else "teleop"
                return "teleop", why, self._teleop.linear, self._teleop.angular
            return "stop", "teleop link lost", 0.0, 0.0

        if self.requested == Mode.AUTONOMY:
            std = self.localization_std_m
            if std is None or std > L.max_localization_std_m:
                shown = "unknown" if std is None else f"{std:.2f} m"
                return "stop", f"localization uncertain ({shown})", 0.0, 0.0
            if not self._autonomy.fresh(now, L.autonomy_timeout_s):
                return "stop", "autonomy commands stale", 0.0, 0.0
            return "autonomy", "autonomy", self._autonomy.linear, self._autonomy.angular

        return "stop", "stop requested", 0.0, 0.0

    def step(self, now: float) -> Decision:
        L = self.limits
        dt = 0.0 if self._last_step is None else max(0.0, now - self._last_step)
        self._last_step = now
        source, reason, lin, ang = self._choose(now)

        if source == "stop":
            # Ramp toward zero; never reverse through it.
            lo, ao = self._out
            dl, da = L.decel_linear * dt, L.decel_angular * dt
            lin = math.copysign(max(0.0, abs(lo) - dl), lo)
            ang = math.copysign(max(0.0, abs(ao) - da), ao)
        else:
            lin = max(-L.max_linear, min(L.max_linear, lin))
            ang = max(-L.max_angular, min(L.max_angular, ang))

        self._out = (lin, ang)
        return Decision(source, reason, lin, ang)
