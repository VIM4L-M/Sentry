"""Smooth vehicle motion for the top-down view (Phase 9 display).

The simulation moves the vehicle one whole tile per tick and turns it 90
degrees at once. Drawn literally, it hops. :class:`VehicleGlide` keeps the
pose that is *drawn* separately from the simulated one and slides it along
each step at constant speed, finishing exactly when the next tick is due, so
the vehicle drives from tile to tile like a car in a game. It only changes
what is drawn; the simulation and every model still see whole tiles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sentry_ai.domain.entities import Vehicle

#: A jump longer than this, in tiles, is a restart or a teleport, not motion.
SNAP_TILES = 2.0


@dataclass(frozen=True)
class VehiclePose:
    """Where to draw the vehicle: tile coordinates of its centre, and its facing.

    ``angle`` is in radians clockwise from north, the convention the vehicle
    glyph is drawn in.
    """

    x: float
    y: float
    angle: float


def heading_angle(vehicle: Vehicle) -> float:
    """The vehicle's heading in radians clockwise from north."""
    dx, dy = vehicle.heading.delta
    return math.atan2(dx, -dy)


class VehicleGlide:
    """Slides the drawn vehicle from its last tile to its current one."""

    def __init__(self) -> None:
        """Start with no pose; the first update places the vehicle exactly."""
        self._start: VehiclePose | None = None
        self._end: VehiclePose | None = None
        self._elapsed = 0.0
        self._duration = 1.0

    @property
    def moving(self) -> bool:
        """Whether the drawn vehicle is part way along a move between two tiles."""
        if self._start is None or self._end is None:
            return False
        travelling = (self._start.x, self._start.y) != (self._end.x, self._end.y)
        return travelling and self._elapsed < self._duration

    def reset(self) -> None:
        """Forget the motion in progress, so the next update snaps."""
        self._start = None
        self._end = None

    def update(self, vehicle: Vehicle, delta_seconds: float, step_seconds: float) -> VehiclePose:
        """Advance by ``delta_seconds`` of frame time and return the pose to draw.

        Args:
            vehicle: The simulated vehicle; its tile and heading are the target.
            delta_seconds: Wall-clock time since the last update.
            step_seconds: Wall-clock time one simulation tick takes now, so a
                step finishes gliding just as the next one begins.
        """
        target = VehiclePose(
            vehicle.position.x + 0.5, vehicle.position.y + 0.5, heading_angle(vehicle)
        )
        if self._end is None or self._start is None:
            self._start = self._end = target
            return target
        if (target.x, target.y, target.angle) != (self._end.x, self._end.y, self._end.angle):
            current = self._pose()
            if math.hypot(target.x - current.x, target.y - current.y) > SNAP_TILES:
                self._start = self._end = target
                return target
            self._start, self._end = current, target
            self._elapsed = 0.0
            self._duration = max(step_seconds, 1e-3)
        else:
            self._elapsed += delta_seconds
        return self._pose()

    def _pose(self) -> VehiclePose:
        assert self._start is not None and self._end is not None
        t = min(1.0, self._elapsed / self._duration)
        eased = t * t * (3 - 2 * t) if self._start.angle != self._end.angle else t
        turn = (self._end.angle - self._start.angle + math.pi) % math.tau - math.pi
        return VehiclePose(
            self._start.x + (self._end.x - self._start.x) * t,
            self._start.y + (self._end.y - self._start.y) * t,
            self._start.angle + turn * eased,
        )
