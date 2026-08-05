"""The command center's mission state machine.

Owns the questions the vehicle itself must not answer: which victim to go
after next, when to give up on one, when to head home, and whether the
mission has succeeded or failed. It drives the global planner
(:class:`~sentry_ai.interfaces.navigation.IRoutePlanner`) and hands the
vehicle one waypoint at a time.

It never touches vehicle physics — battery and damage belong to
:class:`~sentry_ai.simulation.vehicle_controller.VehicleController`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.schema import MissionConfig
from sentry_ai.domain.entities import Position, Vehicle
from sentry_ai.domain.enums import VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.interfaces.navigation import IRoutePlanner, Route
from sentry_ai.simulation.events import EventKind, EventLog

logger = get_logger(__name__)


class MissionPhase(Enum):
    """Where the mission currently stands."""

    PLANNING = "planning"
    EN_ROUTE_TO_VICTIM = "en_route_to_victim"
    RETURNING_TO_HOSPITAL = "returning_to_hospital"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Whether the mission has ended and no further tick will change it."""
        return self in (MissionPhase.COMPLETED, MissionPhase.FAILED)


@dataclass
class MissionStats:
    """Running mission telemetry, surfaced by the HUD and the dashboard."""

    ticks: int = 0
    elapsed_seconds: float = 0.0
    victims_rescued: int = 0
    victims_unreachable: int = 0
    replans: int = 0
    collisions: int = 0
    tiles_travelled: int = 0
    hazard_events: int = 0
    routes_cut_by_hazards: int = 0
    failure_reason: str = ""

    def as_display_rows(self) -> list[tuple[str, str]]:
        """Label/value pairs for direct rendering, in presentation order."""
        return [
            ("Time", f"{self.elapsed_seconds:6.1f}s"),
            ("Rescued", str(self.victims_rescued)),
            ("Unreachable", str(self.victims_unreachable)),
            ("Replans", str(self.replans)),
            ("Collisions", str(self.collisions)),
            ("Tiles", str(self.tiles_travelled)),
            ("Hazards", str(self.hazard_events)),
            ("Routes cut", str(self.routes_cut_by_hazards)),
        ]


@dataclass
class MissionController:
    """Plans, monitors, and adjudicates a single rescue mission.

    Attributes:
        city_map: The world, mutated as victims change status.
        grid: The command center's belief map, kept in sync each tick.
        planner: Global route planner, injected as a port.
        config: Success/failure/replanning rules.
        events: Running record of what happened, for the HUD and demos.
    """

    city_map: CityMap
    grid: OccupancyGrid
    planner: IRoutePlanner
    config: MissionConfig
    stats: MissionStats = field(default_factory=MissionStats)
    events: EventLog = field(default_factory=EventLog)

    _phase: MissionPhase = field(default=MissionPhase.PLANNING, init=False)
    _route: Route = field(default_factory=Route.unreachable, init=False)
    _previous_route: Route = field(default_factory=Route.unreachable, init=False)
    _route_index: int = field(default=0, init=False)
    _abandoned: set[str] = field(default_factory=set, init=False)

    @property
    def phase(self) -> MissionPhase:
        """The current mission phase."""
        return self._phase

    @property
    def route(self) -> Route:
        """The route currently being followed (empty when none is active)."""
        return self._route

    @property
    def previous_route(self) -> Route:
        """The route this one replaced, kept so a replan is visible.

        Drawn greyed out behind the active route: seeing the plan that was
        abandoned next to the plan that replaced it is what makes
        "the world changed and the command center reacted" legible at a
        glance rather than a number in a stats column.
        """
        return self._previous_route

    def record(self, kind: EventKind, message: str) -> None:
        """Log an event both to the log file and to the on-screen record."""
        logger.info("%s", message)
        self.events.record(self.stats.elapsed_seconds, kind, message)

    def next_waypoint(self) -> Position | None:
        """The tile the vehicle should drive toward now, if any."""
        if self._route_index >= len(self._route):
            return None
        return self._route.waypoints[self._route_index]

    def invalidate_route(self) -> None:
        """Discard the current route so the next tick replans from scratch.

        Called when the vehicle reports the world disagreed with the plan —
        it drove into something the belief map said was clear. This is the
        hook Phase 3's newly-detected obstacles will pull.
        """
        self._route = Route.unreachable()
        self._route_index = 0

    def invalidate_route_if_affected(self, changed_tiles: frozenset[Position]) -> bool:
        """Drop the route when a world change touches a tile it still relies on.

        Only the *unvisited* tail of the route matters: debris landing
        behind the vehicle costs nothing, debris landing ahead of it costs
        the whole plan. Returns whether the route was discarded, so the
        caller can count how often hazards actually cut a plan.
        """
        if self._route.is_empty or not changed_tiles:
            return False
        remaining = self._route.waypoints[self._route_index :]
        if not any(waypoint in changed_tiles for waypoint in remaining):
            return False

        self.invalidate_route()
        self.stats.routes_cut_by_hazards += 1
        self.record(EventKind.HAZARD, "world changed under the route — replanning")
        return True

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def update(self, vehicle: Vehicle, delta_seconds: float) -> None:
        """Advance the mission by one tick's worth of consequences.

        Called by the engine *after* the vehicle has moved, so it reacts to
        where the vehicle actually ended up.
        """
        if self._phase.is_terminal:
            return

        self.stats.ticks += 1
        self.stats.elapsed_seconds += delta_seconds
        self.refresh_grid(vehicle)
        self._advance_route(vehicle)
        self._handle_arrivals(vehicle)

        if self._check_failure(vehicle):
            return
        if self.next_waypoint() is None:
            self._replan(vehicle)

    def refresh_grid(self, vehicle: Vehicle) -> None:
        """Rebuild the belief map from the world, then stamp the vehicle on it.

        Public because the engine also calls it the moment a hazard changes
        the city: the command center must not keep planning against a map
        that a collapse has already made wrong.
        """
        self.grid = OccupancyGrid.from_city_map(self.city_map)
        self.grid.mark_vehicle(vehicle.position)

    def _advance_route(self, vehicle: Vehicle) -> None:
        """Consume waypoints the vehicle has already reached."""
        while self._route_index < len(self._route):
            if self._route.waypoints[self._route_index] != vehicle.position:
                break
            self._route_index += 1
            self.stats.tiles_travelled += 1

    def _handle_arrivals(self, vehicle: Vehicle) -> None:
        """Pick up any victim underfoot; unload everyone at the hospital."""
        for victim in self.city_map.victims:
            if (
                victim.status is VictimStatus.TRAPPED
                and victim.position == vehicle.position
                and len(vehicle.onboard_victims) < vehicle.capacity
            ):
                victim.status = VictimStatus.ONBOARD
                vehicle.onboard_victims.append(victim)
                self.record(
                    EventKind.RESCUE,
                    f"picked up {victim.victim_id} at {victim.position.as_tuple()}",
                )

        if vehicle.onboard_victims and self.city_map.safe_zone.contains(vehicle.position):
            for victim in vehicle.onboard_victims:
                victim.status = VictimStatus.RESCUED
                self.stats.victims_rescued += 1
                self.record(EventKind.RESCUE, f"delivered {victim.victim_id} to the hospital")
            vehicle.onboard_victims.clear()

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _replan(self, vehicle: Vehicle) -> None:
        """Choose the next objective and route to it, or end the mission.

        Recurses after abandoning an unreachable objective so the next-best
        one is tried in the same tick. Recursion is bounded: every
        abandonment permanently removes a candidate, and an unreachable
        hospital ends the mission outright.
        """
        target = self._select_target(vehicle)
        if target is None:
            self._finish()
            return

        route = self.planner.plan(self.grid, vehicle.position, target.position)
        if route.is_empty:
            self._abandon(target)
            if not self._phase.is_terminal:
                self._replan(vehicle)
            return

        self._previous_route = self._route
        self._route = route
        self._route_index = 0
        self.stats.replans += 1
        self._phase = (
            MissionPhase.RETURNING_TO_HOSPITAL
            if target.is_hospital
            else MissionPhase.EN_ROUTE_TO_VICTIM
        )
        self.record(
            EventKind.ROUTE,
            f"routing to {target.label} — {len(route)} tiles, cost {route.cost:.1f}",
        )

    def _select_target(self, vehicle: Vehicle) -> _Objective | None:
        """The next objective, or ``None`` when the mission is over.

        Heads home when the vehicle is full, when battery has fallen below
        the configured reserve, or when no rescuable victim is left.
        """
        hospital = _Objective(self.city_map.safe_zone.position, "the hospital", True)
        if vehicle.battery_percent <= self.config.min_battery_to_continue:
            return None if self._is_home(vehicle) else hospital
        if len(vehicle.onboard_victims) >= vehicle.capacity:
            return hospital

        candidates = [
            victim
            for victim in self.city_map.victims
            if victim.status is VictimStatus.TRAPPED and victim.victim_id not in self._abandoned
        ]
        if not candidates:
            if vehicle.onboard_victims or not self._is_home(vehicle):
                return hospital
            return None

        nearest = min(candidates, key=lambda v: v.position.distance_to(vehicle.position))
        return _Objective(nearest.position, nearest.victim_id, False)

    def _abandon(self, target: _Objective) -> None:
        """Record an objective as unreachable so it is never retried."""
        if target.is_hospital:
            self._fail("hospital unreachable")
            return
        self._abandoned.add(target.label)
        self.stats.victims_unreachable += 1
        self.record(EventKind.FAILURE, f"no route to {target.label} — abandoned")

    def _is_home(self, vehicle: Vehicle) -> bool:
        """Whether the vehicle is currently inside the safe zone."""
        return self.city_map.safe_zone.contains(vehicle.position)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_failure(self, vehicle: Vehicle) -> bool:
        """End the mission if time, battery, or health has run out."""
        if self.stats.elapsed_seconds >= self.config.time_limit_seconds:
            self._fail("mission timer expired")
            return True
        if not vehicle.is_operational():
            reason = "battery depleted" if vehicle.battery_percent <= 0.0 else "vehicle destroyed"
            self._fail(reason)
            return True
        return False

    def _finish(self) -> None:
        """Mark the mission complete — every reachable victim is home."""
        self._route = Route.unreachable()
        self._route_index = 0
        self._phase = MissionPhase.COMPLETED
        self.record(
            EventKind.MISSION,
            f"mission complete — {self.stats.victims_rescued} rescued, "
            f"{self.stats.victims_unreachable} unreachable",
        )

    def _fail(self, reason: str) -> None:
        """Mark the mission failed and record why."""
        self._route = Route.unreachable()
        self._route_index = 0
        self._phase = MissionPhase.FAILED
        self.stats.failure_reason = reason
        self.record(EventKind.FAILURE, f"mission failed — {reason}")


@dataclass(frozen=True)
class _Objective:
    """Where the mission controller currently wants the vehicle to be."""

    position: Position
    label: str
    is_hospital: bool
