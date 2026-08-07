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
from sentry_ai.domain.entities import Position, Vehicle, Victim
from sentry_ai.domain.enums import VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.interfaces.navigation import IRoutePlanner, Route
from sentry_ai.interfaces.world import IOccupancyGridSource
from sentry_ai.simulation.events import EventKind, EventLog
from sentry_ai.simulation.grid_source import GroundTruthGridSource

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
    victims_lost: int = 0
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
            ("Lost", str(self.victims_lost)),
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
        grid_source: Where the belief map comes from each tick. Defaults to
            ground truth, which is the Phase 2 behaviour; Phase 3 injects
            the camera pipeline here instead. This class cannot tell the
            difference, and that is the entire point of the seam.
        events: Running record of what happened, for the HUD and demos.
    """

    city_map: CityMap
    grid: OccupancyGrid
    planner: IRoutePlanner
    config: MissionConfig
    grid_source: IOccupancyGridSource = field(default_factory=GroundTruthGridSource)
    stats: MissionStats = field(default_factory=MissionStats)
    events: EventLog = field(default_factory=EventLog)

    _phase: MissionPhase = field(default=MissionPhase.PLANNING, init=False)
    _route: Route = field(default_factory=Route.unreachable, init=False)
    _previous_route: Route = field(default_factory=Route.unreachable, init=False)
    _route_index: int = field(default=0, init=False)
    _abandoned: set[str] = field(default_factory=set, init=False)
    _mourned: set[str] = field(default_factory=set, init=False)

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

    def note_world_change(self, changed_tiles: frozenset[Position]) -> bool:
        """React to a hazard having changed the city under the current plan.

        Also re-admits every victim previously written off as unreachable.
        A corridor that fire had blocked reopens when that fire burns out,
        and a victim given up on then deserves another attempt — leaving
        them abandoned is exactly how someone gets left behind for a reason
        that stopped being true. Re-admitting costs one extra planner call
        per replan and cannot loop: a victim who is still unreachable is
        written off again in the same pass.

        Re-admission takes effect at the *next* replan, so a vehicle part
        way through a delivery finishes that run first rather than turning
        around mid-street.
        """
        self._abandoned.clear()
        self.stats.victims_unreachable = 0
        return self.invalidate_route_if_affected(changed_tiles)

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
        self._record_losses()
        self._advance_route(vehicle)
        self._handle_arrivals(vehicle)

        if self._check_failure(vehicle):
            return
        if self.next_waypoint() is None:
            self._replan(vehicle)

    def refresh_grid(self, vehicle: Vehicle) -> None:
        """Ask the grid source for a fresh belief map.

        Public because the engine also calls it the moment a hazard changes
        the city: the command center must not keep planning against a map
        that a collapse has already made wrong.

        Whether that map is ground truth or a detector's guess is decided by
        whoever composed this controller. Nothing below this line — routing,
        triage, the failure rules — knows or asks.
        """
        self.grid = self.grid_source.grid_for(self.city_map, vehicle.position)

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
        """Choose the next objective and commit to its route, or end the mission."""
        target = self._select_target(vehicle)
        if target is None:
            self._finish()
            return
        if target.route.is_empty:
            # Only the hospital can reach here — unreachable victims are
            # dropped during selection, which already tried every one.
            self._fail("hospital unreachable")
            return
        self._commit(target)

    def _commit(self, target: _Objective) -> None:
        """Adopt an objective's route as the plan being driven."""
        self._previous_route = self._route
        self._route = target.route
        self._route_index = 0
        self.stats.replans += 1
        self._phase = (
            MissionPhase.RETURNING_TO_HOSPITAL
            if target.is_hospital
            else MissionPhase.EN_ROUTE_TO_VICTIM
        )
        self.record(
            EventKind.ROUTE,
            f"routing to {target.label} — {len(target.route)} tiles, "
            f"cost {target.route.cost:.1f}",
        )

    def _select_target(self, vehicle: Vehicle) -> _Objective | None:
        """The next objective, or ``None`` when the mission is over.

        Heads home when the vehicle is full, when battery has fallen below
        the configured reserve, or when nobody is left to rescue.
        """
        if vehicle.battery_percent <= self.config.min_battery_to_continue:
            return None if self._is_home(vehicle) else self._hospital(vehicle)
        if len(vehicle.onboard_victims) >= vehicle.capacity:
            return self._hospital(vehicle)

        victim = self._most_urgent_victim(vehicle)
        if victim is not None:
            return victim
        if vehicle.onboard_victims or not self._is_home(vehicle):
            return self._hospital(vehicle)
        return None

    def _most_urgent_victim(self, vehicle: Vehicle) -> _Objective | None:
        """The victim worth going to next, or ``None`` if none can be reached.

        Ranked by *planned route cost* rather than straight-line distance —
        a victim five tiles away through a wall is not closer than one eight
        tiles down a road, and ranking by air distance sent the vehicle at
        the wrong one. Urgency discounts that cost, so someone losing health
        beside a fire is worth a detour. Victims found to be unreachable are
        written off here, which is why the caller never has to retry.
        """
        best: _Objective | None = None
        best_score = float("inf")
        for victim in self._rescuable():
            route = self.planner.plan(self.grid, vehicle.position, victim.position)
            if route.is_empty:
                self._abandon(victim)
                continue
            score = route.cost - self._urgency_bonus(victim)
            if score < best_score:
                best_score = score
                best = _Objective(victim.position, victim.victim_id, False, route)
        return best

    def _rescuable(self) -> list[Victim]:
        """Every victim the vehicle could still do something for."""
        return [
            victim
            for victim in self.city_map.victims
            if victim.status.is_rescuable and victim.victim_id not in self._abandoned
        ]

    def _urgency_bonus(self, victim: Victim) -> float:
        """Route cost a victim's condition is worth discounting.

        Zero for someone in perfect health, ``urgency_weight`` tiles for
        someone about to die.
        """
        return self.config.urgency_weight * (1.0 - victim.health / 100.0)

    def _hospital(self, vehicle: Vehicle) -> _Objective:
        """The hospital as an objective, with the route to it."""
        position = self.city_map.safe_zone.position
        return _Objective(
            position=position,
            label="the hospital",
            is_hospital=True,
            route=self.planner.plan(self.grid, vehicle.position, position),
        )

    def _abandon(self, victim: Victim) -> None:
        """Write a victim off as unreachable, at least until the world moves."""
        if victim.victim_id in self._abandoned:
            return
        self._abandoned.add(victim.victim_id)
        self.stats.victims_unreachable = len(self._abandoned)
        self.record(EventKind.FAILURE, f"no route to {victim.victim_id} — abandoned")

    def _settle_unreachable_count(self) -> None:
        """Fix the final unreachable count from the world, not the tally.

        During a mission the count tracks who is *currently* written off,
        and :meth:`note_world_change` resets it so a reopened corridor is
        not held against anyone. That makes it the wrong number to report at
        the end: the honest figure is simply who was still trapped when the
        mission stopped.
        """
        self.stats.victims_unreachable = sum(
            1 for victim in self.city_map.victims if victim.status.is_rescuable
        )

    def _record_losses(self) -> None:
        """Count victims who did not survive the wait, announcing each once."""
        lost = [
            victim
            for victim in self.city_map.victims
            if victim.status is VictimStatus.LOST
        ]
        for victim in lost:
            if victim.victim_id not in self._mourned:
                self._mourned.add(victim.victim_id)
                self.record(EventKind.FAILURE, f"lost {victim.victim_id} — arrived too late")
        self.stats.victims_lost = len(lost)

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
        self._settle_unreachable_count()
        self.record(
            EventKind.MISSION,
            f"mission complete — {self.stats.victims_rescued} rescued, "
            f"{self.stats.victims_lost} lost, "
            f"{self.stats.victims_unreachable} unreachable",
        )

    def _fail(self, reason: str) -> None:
        """Mark the mission failed and record why."""
        self._route = Route.unreachable()
        self._route_index = 0
        self._phase = MissionPhase.FAILED
        self._settle_unreachable_count()
        self.stats.failure_reason = reason
        self.record(EventKind.FAILURE, f"mission failed — {reason}")


@dataclass(frozen=True)
class _Objective:
    """Where the mission controller wants the vehicle, and how to get there.

    The route is carried with the objective because choosing between
    victims requires planning to each of them anyway — throwing those
    routes away only to re-plan the winner would be wasted work.
    """

    position: Position
    label: str
    is_hospital: bool
    route: Route
