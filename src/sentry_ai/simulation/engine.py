"""The fixed-timestep simulation engine — the loop everything else hangs off.

One :meth:`SimulationEngine.tick` is one indivisible step of simulated
time: sense, decide, act, adjudicate. The order matters and is fixed, so a
mission replays identically given the same inputs — a precondition for
reinforcement learning in Phase 6 and for reproducible bug reports now.

The engine holds no intelligence of its own. It reads the local controller
through :class:`~sentry_ai.interfaces.navigation.ILocalController` and the
mission rules through :class:`~sentry_ai.simulation.mission.MissionController`,
both injected — swapping the Phase 2 waypoint follower for a trained DQN
changes a composition root, not this file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.schema import SimulationConfig, VehicleConfig
from sentry_ai.domain.entities import Position, Vehicle
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyCode
from sentry_ai.interfaces.navigation import ILocalController, LocalDecision, LocalObservation
from sentry_ai.interfaces.world import IWorldProcess, WorldChange
from sentry_ai.simulation.mission import MissionController, MissionPhase, MissionStats
from sentry_ai.simulation.vehicle_controller import MoveOutcome, VehicleController

logger = get_logger(__name__)


@dataclass(frozen=True)
class TickResult:
    """Everything that happened in one simulation step.

    Attributes:
        tick: 1-based index of the step just executed.
        decision: What the local controller chose, with its Q-values.
        outcome: What physically happened when that choice was applied.
        phase: The mission phase after the step was adjudicated.
        world_change: What hazards altered before the vehicle acted. Empty
            on the great majority of ticks — hazards run on their own,
            slower cadence.
    """

    tick: int
    decision: LocalDecision
    outcome: MoveOutcome
    phase: MissionPhase
    world_change: WorldChange = field(default_factory=WorldChange.none)


class SimulationEngine:
    """Advances the disaster city one fixed timestep at a time."""

    def __init__(
        self,
        city_map: CityMap,
        mission: MissionController,
        controller: ILocalController,
        simulation_config: SimulationConfig,
        vehicle_config: VehicleConfig,
        world_processes: Sequence[IWorldProcess] = (),
    ) -> None:
        """Wire the engine to a world, a mission, and a driver.

        Args:
            city_map: The world being simulated. Mutated as the mission runs.
            mission: Mission state machine and global planner owner.
            controller: Whatever is driving this tick — the Phase 2
                waypoint follower, a keyboard adapter, or a trained DQN.
            simulation_config: Tick rate and mission rules.
            vehicle_config: Physics applied to the vehicle.
            world_processes: Hazards that evolve the city on their own,
                driven in the order given. Defaults to none, which is a
                completely static city.
        """
        self._city_map = city_map
        self._mission = mission
        self._controller = controller
        self._simulation_config = simulation_config
        self._vehicle_config = vehicle_config
        self._vehicle_controller = VehicleController(vehicle_config)
        self._world_processes = tuple(world_processes)
        self._tick_index = 0

        city_map.vehicle.battery_percent = vehicle_config.battery.initial_percent
        city_map.vehicle.capacity = vehicle_config.capacity

    @property
    def is_done(self) -> bool:
        """Whether the mission has reached a terminal phase."""
        return self._mission.phase.is_terminal

    @property
    def stats(self) -> MissionStats:
        """Live mission telemetry."""
        return self._mission.stats

    @property
    def seconds_per_tick(self) -> float:
        """Simulated time one :meth:`tick` represents."""
        return self._simulation_config.seconds_per_tick

    @property
    def mission(self) -> MissionController:
        """The mission this engine is running, for renderers and dashboards."""
        return self._mission

    def tick(self) -> TickResult | None:
        """Run exactly one simulation step.

        Returns ``None`` once the mission is over, so a caller can drive
        the engine in a loop without tracking termination itself.
        """
        if self.is_done:
            return None

        self._tick_index += 1
        vehicle = self._city_map.vehicle
        world_change = self._advance_world(vehicle)
        decision = self._controller.decide(self._observe(vehicle))
        outcome = self._vehicle_controller.apply(vehicle, decision.action, self._mission.grid)

        if outcome.collided:
            self._mission.stats.collisions += 1
            if self._simulation_config.mission.replan_on_blocked_route:
                self._mission.invalidate_route()

        self._mission.update(vehicle, self._simulation_config.seconds_per_tick)
        return TickResult(
            tick=self._tick_index,
            decision=decision,
            outcome=outcome,
            phase=self._mission.phase,
            world_change=world_change,
        )

    def _advance_world(self, vehicle: Vehicle) -> WorldChange:
        """Let hazards evolve the city, then resync the command center's map.

        Runs before the vehicle observes, so it always reacts to the world
        as it is now. The belief map is rebuilt immediately rather than at
        the end of the tick: planning a route across a street that a
        collapse has already blocked is exactly the mistake this ordering
        prevents.
        """
        change = WorldChange.none()
        for process in self._world_processes:
            change = change.merged_with(
                process.advance(self._city_map, self._simulation_config.seconds_per_tick)
            )
        if change.is_empty:
            return change

        self._mission.stats.hazard_events += 1
        self._mission.refresh_grid(vehicle)
        if self._simulation_config.mission.replan_on_blocked_route:
            self._mission.invalidate_route_if_affected(change.changed_tiles)
        logger.info("Tick %d — %s", self._tick_index, change.description)
        return change

    def run(self, max_ticks: int) -> MissionStats:
        """Tick until the mission ends or ``max_ticks`` is reached.

        ``max_ticks`` is a hard stop that protects headless callers (tests,
        RL rollouts) from a controller that never makes progress. Reaching
        it is not a mission failure — the mission simply has not finished.
        """
        for _ in range(max_ticks):
            if self.tick() is None:
                break
        return self._mission.stats

    def _observe(self, vehicle: Vehicle) -> LocalObservation:
        """Build the local controller's view of this instant."""
        waypoint = self._mission.next_waypoint()
        dx, dy = vehicle.heading.delta
        ahead_x, ahead_y = vehicle.position.x + dx, vehicle.position.y + dy
        blocked = ahead_x < 0 or ahead_y < 0
        if not blocked:
            blocked = not self._mission.grid.is_traversable(Position(ahead_x, ahead_y))

        return LocalObservation(
            position=vehicle.position,
            heading=vehicle.heading,
            battery_percent=vehicle.battery_percent,
            next_waypoint=waypoint,
            blocked_ahead=blocked,
            fire_proximity=self._fire_proximity(vehicle.position),
        )

    def _fire_proximity(self, position: Position) -> float:
        """Closeness to the nearest fire the vehicle can sense, 0-1.

        ``1.0`` means standing in it, ``0.0`` means nothing within
        ``vehicle.sensor_range_tiles``.
        """
        sensor_range = self._vehicle_config.sensor_range_tiles
        fires = self._mission.grid.positions_with(OccupancyCode.FIRE)
        if not fires:
            return 0.0
        nearest = min(position.distance_to(fire) for fire in fires)
        if nearest >= sensor_range:
            return 0.0
        return 1.0 - nearest / sensor_range
