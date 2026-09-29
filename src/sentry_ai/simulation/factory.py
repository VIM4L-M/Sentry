"""Wires a mission's parts together, in one place.

Three callers now need an identical mission: the live window, the dataset
builder, and — from Phase 6 — the reinforcement-learning environment. Each
one assembling it by hand is how they drift apart, and a dataset generated
against subtly different physics from the ones the simulation actually runs
is worse than no dataset at all.

The factory takes already-loaded configs rather than a ``ConfigLoader``.
Reading files stays with the composition root; wiring lives here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from sentry_ai.config.schema import SimulationConfig, VehicleConfig
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.navigation import ILocalController
from sentry_ai.interfaces.world import IOccupancyGridSource
from sentry_ai.navigation.astar import AStarPlanner
from sentry_ai.simulation.engine import SimulationEngine
from sentry_ai.simulation.grid_source import GroundTruthGridSource
from sentry_ai.simulation.hazards import build_world_processes
from sentry_ai.simulation.mission import MissionController
from sentry_ai.simulation.traffic import TrafficAwarePhysics, TrafficProcess
from sentry_ai.simulation.waypoint_follower import WaypointFollower


@dataclass(frozen=True)
class Mission:
    """A fully composed mission that has not been started.

    Attributes:
        city_map: The world, which the engine mutates as the mission runs.
        engine: The tick loop, already wired to a planner and a driver.
    """

    city_map: CityMap
    engine: SimulationEngine
    traffic: TrafficProcess | None = None

    @property
    def controller(self) -> MissionController:
        """The command center running this mission."""
        return self.engine.mission


#: Builds a fresh, unstarted mission for a hazard seed. Supplied by a
#: composition root, so the code that runs many missions — dataset
#: capture, trajectory recording — never reads a config file itself.
MissionSource = Callable[[int], Mission]


def build_mission(
    city_map: CityMap,
    simulation_config: SimulationConfig,
    vehicle_config: VehicleConfig,
    *,
    controller: ILocalController | None = None,
    hazard_seed: int | None = None,
    grid_source: IOccupancyGridSource | None = None,
    physics_source: IOccupancyGridSource | None = None,
) -> Mission:
    """Compose a mission over ``city_map``.

    Args:
        city_map: The world to run in. Mutated by the mission, so callers
            wanting several missions must build a fresh map for each.
        simulation_config: Tick rate, mission rules, planner costs, hazards.
        vehicle_config: Vehicle physics.
        controller: What drives the vehicle each tick. Defaults to the
            deterministic :class:`WaypointFollower` — the Phase 2 baseline,
            and the right choice for generating data, since a mission that
            drives itself sensibly visits more of the city than one that
            does not.
        hazard_seed: Overrides ``simulation_config.hazards.seed``. This is
            how the dataset builder gets a *different* disaster from the
            same starting map on every run, which is the only source of
            variety the training set has.
        grid_source: Where the command center's belief map comes from.
            Defaults to
            :class:`~sentry_ai.simulation.grid_source.GroundTruthGridSource`
            — perfect perception, the Phase 2 behaviour. Pass
            :class:`~sentry_ai.perception.grid_source.DetectedGridSource`
            to run the mission on what the cameras actually see.
        physics_source: What the vehicle collides with. Defaults to the
            belief map; see :class:`SimulationEngine`.

    Returns:
        An unstarted :class:`Mission`.
    """
    hazards = simulation_config.hazards
    if hazard_seed is not None:
        hazards = replace(hazards, seed=hazard_seed)

    source = grid_source if grid_source is not None else GroundTruthGridSource()
    processes = build_world_processes(hazards)
    traffic = None
    if simulation_config.traffic.enabled:
        # Road users: a world process, solid to the vehicle's physics, and
        # seen by its sensors — never on the command center's map.
        traffic = TrafficProcess(simulation_config.traffic)
        processes.append(traffic)
        physics_source = TrafficAwarePhysics(physics_source or GroundTruthGridSource(), traffic)
    mission = MissionController(
        city_map=city_map,
        # The mission refreshes this on its first tick; seeding it from the
        # same source keeps a freshly built mission consistent with one that
        # has already run, which the HUD renders before any tick happens.
        grid=source.grid_for(city_map, city_map.vehicle.position),
        planner=AStarPlanner(simulation_config.planner),
        config=simulation_config.mission,
        grid_source=source,
    )
    engine = SimulationEngine(
        city_map=city_map,
        mission=mission,
        controller=controller if controller is not None else WaypointFollower(),
        simulation_config=simulation_config,
        vehicle_config=vehicle_config,
        world_processes=processes,
        physics_source=physics_source,
        road_users=traffic.occupants if traffic is not None else None,
    )
    return Mission(city_map=city_map, engine=engine, traffic=traffic)
