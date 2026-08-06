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

from dataclasses import dataclass, replace

from sentry_ai.config.schema import SimulationConfig, VehicleConfig
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.interfaces.navigation import ILocalController
from sentry_ai.navigation.astar import AStarPlanner
from sentry_ai.simulation.engine import SimulationEngine
from sentry_ai.simulation.hazards import build_world_processes
from sentry_ai.simulation.mission import MissionController
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

    @property
    def controller(self) -> MissionController:
        """The command center running this mission."""
        return self.engine.mission


def build_mission(
    city_map: CityMap,
    simulation_config: SimulationConfig,
    vehicle_config: VehicleConfig,
    *,
    controller: ILocalController | None = None,
    hazard_seed: int | None = None,
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

    Returns:
        An unstarted :class:`Mission`.
    """
    hazards = simulation_config.hazards
    if hazard_seed is not None:
        hazards = replace(hazards, seed=hazard_seed)

    mission = MissionController(
        city_map=city_map,
        grid=OccupancyGrid.from_city_map(city_map),
        planner=AStarPlanner(simulation_config.planner),
        config=simulation_config.mission,
    )
    engine = SimulationEngine(
        city_map=city_map,
        mission=mission,
        controller=controller if controller is not None else WaypointFollower(),
        simulation_config=simulation_config,
        vehicle_config=vehicle_config,
        world_processes=build_world_processes(hazards),
    )
    return Mission(city_map=city_map, engine=engine)
