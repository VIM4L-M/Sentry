"""Builds the many varied missions offline training runs through.

Phase 5 recorded trajectories and Phase 6 trains a policy; both need "a
fresh mission for this seed, starting somewhere different, driven by this
controller". It lives here once rather than in each script.

**Why the start is randomised.** On the shipped map every mission starts
on one tile and drives to the same four victims, so only the hazards vary —
and they rarely change the route. Phase 5 measured the consequence: 99.4%
of held-out windows were copies of training windows. A random road-tile
start per seed makes each mission's route its own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from sentry_ai.common.exceptions import ConfigurationError
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import AppConfig, SimulationConfig, VehicleConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import TerrainType
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.navigation import ILocalController
from sentry_ai.interfaces.world import IOccupancyGridSource
from sentry_ai.simulation.factory import Mission, build_mission
from sentry_ai.simulation.grid_source import GroundTruthGridSource, LaggedGridSource

#: Builds a fresh mission for a seed, driven by the given controller (or the
#: default waypoint follower when ``None``).
DrivenMissionSource = Callable[[int, ILocalController | None], Mission]


class MissionFactory:
    """Fresh missions from one map config, varied by seed."""

    def __init__(
        self,
        map_data: dict[str, Any],
        simulation_config: SimulationConfig,
        vehicle_config: VehicleConfig,
        randomise_start: bool = True,
        belief_lag: int = 0,
    ) -> None:
        """Create a factory.

        Args:
            map_data: The parsed map YAML. Kept as data, not a ``CityMap``,
                because a mission mutates its city and each needs a new one.
            simulation_config: Mission rules, planner, hazards.
            vehicle_config: Vehicle physics.
            randomise_start: Start each mission on a road tile drawn from
                its seed. ``False`` keeps the map's own start tile.
            belief_lag: Refreshes the command center's map trails the world
                by. ``0`` is the fresh ground-truth map of Phases 2-6. Above
                that the map lags, and the vehicle's physics switches to
                the true world, so a hazard the map has not caught up with
                can still be hit — the Phase 7 scenario.

        Raises:
            ValueError: If ``belief_lag`` is negative.
        """
        if belief_lag < 0:
            raise ValueError(f"belief_lag must be non-negative, got {belief_lag}")
        self._map_data = map_data
        self._simulation_config = simulation_config
        self._vehicle_config = vehicle_config
        self._randomise_start = randomise_start
        self._belief_lag = belief_lag
        self._starts = start_positions(CityMap.from_config(map_data))

    @classmethod
    def from_app_config(
        cls,
        loader: ConfigLoader,
        app_config: AppConfig,
        randomise_start: bool = True,
        belief_lag: int = 0,
    ) -> MissionFactory:
        """A factory for the map and rules ``app_config`` points at.

        Raises:
            ConfigurationError: If the app config names no simulation or
                vehicle config — there is nothing to run a mission with.
        """
        if app_config.simulation_config_path is None or app_config.vehicle_config_path is None:
            raise ConfigurationError(
                "app config must set 'simulation_config' and 'vehicle_config' to run missions"
            )
        return cls(
            map_data=loader.load_yaml(app_config.map_config_path),
            simulation_config=loader.load_simulation_config(app_config.simulation_config_path),
            vehicle_config=loader.load_vehicle_config(app_config.vehicle_config_path),
            randomise_start=randomise_start,
            belief_lag=belief_lag,
        )

    def build(self, seed: int, controller: ILocalController | None = None) -> Mission:
        """A new, unstarted mission for ``seed``.

        The same seed always gives the same disaster and the same start, so
        two controllers can be compared on identical missions.
        """
        return build_mission(
            city_map=CityMap.from_config(self.map_data_for(seed)),
            simulation_config=self._simulation_config,
            vehicle_config=self._vehicle_config,
            controller=controller,
            hazard_seed=seed,
            **self._belief_sources(),
        )

    def _belief_sources(self) -> dict[str, IOccupancyGridSource]:
        """The grid and physics sources for the configured lag, if any."""
        if self._belief_lag == 0:
            return {}
        return {
            "grid_source": LaggedGridSource(GroundTruthGridSource(), self._belief_lag),
            "physics_source": GroundTruthGridSource(),
        }

    def map_data_for(self, seed: int) -> dict[str, Any]:
        """The map config for ``seed``, with its start tile applied.

        The map is rebuilt from config with the new start, so ``CityMap``'s
        own validation still checks it.
        """
        if not self._randomise_start:
            return self._map_data
        start = self._starts[int(np.random.default_rng(seed).integers(len(self._starts)))]
        return {**self._map_data, "vehicle_start": [start.x, start.y]}


def start_positions(city_map: CityMap) -> list[Position]:
    """Every tile a mission could start the vehicle on, in row order.

    Open road only: not a building, not the hospital, and not a tile a
    victim, fire or obstacle already holds. Row order, so a seeded choice
    from this list is reproducible.
    """
    taken = {victim.position for victim in city_map.victims}
    taken |= {fire.position for fire in city_map.fires}
    taken |= {obstacle.position for obstacle in city_map.obstacles}
    return [
        Position(x, y)
        for y in range(city_map.height)
        for x in range(city_map.width)
        if city_map.tile_at(Position(x, y)) is TerrainType.ROAD and Position(x, y) not in taken
    ]
