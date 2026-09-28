"""Unit tests for road users and the emergency brake (Phase 9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentry_ai.common.exceptions import ConfigValidationError
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import TrafficConfig
from sentry_ai.decision.emergency_brake import EmergencyBrake
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading, TerrainType
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.domain.traffic import AgentKind
from sentry_ai.interfaces.navigation import (
    ILocalController,
    LocalAction,
    LocalDecision,
    LocalObservation,
)
from sentry_ai.simulation.grid_source import GroundTruthGridSource
from sentry_ai.simulation.traffic import TrafficAwarePhysics, TrafficProcess

CONFIG = TrafficConfig(enabled=True, cars=6, pedestrians=6, seed=7)


@pytest.fixture
def city(project_root: Path) -> CityMap:
    loader = ConfigLoader(project_root=project_root)
    return CityMap.from_config(loader.load_yaml("configs/maps/city_default.yaml"))


def _run(city: CityMap, ticks: int, config: TrafficConfig = CONFIG) -> TrafficProcess:
    process = TrafficProcess(config)
    for _ in range(ticks):
        change = process.advance(city, 0.1)
        assert change.is_empty
    return process


class TestTrafficProcess:
    def test_disabled_spawns_nothing(self, city: CityMap) -> None:
        _run(city, 5, TrafficConfig())
        assert city.traffic == []

    def test_spawns_the_configured_road_users(self, city: CityMap) -> None:
        _run(city, 1)
        kinds = [agent.kind for agent in city.traffic]
        assert kinds.count(AgentKind.CAR) == 6
        assert kinds.count(AgentKind.PEDESTRIAN) == 6

    def test_cars_stay_on_roads_and_nobody_shares_a_tile(self, city: CityMap) -> None:
        process = TrafficProcess(CONFIG)
        for _ in range(60):
            process.advance(city, 0.1)
            tiles = [agent.position for agent in city.traffic]
            assert len(tiles) == len(set(tiles))
            assert city.vehicle.position not in tiles
            for agent in city.traffic:
                if agent.kind is AgentKind.CAR:
                    assert city.tile_at(agent.position) is TerrainType.ROAD

    def test_agents_move(self, city: CityMap) -> None:
        _run(city, 1)
        start = {agent.agent_id: agent.position for agent in city.traffic}
        process = TrafficProcess(CONFIG)
        city.traffic.clear()
        for _ in range(40):
            process.advance(city, 0.1)
        moved = [a for a in city.traffic if a.position != start.get(a.agent_id)]
        assert moved

    def test_same_seed_replays_identically(self, city: CityMap, project_root: Path) -> None:
        loader = ConfigLoader(project_root=project_root)
        other = CityMap.from_config(loader.load_yaml("configs/maps/city_default.yaml"))
        _run(city, 30)
        _run(other, 30)
        assert [a.position for a in city.traffic] == [a.position for a in other.traffic]

    def test_occupants_report_every_agent(self, city: CityMap) -> None:
        process = _run(city, 3)
        occupants = process.occupants()
        assert len(occupants) == len(city.traffic)
        for agent in city.traffic:
            assert occupants[agent.position.as_tuple()] is agent.kind


class TestTrafficAwarePhysics:
    def test_road_users_are_solid_and_the_inner_grid_is_untouched(self, city: CityMap) -> None:
        process = _run(city, 2)
        inner = GroundTruthGridSource()
        grid = TrafficAwarePhysics(inner, process).grid_for(city, city.vehicle.position)
        truth = inner.grid_for(city, city.vehicle.position)
        for agent in city.traffic:
            assert grid.code_at(agent.position) is OccupancyCode.DEBRIS
            assert isinstance(truth, OccupancyGrid)


class _Always(ILocalController):
    def __init__(self, action: LocalAction) -> None:
        self.action = action

    def decide(self, observation: LocalObservation) -> LocalDecision:
        return LocalDecision(self.action, {a: 0.0 for a in LocalAction})


def _observation() -> LocalObservation:
    return LocalObservation(
        position=Position(5, 5),
        heading=Heading.EAST,
        battery_percent=90.0,
        next_waypoint=Position(8, 5),
        blocked_ahead=False,
        fire_proximity=0.0,
    )


class TestEmergencyBrake:
    def test_stops_for_a_person_ahead(self) -> None:
        brake = EmergencyBrake(
            _Always(LocalAction.MOVE_FORWARD), lambda: {(6, 5): AgentKind.PEDESTRIAN}
        )
        assert brake.decide(_observation()).action is LocalAction.STOP
        assert brake.brakes[AgentKind.PEDESTRIAN] == 1
        assert brake.last is not None and brake.last.overruled is LocalAction.MOVE_FORWARD

    def test_stops_for_a_car_behind_when_reversing(self) -> None:
        brake = EmergencyBrake(_Always(LocalAction.REVERSE), lambda: {(4, 5): AgentKind.CAR})
        assert brake.decide(_observation()).action is LocalAction.STOP

    def test_passes_through_when_the_way_is_clear(self) -> None:
        brake = EmergencyBrake(_Always(LocalAction.MOVE_FORWARD), lambda: {(9, 9): AgentKind.CAR})
        assert brake.decide(_observation()).action is LocalAction.MOVE_FORWARD
        assert brake.last is None

    def test_turning_never_brakes(self) -> None:
        brake = EmergencyBrake(_Always(LocalAction.TURN_LEFT), lambda: {(6, 5): AgentKind.CAR})
        assert brake.decide(_observation()).action is LocalAction.TURN_LEFT

    def test_switched_off_it_drives_on_and_counts_the_hit(self) -> None:
        brake = EmergencyBrake(_Always(LocalAction.MOVE_FORWARD), lambda: {(6, 5): AgentKind.CAR})
        brake.enabled = False
        assert brake.decide(_observation()).action is LocalAction.MOVE_FORWARD
        assert brake.hits[AgentKind.CAR] == 1


class TestTrafficConfig:
    def test_rejects_bad_values(self) -> None:
        with pytest.raises(ConfigValidationError):
            TrafficConfig(cars=-1)
        with pytest.raises(ConfigValidationError):
            TrafficConfig(car_step_ticks=0)
        with pytest.raises(ConfigValidationError):
            TrafficConfig(crossing_chance=1.5)

    def test_city_config_enables_traffic(self, project_root: Path) -> None:
        loader = ConfigLoader(project_root=project_root)
        traffic = loader.load_simulation_config("configs/simulation_city.yaml").traffic
        assert traffic.enabled and traffic.cars > 0 and traffic.pedestrians > 0

    def test_default_config_has_no_traffic(self, project_root: Path) -> None:
        loader = ConfigLoader(project_root=project_root)
        assert not loader.load_simulation_config("configs/simulation.yaml").traffic.enabled
