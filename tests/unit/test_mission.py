"""Unit tests for sentry_ai.simulation.mission.MissionController."""

from __future__ import annotations

from typing import Any

import pytest

from sentry_ai.config.schema import MissionConfig, PlannerConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.navigation.astar import AStarPlanner
from sentry_ai.simulation.mission import MissionController, MissionPhase

_TICK = 0.1

# 5x3 open city. Hospital at (0, 0), vehicle beside it, one reachable victim
# at (4, 0) and one sealed inside a building block at (2, 2).
#   row0: .....
#   row1: ..#..
#   row2: .###.
_MAP_DATA: dict[str, Any] = {
    "width": 5,
    "height": 3,
    "grid": [".....", "..#..", ".###."],
    "safe_zone": {"position": [0, 0], "radius": 1, "capacity": 4},
    "vehicle_start": [1, 0],
    "victims": [
        {"id": "reachable", "position": [4, 0]},
        {"id": "sealed", "position": [2, 2]},
    ],
    "fires": [],
}


def _controller(city_map: CityMap, **overrides: Any) -> MissionController:
    return MissionController(
        city_map=city_map,
        grid=OccupancyGrid.from_city_map(city_map),
        planner=AStarPlanner(PlannerConfig(fire_risk_penalty=0.0, turn_penalty=0.0)),
        config=MissionConfig(**overrides),
    )


@pytest.fixture
def city_map() -> CityMap:
    return CityMap.from_config(_MAP_DATA)


class TestPlanning:
    def test_starts_in_planning_with_no_waypoint(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        assert mission.phase is MissionPhase.PLANNING
        assert mission.next_waypoint() is None

    def test_first_update_routes_toward_a_victim(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        mission.update(city_map.vehicle, _TICK)
        assert mission.phase is MissionPhase.EN_ROUTE_TO_VICTIM
        assert mission.route.goal == Position(4, 0)
        assert mission.stats.replans == 1

    def test_unreachable_victim_is_abandoned_not_retried(self, city_map: CityMap) -> None:
        city_map.victims[0].status = VictimStatus.RESCUED  # leave only the sealed one
        mission = _controller(city_map)
        mission.update(city_map.vehicle, _TICK)
        assert mission.stats.victims_unreachable == 1
        assert mission.phase is not MissionPhase.EN_ROUTE_TO_VICTIM

    def test_invalidate_route_forces_a_replan(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        mission.update(city_map.vehicle, _TICK)
        mission.invalidate_route()
        assert mission.next_waypoint() is None
        mission.update(city_map.vehicle, _TICK)
        assert mission.stats.replans == 2


class TestArrivals:
    def test_picks_up_a_victim_it_is_standing_on(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        city_map.vehicle.position = Position(4, 0)
        mission.update(city_map.vehicle, _TICK)
        assert city_map.victims[0].status is VictimStatus.ONBOARD
        assert city_map.vehicle.onboard_victims == [city_map.victims[0]]

    def test_does_not_overfill_the_vehicle(self, city_map: CityMap) -> None:
        city_map.vehicle.capacity = 1
        city_map.vehicle.onboard_victims.append(city_map.victims[1])
        mission = _controller(city_map)
        city_map.vehicle.position = Position(4, 0)
        mission.update(city_map.vehicle, _TICK)
        assert city_map.victims[0].status is VictimStatus.TRAPPED

    def test_delivers_onboard_victims_at_the_hospital(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        victim = city_map.victims[0]
        victim.status = VictimStatus.ONBOARD
        city_map.vehicle.onboard_victims.append(victim)
        city_map.vehicle.position = Position(0, 0)
        mission.update(city_map.vehicle, _TICK)
        assert victim.status is VictimStatus.RESCUED
        assert city_map.vehicle.onboard_victims == []
        assert mission.stats.victims_rescued == 1


class TestTermination:
    def test_expired_timer_fails_the_mission(self, city_map: CityMap) -> None:
        mission = _controller(city_map, time_limit_seconds=_TICK)
        mission.update(city_map.vehicle, _TICK)
        assert mission.phase is MissionPhase.FAILED
        assert "timer" in mission.stats.failure_reason

    def test_flat_battery_fails_the_mission(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        city_map.vehicle.battery_percent = 0.0
        mission.update(city_map.vehicle, _TICK)
        assert mission.phase is MissionPhase.FAILED
        assert "battery" in mission.stats.failure_reason

    def test_destroyed_vehicle_fails_the_mission(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        city_map.vehicle.health_percent = 0.0
        mission.update(city_map.vehicle, _TICK)
        assert mission.phase is MissionPhase.FAILED
        assert "destroyed" in mission.stats.failure_reason

    def test_low_battery_sends_the_vehicle_home(self, city_map: CityMap) -> None:
        mission = _controller(city_map, min_battery_to_continue=50.0)
        city_map.vehicle.battery_percent = 40.0
        city_map.vehicle.position = Position(4, 0)
        mission.update(city_map.vehicle, _TICK)
        assert mission.phase is MissionPhase.RETURNING_TO_HOSPITAL
        assert mission.route.goal == city_map.safe_zone.position

    def test_completes_once_nothing_reachable_remains(self, city_map: CityMap) -> None:
        for victim in city_map.victims:
            victim.status = VictimStatus.RESCUED
        mission = _controller(city_map)
        mission.update(city_map.vehicle, _TICK)  # routes home from (1, 0)
        city_map.vehicle.position = city_map.safe_zone.position
        mission.update(city_map.vehicle, _TICK)
        mission.update(city_map.vehicle, _TICK)
        assert mission.phase is MissionPhase.COMPLETED

    def test_terminal_phase_ignores_further_updates(self, city_map: CityMap) -> None:
        mission = _controller(city_map, time_limit_seconds=_TICK)
        mission.update(city_map.vehicle, _TICK)
        ticks_at_failure = mission.stats.ticks
        mission.update(city_map.vehicle, _TICK)
        assert mission.stats.ticks == ticks_at_failure


class TestStats:
    def test_display_rows_cover_every_headline_metric(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        labels = [label for label, _ in mission.stats.as_display_rows()]
        assert labels == ["Time", "Rescued", "Unreachable", "Replans", "Collisions", "Tiles"]
