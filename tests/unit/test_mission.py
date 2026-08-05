"""Unit tests for sentry_ai.simulation.mission.MissionController."""

from __future__ import annotations

from typing import Any

import pytest

from sentry_ai.config.schema import MissionConfig, PlannerConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import TerrainType, VictimStatus
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


class TestReactingToWorldChanges:
    """What the command center does when a hazard moves the world under it."""

    def test_a_change_nowhere_near_the_route_is_ignored(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        mission.update(city_map.vehicle, _TICK)
        route_before = mission.route

        assert mission.invalidate_route_if_affected(frozenset({Position(0, 2)})) is False
        assert mission.route is route_before

    def test_a_change_on_the_route_ahead_discards_the_plan(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        mission.update(city_map.vehicle, _TICK)
        ahead = mission.route.waypoints[-1]

        assert mission.invalidate_route_if_affected(frozenset({ahead})) is True
        assert mission.next_waypoint() is None
        assert mission.stats.routes_cut_by_hazards == 1

    def test_a_change_behind_the_vehicle_costs_nothing(self, city_map: CityMap) -> None:
        """Debris landing on ground already covered must not force a replan."""
        mission = _controller(city_map)
        mission.update(city_map.vehicle, _TICK)  # plans the route
        mission.update(city_map.vehicle, _TICK)  # consumes the waypoint underfoot
        already_passed = mission.route.waypoints[0]

        # Step onto the next waypoint so the first is genuinely behind us.
        city_map.vehicle.position = mission.route.waypoints[1]
        mission.update(city_map.vehicle, _TICK)

        assert mission.invalidate_route_if_affected(frozenset({already_passed})) is False
        assert mission.stats.routes_cut_by_hazards == 0

    def test_an_empty_change_set_is_a_no_op(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        mission.update(city_map.vehicle, _TICK)
        assert mission.invalidate_route_if_affected(frozenset()) is False

    def test_nothing_to_invalidate_when_no_route_is_active(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        assert mission.invalidate_route_if_affected(frozenset({Position(1, 0)})) is False

    def test_refresh_grid_picks_up_terrain_that_changed(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        mission.update(city_map.vehicle, _TICK)
        assert mission.grid.is_traversable(Position(3, 0)) is True

        city_map.terrain[Position(3, 0)] = TerrainType.COLLAPSED_BUILDING
        mission.refresh_grid(city_map.vehicle)
        assert mission.grid.is_traversable(Position(3, 0)) is False

    def test_a_route_cut_is_replanned_on_the_next_update(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        mission.update(city_map.vehicle, _TICK)
        replans_before = mission.stats.replans

        mission.invalidate_route_if_affected(frozenset({mission.route.waypoints[-1]}))
        mission.update(city_map.vehicle, _TICK)
        assert mission.stats.replans == replans_before + 1
        assert mission.next_waypoint() is not None


class TestChoosingWhoToSaveFirst:
    """Ranking is by planned route cost, discounted by how urgent a victim is."""

    # 11x3 of open ground. The vehicle sits at (5, 1) between a victim two
    # tiles east and one four tiles west.
    _MAP: dict[str, Any] = {
        "width": 11,
        "height": 3,
        "grid": ["...........", "...........", "..........."],
        "safe_zone": {"position": [0, 0], "radius": 1, "capacity": 4},
        "vehicle_start": [5, 1],
        "victims": [
            {"id": "close_and_stable", "position": [7, 1]},
            {"id": "far_and_critical", "position": [1, 1]},
        ],
        "fires": [],
    }

    def _wounded(self, urgency: float) -> tuple[MissionController, CityMap]:
        city_map = CityMap.from_config(self._MAP)
        city_map.victims[1].health = 10
        return _controller(city_map, urgency_weight=urgency), city_map

    def test_without_urgency_it_takes_the_cheapest_victim_to_reach(self) -> None:
        mission, city_map = self._wounded(urgency=0.0)
        mission.update(city_map.vehicle, _TICK)
        assert mission.route.goal == Position(7, 1)

    def test_urgency_makes_it_cross_to_the_dying_one(self) -> None:
        """The whole point: a victim running out of time is worth a detour."""
        mission, city_map = self._wounded(urgency=14.0)
        mission.update(city_map.vehicle, _TICK)
        assert mission.route.goal == Position(1, 1)

    def test_urgency_changes_nothing_when_everyone_is_healthy(self) -> None:
        city_map = CityMap.from_config(self._MAP)
        mission = _controller(city_map, urgency_weight=14.0)
        mission.update(city_map.vehicle, _TICK)
        assert mission.route.goal == Position(7, 1)

    def test_ranking_uses_route_cost_not_straight_line_distance(self) -> None:
        """A victim behind a wall is not close, however near they look."""
        data: dict[str, Any] = {
            "width": 5,
            "height": 3,
            "grid": [".....", "####.", "....."],
            "safe_zone": {"position": [0, 0], "radius": 1, "capacity": 4},
            "vehicle_start": [0, 0],
            "victims": [
                {"id": "near_in_air", "position": [0, 2]},
                {"id": "near_by_road", "position": [3, 0]},
            ],
            "fires": [],
        }
        city_map = CityMap.from_config(data)
        mission = _controller(city_map, urgency_weight=0.0)
        mission.update(city_map.vehicle, _TICK)

        # (0, 2) is two tiles away in a straight line but ten by road.
        assert mission.route.goal == Position(3, 0)


class TestLostVictims:
    def test_a_lost_victim_is_not_a_rescue_target(self, city_map: CityMap) -> None:
        city_map.victims[0].status = VictimStatus.LOST
        mission = _controller(city_map)
        mission.update(city_map.vehicle, _TICK)
        assert mission.route.goal != Position(4, 0)

    def test_losses_are_counted_and_announced_once(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        city_map.victims[0].status = VictimStatus.LOST
        mission.update(city_map.vehicle, _TICK)
        mission.update(city_map.vehicle, _TICK)

        assert mission.stats.victims_lost == 1
        announcements = [
            event for event in mission.events if "arrived too late" in event.message
        ]
        assert len(announcements) == 1

    def test_a_mission_completes_even_if_everyone_is_lost(self, city_map: CityMap) -> None:
        for victim in city_map.victims:
            victim.status = VictimStatus.LOST
        mission = _controller(city_map)
        city_map.vehicle.position = city_map.safe_zone.position
        mission.update(city_map.vehicle, _TICK)
        assert mission.phase is MissionPhase.COMPLETED
        assert mission.stats.victims_lost == 2


class TestReadmittingAbandonedVictims:
    def test_a_world_change_gives_an_abandoned_victim_another_chance(
        self, city_map: CityMap
    ) -> None:
        """A corridor blocked by fire reopens when that fire burns out."""
        city_map.victims[0].status = VictimStatus.RESCUED  # leave only the sealed one
        city_map.vehicle.position = Position(4, 0)  # away from home, so it keeps running
        mission = _controller(city_map)
        mission.update(city_map.vehicle, _TICK)
        assert mission.stats.victims_unreachable == 1

        # Open the wall, then tell the controller the world moved.
        city_map.terrain.pop(Position(2, 1), None)
        mission.refresh_grid(city_map.vehicle)
        mission.note_world_change(frozenset({Position(2, 1)}))
        assert mission.stats.victims_unreachable == 0

        # Re-admission takes effect at the next replan — a vehicle part-way
        # through a delivery finishes that run first, which is correct.
        mission.invalidate_route()
        mission.update(city_map.vehicle, _TICK)
        assert mission.route.goal == Position(2, 2)

    def test_a_still_unreachable_victim_is_written_off_again(
        self, city_map: CityMap
    ) -> None:
        city_map.victims[0].status = VictimStatus.RESCUED
        city_map.vehicle.position = Position(4, 0)
        mission = _controller(city_map)
        mission.update(city_map.vehicle, _TICK)
        mission.note_world_change(frozenset({Position(0, 2)}))
        mission.invalidate_route()
        mission.update(city_map.vehicle, _TICK)
        assert mission.stats.victims_unreachable == 1


class TestStats:
    def test_display_rows_cover_every_headline_metric(self, city_map: CityMap) -> None:
        mission = _controller(city_map)
        labels = [label for label, _ in mission.stats.as_display_rows()]
        assert labels == [
            "Time",
            "Rescued",
            "Lost",
            "Unreachable",
            "Replans",
            "Collisions",
            "Tiles",
            "Hazards",
            "Routes cut",
        ]
