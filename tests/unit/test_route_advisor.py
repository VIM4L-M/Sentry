"""Unit tests for safety-aware route comparison (sentry_ai.navigation.route_advisor)."""

from __future__ import annotations

from sentry_ai.config.schema import PlannerConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.interfaces.navigation import Route
from sentry_ai.navigation.astar import AStarPlanner
from sentry_ai.navigation.route_advisor import (
    RouteAdvisor,
    RouteOption,
    pick_safest,
    traffic_cost,
)


def _two_streets() -> OccupancyGrid:
    """Two parallel east-west streets (rows 1 and 5) joined at both ends."""
    grid = OccupancyGrid.empty(12, 7)
    grid.cells[:, :] = OccupancyCode.BUILDING
    grid.cells[1, :] = OccupancyCode.ROAD
    grid.cells[5, :] = OccupancyCode.ROAD
    grid.cells[1:6, 0] = OccupancyCode.ROAD
    grid.cells[1:6, 11] = OccupancyCode.ROAD
    return grid


def test_traffic_cost_weights_the_tile_and_its_neighbours() -> None:
    cost = traffic_cost([(3, 3)], weight=4.0)
    assert cost[(3, 3)] == 4.0
    assert cost[(4, 3)] == cost[(3, 2)] == 2.0
    assert (5, 5) not in cost


def test_the_planner_avoids_a_busy_street_when_told_about_it() -> None:
    grid = _two_streets()
    start, goal = Position(0, 1), Position(11, 1)
    plain = AStarPlanner(PlannerConfig()).plan(grid, start, goal)
    assert all(p.y == 1 for p in plain.waypoints)
    busy = traffic_cost([(x, 1) for x in range(2, 10)], weight=5.0)
    careful = AStarPlanner(PlannerConfig(), extra_cost=lambda: busy).plan(grid, start, goal)
    assert any(p.y == 5 for p in careful.waypoints)


def test_the_advisor_scores_three_routes() -> None:
    grid = _two_streets()
    advisor = RouteAdvisor(PlannerConfig(), metres_per_tile=20.0, cruise_kmh=36.0)
    occupied = [(x, 1) for x in range(2, 10)]
    options = advisor.compare(grid, Position(0, 1), Position(11, 1), occupied)
    assert [o.name for o in options] == ["A", "B", "C"]
    shortest, safest = options[0], options[1]
    assert shortest.road_users > safest.road_users
    assert shortest.traffic_level == "Heavy"
    assert shortest.length_m == 11 * 20.0
    assert shortest.eta_minutes == shortest.length_m / 10.0 / 60.0


def test_fire_beside_a_route_raises_its_risk() -> None:
    grid = _two_streets()
    grid.mark(Position(5, 2), OccupancyCode.FIRE)
    advisor = RouteAdvisor(PlannerConfig(), metres_per_tile=20.0, cruise_kmh=36.0)
    options = advisor.compare(grid, Position(0, 1), Position(11, 1), [])
    assert options[0].risk_level == "High"


def test_pick_safest_prefers_route_b() -> None:
    route = Route(waypoints=(Position(0, 0), Position(1, 0)), cost=1.0)
    a = RouteOption("A", "shortest", route, 20.0, 5, 1, 0.1)
    b = RouteOption("B", "safest", route, 40.0, 0, None, 0.2)
    assert pick_safest([a, b]) is b
    assert pick_safest([]) is None
