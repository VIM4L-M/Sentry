"""Unit tests for sentry_ai.navigation.astar.AStarPlanner."""

from __future__ import annotations

import pytest

from sentry_ai.config.schema import PlannerConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.navigation.astar import AStarPlanner

_NO_PENALTIES = PlannerConfig(fire_risk_penalty=0.0, turn_penalty=0.0)


def _grid(rows: list[str]) -> OccupancyGrid:
    """Build a grid from ASCII art: ``.`` road, ``#`` building, ``F`` fire."""
    symbols = {".": OccupancyCode.ROAD, "#": OccupancyCode.BUILDING, "F": OccupancyCode.FIRE}
    grid = OccupancyGrid.empty(width=len(rows[0]), height=len(rows))
    for y, row in enumerate(rows):
        for x, symbol in enumerate(row):
            grid.mark(Position(x, y), symbols[symbol])
    return grid


class TestBasicPathfinding:
    def test_straight_corridor_is_tile_by_tile(self) -> None:
        planner = AStarPlanner(_NO_PENALTIES)
        route = planner.plan(_grid(["....."]), Position(0, 0), Position(4, 0))
        assert [p.as_tuple() for p in route] == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        assert route.cost == pytest.approx(4.0)

    def test_route_to_self_is_a_single_waypoint(self) -> None:
        planner = AStarPlanner(_NO_PENALTIES)
        route = planner.plan(_grid(["..."]), Position(1, 0), Position(1, 0))
        assert len(route) == 1
        assert route.cost == 0.0

    def test_detours_around_a_wall(self) -> None:
        planner = AStarPlanner(_NO_PENALTIES)
        route = planner.plan(_grid([".#.", ".#.", "..."]), Position(0, 0), Position(2, 0))
        coords = [p.as_tuple() for p in route]
        assert coords[0] == (0, 0)
        assert coords[-1] == (2, 0)
        assert (1, 0) not in coords and (1, 1) not in coords

    def test_every_waypoint_is_adjacent_to_the_last(self) -> None:
        planner = AStarPlanner(_NO_PENALTIES)
        route = planner.plan(_grid([".#.", ".#.", "..."]), Position(0, 0), Position(2, 0))
        steps = zip(route.waypoints, route.waypoints[1:], strict=False)
        assert all(abs(a.x - b.x) + abs(a.y - b.y) == 1 for a, b in steps)


class TestUnreachable:
    def test_walled_off_goal_returns_an_empty_route(self) -> None:
        planner = AStarPlanner(_NO_PENALTIES)
        route = planner.plan(_grid(["..#..", "..#..", "..#.."]), Position(0, 0), Position(4, 0))
        assert route.is_empty
        assert route.goal is None

    def test_impassable_goal_returns_an_empty_route(self) -> None:
        planner = AStarPlanner(_NO_PENALTIES)
        assert planner.plan(_grid([".#."]), Position(0, 0), Position(1, 0)).is_empty


class TestEscapingAnImpassableStart:
    """A vehicle engulfed by spreading fire still has to be given a way out."""

    def test_impassable_start_still_yields_a_route_out(self) -> None:
        planner = AStarPlanner(_NO_PENALTIES)
        route = planner.plan(_grid(["F.."]), Position(0, 0), Position(2, 0))
        assert [p.as_tuple() for p in route] == [(0, 0), (1, 0), (2, 0)]

    def test_the_route_out_never_re_enters_danger(self) -> None:
        planner = AStarPlanner(_NO_PENALTIES)
        route = planner.plan(_grid(["FF.", "...", "..."]), Position(0, 0), Position(2, 0))
        assert not route.is_empty
        # Only the start may be a fire tile; the planner must route around
        # (1, 0) rather than driving straight through it.
        assert (1, 0) not in [p.as_tuple() for p in route]

    def test_a_vehicle_walled_in_completely_has_no_route(self) -> None:
        planner = AStarPlanner(_NO_PENALTIES)
        rows = ["#####", "#F#.#", "#####"]
        assert planner.plan(_grid(rows), Position(1, 1), Position(3, 1)).is_empty


class TestCostModel:
    def test_fire_penalty_pushes_the_route_away_from_heat(self) -> None:
        rows = [".....", ".....", "..F..", ".....", "....."]
        risky = AStarPlanner(PlannerConfig(fire_risk_penalty=20.0, fire_risk_radius=2))
        route = risky.plan(_grid(rows), Position(0, 2), Position(4, 2))
        assert not route.is_empty
        # The direct line runs through row 2; a heavy penalty must detour off it.
        assert any(p.y != 2 for p in route)

    def test_zero_penalty_takes_the_direct_line(self) -> None:
        rows = [".....", ".....", ".....", ".....", "....."]
        planner = AStarPlanner(_NO_PENALTIES)
        route = planner.plan(_grid(rows), Position(0, 2), Position(4, 2))
        assert all(p.y == 2 for p in route)

    def test_turn_penalty_is_charged_on_direction_changes(self) -> None:
        rows = ["...", "...", "..."]
        straight = AStarPlanner(_NO_PENALTIES).plan(_grid(rows), Position(0, 0), Position(2, 2))
        turning = AStarPlanner(PlannerConfig(fire_risk_penalty=0.0, turn_penalty=1.0)).plan(
            _grid(rows), Position(0, 0), Position(2, 2)
        )
        assert turning.cost > straight.cost

    def test_default_config_is_used_when_none_is_injected(self) -> None:
        route = AStarPlanner().plan(_grid(["..."]), Position(0, 0), Position(2, 0))
        assert len(route) == 3
