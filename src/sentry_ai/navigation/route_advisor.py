"""Compares candidate routes on safety, not just length (Phase 9).

An emergency vehicle in heavy traffic should not simply take the shortest
street: a congested one is slower and puts more people at risk, and one
beside a fire can close at any moment. The command center therefore plans
with a traffic cost (:func:`traffic_cost`, given to :class:`AStarPlanner` as
``extra_cost``), and :class:`RouteAdvisor` lays the alternatives side by side
for the operator:

* **A — shortest**: distance only, no risk or traffic cost;
* **B — safest**: fire risk *and* traffic, the one the command center drives;
* **C — clear of hazards**: fire risk only, weighted heavier.

Each is scored on length, road users along it, closeness to fire and an ETA
at the configured cruise speed. Road users are never on the command
center's map, so their tiles are passed in by whoever owns the traffic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from sentry_ai.common.types import GridCoordinate
from sentry_ai.config.schema import PlannerConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.occupancy import OccupancyCode
from sentry_ai.interfaces.navigation import OccupancyGridLike, Route
from sentry_ai.navigation.astar import AStarPlanner

#: Extra planning cost of a tile a road user stands on; half on its neighbours.
TRAFFIC_WEIGHT = 3.0

#: Road users per km of route: at most this is "Low", at most the next "Moderate".
_LOW_TRAFFIC, _MODERATE_TRAFFIC = 3.0, 8.0

#: Tiles from the nearest fire: at most this is "High" risk, at most the next "Moderate".
_HIGH_RISK, _MODERATE_RISK = 2, 4

_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def traffic_cost(
    occupied: Iterable[GridCoordinate], weight: float = TRAFFIC_WEIGHT
) -> dict[GridCoordinate, float]:
    """Planning cost from road users: ``weight`` on their tile, half beside it."""
    cost: dict[GridCoordinate, float] = {}
    for x, y in occupied:
        cost[(x, y)] = cost.get((x, y), 0.0) + weight
        for dx, dy in _NEIGHBOURS:
            tile = (x + dx, y + dy)
            cost[tile] = cost.get(tile, 0.0) + weight / 2
    return cost


@dataclass(frozen=True)
class RouteOption:
    """One candidate route and how it scores.

    Attributes:
        name: "A", "B" or "C".
        purpose: What it was planned for, for the display.
        route: The waypoints.
        length_m: Real length, when the map has a scale; else tiles.
        road_users: Road users on or beside the route right now.
        fire_distance: Tiles from the route to the nearest known fire
            (``None`` when there is no fire).
        eta_minutes: At the cruise speed.
    """

    name: str
    purpose: str
    route: Route
    length_m: float
    road_users: int
    fire_distance: int | None
    eta_minutes: float

    @property
    def traffic_level(self) -> str:
        """ "Low", "Moderate" or "Heavy", by road users per kilometre."""
        per_km = self.road_users / max(0.2, self.length_m / 1000.0)
        if per_km <= _LOW_TRAFFIC:
            return "Low"
        return "Moderate" if per_km <= _MODERATE_TRAFFIC else "Heavy"

    @property
    def risk_level(self) -> str:
        """ "Safe", "Moderate" or "High", by closeness to fire."""
        if self.fire_distance is None or self.fire_distance > _MODERATE_RISK:
            return "Safe"
        return "High" if self.fire_distance <= _HIGH_RISK else "Moderate"


class RouteAdvisor:
    """Plans and scores the three candidate routes to the current goal."""

    def __init__(self, config: PlannerConfig, metres_per_tile: float, cruise_kmh: float) -> None:
        """Create an advisor using the command center's own cost model."""
        self._config = config
        self._metres_per_tile = metres_per_tile
        self._cruise_kmh = cruise_kmh

    def compare(
        self,
        grid: OccupancyGridLike,
        start: Position,
        goal: Position,
        occupied: Iterable[GridCoordinate],
    ) -> list[RouteOption]:
        """The three options for ``start`` to ``goal``; unreachable ones are left out."""
        occupied = list(occupied)
        busy = traffic_cost(occupied)
        careful = replace(self._config, fire_risk_penalty=self._config.fire_risk_penalty * 3)
        planners = (
            ("A", "shortest", AStarPlanner(replace(self._config, fire_risk_penalty=0.0))),
            ("B", "safest", AStarPlanner(self._config, extra_cost=lambda: busy)),
            (
                "C",
                "clear of hazards",
                AStarPlanner(careful),
            ),
        )
        fires = _fire_tiles(grid)
        options = []
        for name, purpose, planner in planners:
            route = planner.plan(grid, start, goal)
            if not route.is_empty:
                options.append(self._score(name, purpose, route, occupied, fires))
        return options

    def _score(
        self,
        name: str,
        purpose: str,
        route: Route,
        occupied: list[GridCoordinate],
        fires: list[GridCoordinate],
    ) -> RouteOption:
        tiles = {p.as_tuple() for p in route.waypoints}
        near = tiles | {(x + dx, y + dy) for x, y in tiles for dx, dy in _NEIGHBOURS}
        users = sum(1 for tile in occupied if tile in near)
        fire_distance = min(
            (abs(x - fx) + abs(y - fy) for x, y in tiles for fx, fy in fires), default=None
        )
        length = max(0, len(route) - 1) * self._metres_per_tile
        eta = length / (self._cruise_kmh / 3.6) / 60.0
        return RouteOption(name, purpose, route, length, users, fire_distance, eta)


def pick_safest(options: list[RouteOption]) -> RouteOption | None:
    """The route to drive: B when it exists, else the lowest-risk, least busy one."""
    for option in options:
        if option.name == "B":
            return option
    order = {"Safe": 0, "Moderate": 1, "High": 2}
    return min(options, key=lambda o: (order[o.risk_level], o.road_users, o.length_m), default=None)


def _fire_tiles(grid: OccupancyGridLike) -> list[GridCoordinate]:
    cells = getattr(grid, "cells", None)
    if cells is not None:
        import numpy as np  # noqa: PLC0415 - numpy is already loaded wherever grids exist

        ys, xs = np.nonzero(cells == int(OccupancyCode.FIRE))
        return [(int(x), int(y)) for y, x in zip(ys, xs, strict=True)]
    return [
        (x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.code_at(Position(x, y)) is OccupancyCode.FIRE
    ]
