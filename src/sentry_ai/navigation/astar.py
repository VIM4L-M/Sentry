"""A* shortest-path search over the occupancy grid.

Implements :class:`~sentry_ai.interfaces.navigation.IRoutePlanner`. The
search is 4-connected (the vehicle drives on tile edges, never diagonally)
and costs more than raw distance: tiles near a known fire carry a
configurable risk penalty, and changing direction carries a small turn
penalty. Both come from :class:`~sentry_ai.config.schema.PlannerConfig` —
no cost constant is written into this module.

Because the turn penalty makes cost depend on *how* a tile was entered, the
search runs over ``(position, incoming heading)`` states rather than bare
positions. That is a 4x larger state space and still trivial at city scale.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Mapping
from itertools import count

import numpy as np

from sentry_ai.common.logging_config import get_logger
from sentry_ai.common.types import GridCoordinate
from sentry_ai.config.schema import PlannerConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading
from sentry_ai.domain.occupancy import OccupancyCode
from sentry_ai.interfaces.navigation import IRoutePlanner, OccupancyGridLike, Route

logger = get_logger(__name__)

#: Base cost of entering any traversable tile, before risk/turn penalties.
_STEP_COST = 1.0

#: Sentinel "no previous heading" index, so the first move is never charged
#: a turn penalty regardless of which way the vehicle happens to face.
_NO_HEADING = -1

_HEADINGS: tuple[Heading, ...] = (Heading.NORTH, Heading.EAST, Heading.SOUTH, Heading.WEST)

#: ``_HEADINGS``' step offsets, looked up once instead of per expanded node.
_DELTAS: tuple[tuple[int, int], ...] = tuple(heading.delta for heading in _HEADINGS)

#: Codes no route may enter; mirrors ``OccupancyCode.is_traversable``.
_IMPASSABLE = tuple(code for code in OccupancyCode if not code.is_traversable)

#: A search node: the tile, and the index into ``_HEADINGS`` of the move
#: that entered it (or ``_NO_HEADING`` at the start).
_State = tuple[GridCoordinate, int]


class AStarPlanner(IRoutePlanner):
    """Plans risk-aware shortest routes across an occupancy grid."""

    def __init__(
        self,
        config: PlannerConfig | None = None,
        extra_cost: Callable[[], Mapping[GridCoordinate, float]] | None = None,
    ) -> None:
        """Create a planner.

        Args:
            config: Cost model. Defaults to :class:`PlannerConfig`'s own
                defaults, which is what unit tests and the Phase 2
                stand-in controller use.
            extra_cost: Extra cost per tile, read at every plan — how the
                command center steers a route away from heavy traffic
                (``navigation/route_advisor.py``). ``None`` adds nothing.
        """
        self._config = config if config is not None else PlannerConfig()
        self._extra_cost = extra_cost

    def plan(self, grid: OccupancyGridLike, start: Position, goal: Position) -> Route:
        """Return the cheapest route from ``start`` to ``goal``.

        Returns :meth:`Route.unreachable` when the goal is impassable or no
        path exists — an ordinary mission state, not an error (see
        :class:`IRoutePlanner`).

        An impassable *start* is not an error either, and deliberately does
        not short-circuit: once hazards can change the world underneath a
        stationary vehicle, fire can engulf the tile it is sitting on, and
        that is precisely the moment it most needs a route out. The search
        still refuses to *enter* impassable tiles, so any route found leaves
        the danger and never re-enters it.
        """
        if not grid.is_traversable(goal):
            return Route.unreachable()
        if start == goal:
            return Route(waypoints=(start,), cost=0.0)

        risk = self._risk_field(grid)
        if self._extra_cost is not None:
            for tile, cost in self._extra_cost().items():
                risk[tile] = risk.get(tile, 0.0) + cost
        passable = _passable_lookup(grid)
        goal_coord = goal.as_tuple()
        start_state: _State = (start.as_tuple(), _NO_HEADING)

        tie_break = count()
        frontier: list[tuple[float, int, _State]] = [
            (self._heuristic(start.as_tuple(), goal_coord), next(tie_break), start_state)
        ]
        best_cost: dict[_State, float] = {start_state: 0.0}
        came_from: dict[_State, _State] = {}

        while frontier:
            _, _, state = heapq.heappop(frontier)
            coord, _ = state
            if coord == goal_coord:
                return self._reconstruct(came_from, state, best_cost[state])

            for neighbour, step_cost in self._neighbours(passable, state, risk):
                tentative = best_cost[state] + step_cost
                if tentative >= best_cost.get(neighbour, float("inf")):
                    continue
                best_cost[neighbour] = tentative
                came_from[neighbour] = state
                priority = tentative + self._heuristic(neighbour[0], goal_coord)
                heapq.heappush(frontier, (priority, next(tie_break), neighbour))

        logger.info("No route from %s to %s", start.as_tuple(), goal.as_tuple())
        return Route.unreachable()

    # ------------------------------------------------------------------
    # Search internals
    # ------------------------------------------------------------------

    def _neighbours(
        self,
        passable: Callable[[int, int], bool],
        state: _State,
        risk: dict[GridCoordinate, float],
    ) -> list[tuple[_State, float]]:
        """Every traversable state reachable in one step, with its entry cost."""
        (x, y), previous_heading = state
        results: list[tuple[_State, float]] = []
        for index, (dx, dy) in enumerate(_DELTAS):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or not passable(nx, ny):
                continue
            cost = _STEP_COST + risk.get((nx, ny), 0.0)
            if previous_heading != _NO_HEADING and index != previous_heading:
                cost += self._config.turn_penalty
            results.append((((nx, ny), index), cost))
        return results

    def _risk_field(self, grid: OccupancyGridLike) -> dict[GridCoordinate, float]:
        """Extra cost per tile from proximity to known fire.

        Penalty falls off linearly with distance so the planner prefers
        skirting a fire's edge over cutting close to its center, and an
        entirely fire-free grid yields an empty dict (zero overhead).
        """
        config = self._config
        if config.fire_risk_penalty <= 0.0:
            return {}

        radius = config.fire_risk_radius
        risk: dict[GridCoordinate, float] = {}
        for x, y in _fire_tiles(grid):
            self._accumulate_risk(risk, Position(x, y), radius, config.fire_risk_penalty)
        return risk

    @staticmethod
    def _accumulate_risk(
        risk: dict[GridCoordinate, float], fire: Position, radius: int, penalty: float
    ) -> None:
        """Add ``fire``'s distance-weighted penalty into the ``risk`` field."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = fire.x + dx, fire.y + dy
                if x < 0 or y < 0:
                    continue
                distance = Position(x, y).distance_to(fire)
                if distance > radius:
                    continue
                weight = 1.0 - distance / (radius + 1)
                risk[(x, y)] = max(risk.get((x, y), 0.0), penalty * weight)

    @staticmethod
    def _heuristic(coord: GridCoordinate, goal: GridCoordinate) -> float:
        """Manhattan distance — admissible for 4-connected unit-cost movement.

        Stays admissible once risk and turn penalties are added, because
        those only ever *increase* real cost above the estimate.
        """
        return float(abs(coord[0] - goal[0]) + abs(coord[1] - goal[1]))

    @staticmethod
    def _reconstruct(came_from: dict[_State, _State], goal: _State, cost: float) -> Route:
        """Walk ``came_from`` back from ``goal`` into a start-to-goal route."""
        coords: list[GridCoordinate] = []
        state: _State | None = goal
        while state is not None:
            coords.append(state[0])
            state = came_from.get(state)
        coords.reverse()
        return Route(waypoints=tuple(Position(x, y) for x, y in coords), cost=cost)


def _passable_lookup(grid: OccupancyGridLike) -> Callable[[int, int], bool]:
    """``(x, y) -> traversable``, precomputed once per search when the grid allows.

    On an :class:`~sentry_ai.domain.occupancy.OccupancyGrid` the whole
    traversability mask is taken from its ``cells`` array in one numpy call
    and read as nested lists; asking the grid tile by tile built a
    ``Position`` and an ``OccupancyCode`` per lookup and made a single plan on
    a 200x200 city take 0.4 s. Any other grid falls back to asking it.
    """
    cells = getattr(grid, "cells", None)
    if isinstance(cells, np.ndarray) and cells.ndim == 2:
        blocked = np.isin(cells, [int(code) for code in _IMPASSABLE])
        rows: list[list[bool]] = (~blocked).tolist()
        height, width = cells.shape
        return lambda x, y: x < width and y < height and rows[y][x]
    return lambda x, y: grid.is_traversable(Position(x, y))


def _fire_tiles(grid: OccupancyGridLike) -> list[GridCoordinate]:
    """Every tile the grid marks as fire, found with numpy when it can be."""
    cells = getattr(grid, "cells", None)
    if isinstance(cells, np.ndarray) and cells.ndim == 2:
        ys, xs = np.nonzero(cells == int(OccupancyCode.FIRE))
        return [(int(x), int(y)) for y, x in zip(ys, xs, strict=True)]
    return [
        (x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.code_at(Position(x, y)) is OccupancyCode.FIRE
    ]
