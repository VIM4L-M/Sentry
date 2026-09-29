"""Ports for the two-tier navigation stack: global routing and local control.

The two tiers are deliberately separated, and neither replaces the other:

* **Global** — :class:`IRoutePlanner`, implemented in ``navigation/astar.py``
  by a deterministic A* search over the command center's
  :class:`~sentry_ai.domain.occupancy.OccupancyGrid`. It answers "which way
  across the city", producing sparse waypoints. It is not learned: a
  classical shortest-path search is exact, explainable, and instant to
  replan, which is what a command center needs.
* **Local** — :class:`ILocalController`, implemented in Phase 6 by a Deep
  Q-Network (Unit V). It answers "what do I do in the next tick, given what
  my own camera just saw" — swerve around a newly-fallen obstacle, hold
  until a fire dies down, back out of a dead end. It never plans a
  city-scale route.

Both are contracts only; this module imports no model library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sentry_ai.common.exceptions import DomainValidationError
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading
from sentry_ai.domain.occupancy import OccupancyCode


class LocalAction(Enum):
    """The egocentric actions the local controller chooses between each tick.

    Egocentric (relative to the vehicle's :class:`Heading`) rather than
    absolute compass moves, so a learned policy generalizes across
    approach directions instead of memorizing one per heading.
    """

    MOVE_FORWARD = "move_forward"
    REVERSE = "reverse"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"


#: Stable ordering for encoding actions as network output indices. Appending
#: is safe; reordering invalidates every trained policy checkpoint.
LOCAL_ACTION_ORDER: tuple[LocalAction, ...] = (
    LocalAction.MOVE_FORWARD,
    LocalAction.REVERSE,
    LocalAction.TURN_LEFT,
    LocalAction.TURN_RIGHT,
    LocalAction.STOP,
)


@dataclass(frozen=True)
class Route:
    """An ordered sequence of waypoints from a start tile to a goal tile.

    Attributes:
        waypoints: Tile positions to visit in order. The first entry is the
            start tile and the last is the goal, so a route to an adjacent
            tile has length 2 and a route to the current tile has length 1.
        cost: Total traversal cost the planner accumulated, in tile units
            (including any risk penalties it applied).
    """

    waypoints: tuple[Position, ...]
    cost: float

    def __post_init__(self) -> None:
        if self.cost < 0.0:
            raise DomainValidationError(f"Route.cost must be non-negative, got {self.cost}")

    def __len__(self) -> int:
        return len(self.waypoints)

    def __iter__(self) -> Iterator[Position]:
        return iter(self.waypoints)

    @property
    def is_empty(self) -> bool:
        """Whether the planner failed to find any route at all."""
        return not self.waypoints

    @property
    def goal(self) -> Position | None:
        """The final waypoint, or ``None`` for an empty route."""
        return self.waypoints[-1] if self.waypoints else None

    @classmethod
    def unreachable(cls) -> Route:
        """The empty route a planner returns when no path exists."""
        return cls(waypoints=(), cost=0.0)


@dataclass(frozen=True)
class LocalObservation:
    """What the local controller sees when it decides a single tick.

    Deliberately a typed record rather than a bare tensor: the simulation
    builds it from real state, tests can construct one by hand, and only
    :meth:`as_array` commits to a numeric layout. Phase 6 feeds
    :meth:`as_array` to the DQN; nothing else depends on that layout.

    Attributes:
        position: The vehicle's current tile.
        heading: The direction the vehicle currently faces.
        battery_percent: Remaining battery, 0-100.
        next_waypoint: The waypoint currently being driven toward, or
            ``None`` when the route is finished.
        blocked_ahead: Whether the tile directly in front is impassable.
        fire_proximity: Normalized 0-1 closeness to the nearest known fire
            (1.0 = on top of it, 0.0 = none within sensor range).
        road_user_ahead: A car or pedestrian is on the tile directly in
            front, as the vehicle's own sensors see it (Phase 9 traffic).
            ``False`` wherever there is no traffic.
        road_user_ahead_far: One is two tiles ahead — the tile the vehicle
            would reach next after this one.
    """

    position: Position
    heading: Heading
    battery_percent: float
    next_waypoint: Position | None
    blocked_ahead: bool
    fire_proximity: float
    road_user_ahead: bool = False
    road_user_ahead_far: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.battery_percent <= 100.0:
            raise DomainValidationError(
                f"LocalObservation.battery_percent must be within 0.0-100.0, "
                f"got {self.battery_percent}"
            )
        if not 0.0 <= self.fire_proximity <= 1.0:
            raise DomainValidationError(
                f"LocalObservation.fire_proximity must be within 0.0-1.0, "
                f"got {self.fire_proximity}"
            )

    def as_array(self) -> NDArray[np.float32]:
        """Flatten to the fixed-length vector the DQN consumes.

        Layout (length 9): ``[x, y, heading_dx, heading_dy, battery_fraction,
        delta_x_to_waypoint, delta_y_to_waypoint, blocked_ahead,
        fire_proximity]``. Waypoint deltas are zero when there is no active
        waypoint.
        """
        heading_dx, heading_dy = self.heading.delta
        if self.next_waypoint is None:
            delta_x, delta_y = 0, 0
        else:
            delta_x = self.next_waypoint.x - self.position.x
            delta_y = self.next_waypoint.y - self.position.y
        return np.array(
            [
                self.position.x,
                self.position.y,
                heading_dx,
                heading_dy,
                self.battery_percent / 100.0,
                delta_x,
                delta_y,
                float(self.blocked_ahead),
                self.fire_proximity,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class LocalDecision:
    """A local controller's chosen action plus the Q-values behind it."""

    action: LocalAction
    q_values: dict[LocalAction, float]


class IRoutePlanner(ABC):
    """Plans a global route across the occupancy grid.

    Implemented by ``navigation.astar.AStarPlanner``. Called by the command
    center on mission start and again on every replan trigger.
    """

    @abstractmethod
    def plan(self, grid: OccupancyGridLike, start: Position, goal: Position) -> Route:
        """Return the best route from ``start`` to ``goal``.

        Returns :meth:`Route.unreachable` — never raises — when the goal is
        unreachable, because "no path exists right now" is an ordinary
        mission state the command center reacts to, not an error.
        """
        raise NotImplementedError


class ILocalController(ABC):
    """Chooses one egocentric action per tick from a local observation.

    Implemented in Phase 6 by a Stable-Baselines3 DQN (Unit V), and in
    Phase 2 by a deterministic waypoint-following stand-in so the
    simulation is drivable before any model is trained.
    """

    @abstractmethod
    def decide(self, observation: LocalObservation) -> LocalDecision:
        """Return the action to execute this tick, with its Q-value estimates."""
        raise NotImplementedError


class OccupancyGridLike(Protocol):
    """The slice of :class:`~sentry_ai.domain.occupancy.OccupancyGrid` a planner needs.

    A ``Protocol`` rather than an ABC so ``OccupancyGrid`` satisfies it
    structurally — ``interfaces`` must not force a base class onto a domain
    type. Kept minimal so a planner can be unit-tested against a hand-built
    stub, and so alternative grid representations stay substitutable.
    """

    @property
    def width(self) -> int:
        """Number of tile columns."""
        ...

    @property
    def height(self) -> int:
        """Number of tile rows."""
        ...

    def is_traversable(self, position: Position) -> bool:
        """Whether a route may pass through ``position``."""
        ...

    def code_at(self, position: Position) -> OccupancyCode:
        """The code stored at ``position``, used for risk-aware costing."""
        ...
