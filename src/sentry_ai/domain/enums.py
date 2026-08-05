"""Domain enumerations shared by entities, the map, and (in later phases)
the simulation, rendering, and perception layers.
"""

from __future__ import annotations

from enum import Enum


class TerrainType(Enum):
    """What occupies a single tile of the city grid.

    Membership determines both how the tile is drawn (``rendering/theme.py``)
    and whether the vehicle can enter it (:attr:`blocks_movement`).
    """

    OPEN_GROUND = "open_ground"
    ROAD = "road"
    BUILDING = "building"
    COLLAPSED_BUILDING = "collapsed_building"
    RUBBLE = "rubble"
    TREE = "tree"
    SAFE_ZONE = "safe_zone"
    BLOCKED_ROAD = "blocked_road"

    @property
    def blocks_movement(self) -> bool:
        """Whether a vehicle can enter a tile of this terrain type."""
        return self in _BLOCKING_TERRAIN


_BLOCKING_TERRAIN = frozenset(
    {
        TerrainType.BUILDING,
        TerrainType.COLLAPSED_BUILDING,
        TerrainType.RUBBLE,
        TerrainType.TREE,
        TerrainType.BLOCKED_ROAD,
    }
)


class Heading(Enum):
    """Which way the rescue vehicle currently faces.

    The vehicle is heading-aware because its local action space is
    egocentric (turn left / turn right / forward / reverse), not absolute —
    see ``interfaces/navigation.py``. Values are the ``(dx, dy)`` step taken
    when moving forward, in tile space with ``y`` growing downward.
    """

    NORTH = (0, -1)
    EAST = (1, 0)
    SOUTH = (0, 1)
    WEST = (-1, 0)

    @property
    def delta(self) -> tuple[int, int]:
        """The ``(dx, dy)`` offset of one forward step in this heading."""
        return self.value

    def turn_left(self) -> Heading:
        """The heading 90 degrees counter-clockwise from this one."""
        order = _HEADING_CLOCKWISE
        return order[(order.index(self) - 1) % len(order)]

    def turn_right(self) -> Heading:
        """The heading 90 degrees clockwise from this one."""
        order = _HEADING_CLOCKWISE
        return order[(order.index(self) + 1) % len(order)]

    def opposite(self) -> Heading:
        """The heading 180 degrees from this one."""
        return self.turn_right().turn_right()


_HEADING_CLOCKWISE: tuple[Heading, ...] = (
    Heading.NORTH,
    Heading.EAST,
    Heading.SOUTH,
    Heading.WEST,
)


class VictimStatus(Enum):
    """Lifecycle state of a :class:`~sentry_ai.domain.entities.Victim`.

    ``LOST`` is terminal and is the cost of arriving too late: a trapped
    victim's health drains while they wait, and faster beside a fire. It
    exists so that taking the nearest victim first is a *decision* with
    consequences rather than the only sensible rule.
    """

    TRAPPED = "trapped"
    ONBOARD = "onboard"
    RESCUED = "rescued"
    LOST = "lost"

    @property
    def is_rescuable(self) -> bool:
        """Whether the vehicle can still do anything for this victim."""
        return self is VictimStatus.TRAPPED


class EntityKind(Enum):
    """Coarse discriminator for heterogeneous entity collections.

    Used wherever code needs to tell entity types apart without an
    ``isinstance`` chain — e.g. the renderer picking a sprite/color, or a
    future detector labeling what it sees.
    """

    VEHICLE = "vehicle"
    VICTIM = "victim"
    FIRE = "fire"
    SMOKE = "smoke"
    OBSTACLE = "obstacle"
