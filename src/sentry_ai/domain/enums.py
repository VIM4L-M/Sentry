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


class VictimStatus(Enum):
    """Lifecycle state of a :class:`~sentry_ai.domain.entities.Victim`."""

    TRAPPED = "trapped"
    ONBOARD = "onboard"
    RESCUED = "rescued"


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
