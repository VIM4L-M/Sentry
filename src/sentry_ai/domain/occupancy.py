"""The occupancy grid: the command center's navigable view of the city.

An :class:`OccupancyGrid` is a coarse, integer-coded snapshot of what the
command center currently *believes* occupies every tile — the single input
the global route planner (A*) consumes. It is deliberately lossier than
:class:`~sentry_ai.domain.map.CityMap`: a planner needs "can I drive here
and how risky is it", not entity identities.

Two producers fill it, both writing the same codes:

* :meth:`OccupancyGrid.from_city_map` — ground truth, straight from the
  simulated world. Used until Phase 3 exists, and forever after as the
  reference the detector-driven grid is scored against.
* The Phase 3 CCTV pipeline — YOLO detections projected into world
  coordinates and merged, via :meth:`mark`.

Belief, not truth: nothing here assumes the grid is correct or complete.
Cells the cameras have never seen keep whatever the caller last wrote.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sentry_ai.common.exceptions import DomainValidationError
from sentry_ai.domain.entities import FireSource, Position, SafeZone, Vehicle, Victim
from sentry_ai.domain.enums import TerrainType, VictimStatus


class OccupancyCode(IntEnum):
    """What a single occupancy-grid cell holds.

    ``IntEnum`` rather than ``Enum`` on purpose: these values are written
    directly into a ``uint8`` array that later becomes a network input, so
    the numeric value *is* the contract.
    """

    ROAD = 0
    BUILDING = 1
    FIRE = 2
    DEBRIS = 3
    VICTIM = 4
    HOSPITAL = 5
    VEHICLE = 6

    @property
    def is_traversable(self) -> bool:
        """Whether the rescue vehicle may plan a route through this cell."""
        return self not in _IMPASSABLE_CODES


#: Codes a route may never pass through. ``VICTIM``/``HOSPITAL``/``VEHICLE``
#: are traversable: they are *destinations* or transient occupants, not walls.
_IMPASSABLE_CODES = frozenset({OccupancyCode.BUILDING, OccupancyCode.FIRE, OccupancyCode.DEBRIS})

#: How a ground-truth :class:`TerrainType` projects onto an occupancy code.
#: ``OPEN_GROUND`` and ``ROAD`` collapse to ``ROAD`` — the planner only cares
#: that both are free space. Every damaged/obstructing terrain type collapses
#: to ``DEBRIS``, matching what the Phase 3 detector is trained to emit.
TERRAIN_TO_OCCUPANCY: dict[TerrainType, OccupancyCode] = {
    TerrainType.OPEN_GROUND: OccupancyCode.ROAD,
    TerrainType.ROAD: OccupancyCode.ROAD,
    TerrainType.BUILDING: OccupancyCode.BUILDING,
    TerrainType.COLLAPSED_BUILDING: OccupancyCode.DEBRIS,
    TerrainType.RUBBLE: OccupancyCode.DEBRIS,
    TerrainType.TREE: OccupancyCode.DEBRIS,
    TerrainType.BLOCKED_ROAD: OccupancyCode.DEBRIS,
    TerrainType.SAFE_ZONE: OccupancyCode.HOSPITAL,
}


class CityMapLike(Protocol):
    """Exactly what :meth:`OccupancyGrid.from_city_map` reads off a city.

    :class:`~sentry_ai.domain.map.CityMap` satisfies this structurally, with
    no inheritance and no import — which keeps ``occupancy`` importable from
    ``map`` later without risking a cycle in either direction.
    """

    width: int
    height: int
    safe_zone: SafeZone
    vehicle: Vehicle

    # Read-only members: declared as properties so a concrete ``list``
    # attribute satisfies them. A plain ``Sequence`` annotation would be
    # invariant and reject ``CityMap``'s ``list`` fields.

    @property
    def fires(self) -> Sequence[FireSource]:
        """Every active fire source."""
        ...

    @property
    def victims(self) -> Sequence[Victim]:
        """Every victim, whatever their rescue status."""
        ...

    def tile_at(self, position: Position) -> TerrainType:
        """The terrain type at ``position``."""
        ...


@dataclass
class OccupancyGrid:
    """An integer-coded, mutable belief map of the city, indexed ``[y, x]``.

    Attributes:
        cells: ``uint8`` array of shape ``(height, width)`` holding
            :class:`OccupancyCode` values. Row-major (``cells[y, x]``) to
            match image/array convention, while every public method takes a
            :class:`Position` in ``(x, y)`` tile space.
    """

    cells: NDArray[np.uint8]

    def __post_init__(self) -> None:
        if self.cells.ndim != 2:
            raise DomainValidationError(
                f"OccupancyGrid.cells must be 2-dimensional, got shape {self.cells.shape}"
            )
        if self.cells.size == 0:
            raise DomainValidationError("OccupancyGrid.cells must not be empty")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls, width: int, height: int) -> OccupancyGrid:
        """An all-``ROAD`` (fully free, fully unknown-as-free) grid."""
        if width <= 0 or height <= 0:
            raise DomainValidationError(
                f"OccupancyGrid dimensions must be positive, got {width}x{height}"
            )
        return cls(cells=np.full((height, width), OccupancyCode.ROAD, dtype=np.uint8))

    @classmethod
    def from_city_map(cls, city_map: CityMapLike) -> OccupancyGrid:
        """Project a :class:`~sentry_ai.domain.map.CityMap` into ground truth.

        Layering order matters — later writes win: terrain, then fires,
        then victims, then the hospital, then the vehicle. A victim standing
        in debris must read as ``VICTIM`` so the planner can route *to* it.
        Only ``TRAPPED`` victims are stamped: once someone is aboard or
        delivered, their old tile reverts to plain terrain.

        Typed against a structural ``CityMapLike`` protocol rather than
        importing ``CityMap`` directly, so ``domain.map`` stays free to
        import this module without a cycle.
        """
        grid = cls.empty(city_map.width, city_map.height)
        for y in range(city_map.height):
            for x in range(city_map.width):
                terrain = city_map.tile_at(Position(x, y))
                grid.cells[y, x] = TERRAIN_TO_OCCUPANCY[terrain]

        for fire in city_map.fires:
            grid.mark_radius(fire.position, fire.radius, OccupancyCode.FIRE)
        for victim in city_map.victims:
            if victim.status is VictimStatus.TRAPPED:
                grid.mark(victim.position, OccupancyCode.VICTIM)
        grid.mark(city_map.safe_zone.position, OccupancyCode.HOSPITAL)
        grid.mark(city_map.vehicle.position, OccupancyCode.VEHICLE)
        return grid

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Number of tile columns."""
        return int(self.cells.shape[1])

    @property
    def height(self) -> int:
        """Number of tile rows."""
        return int(self.cells.shape[0])

    def in_bounds(self, position: Position) -> bool:
        """Whether ``position`` addresses a cell inside this grid."""
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def code_at(self, position: Position) -> OccupancyCode:
        """The code stored at ``position``.

        Raises:
            DomainValidationError: If ``position`` is outside the grid.
        """
        self._require_in_bounds(position)
        return OccupancyCode(int(self.cells[position.y, position.x]))

    def is_traversable(self, position: Position) -> bool:
        """Whether a route may pass through ``position``.

        Out-of-bounds positions are not traversable — callers exploring
        neighbours can ask without pre-checking bounds.
        """
        return self.in_bounds(position) and self.code_at(position).is_traversable

    def positions_with(self, code: OccupancyCode) -> list[Position]:
        """Every position currently holding ``code``, in row-major order."""
        ys, xs = np.nonzero(self.cells == code)
        return [Position(int(x), int(y)) for y, x in zip(ys, xs, strict=True)]

    def as_array(self) -> NDArray[np.uint8]:
        """A defensive copy of the raw ``[y, x]`` array, for model input."""
        return self.cells.copy()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mark(self, position: Position, code: OccupancyCode) -> None:
        """Write ``code`` at ``position``.

        Raises:
            DomainValidationError: If ``position`` is outside the grid.
        """
        self._require_in_bounds(position)
        self.cells[position.y, position.x] = code

    def mark_radius(self, center: Position, radius: int, code: OccupancyCode) -> None:
        """Write ``code`` across every in-bounds cell within ``radius`` of ``center``.

        Uses Euclidean distance, so a fire's footprint is a disc rather than
        a square. Out-of-bounds cells are skipped, not an error.
        """
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = center.x + dx, center.y + dy
                if x < 0 or y < 0:
                    continue
                position = Position(x, y)
                if self.in_bounds(position) and center.distance_to(position) <= radius:
                    self.cells[y, x] = code

    def _require_in_bounds(self, position: Position) -> None:
        if not self.in_bounds(position):
            raise DomainValidationError(
                f"Position {position.as_tuple()} is outside the "
                f"{self.width}x{self.height} occupancy grid"
            )


