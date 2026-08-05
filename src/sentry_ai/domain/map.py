"""The ``CityMap`` aggregate: the disaster city's layout and its entities.

``CityMap`` is the single object the rendering layer, and (from Phase 2)
the simulation engine, query for "what does the world look like right
now". It owns validation of every cross-entity invariant — bounds,
walkability, unique ids — that no single entity can check on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sentry_ai.common.exceptions import DomainValidationError
from sentry_ai.domain.entities import FireSource, Obstacle, Position, SafeZone, Vehicle, Victim
from sentry_ai.domain.enums import TerrainType

#: Any entity that can occupy a tile and be queried by proximity.
MapEntity = Victim | FireSource | Obstacle | Vehicle

#: Default single-character terrain legend used when a map config doesn't
#: override it via a ``terrain_legend`` section. ``SAFE_ZONE`` is
#: deliberately absent: it carries extra metadata (radius, capacity) and is
#: always specified via the explicit ``safe_zone`` key instead of a grid glyph.
DEFAULT_TERRAIN_LEGEND: dict[str, TerrainType] = {
    ".": TerrainType.OPEN_GROUND,
    "#": TerrainType.BUILDING,
    "=": TerrainType.ROAD,
    "x": TerrainType.COLLAPSED_BUILDING,
    "r": TerrainType.RUBBLE,
    "t": TerrainType.TREE,
    "!": TerrainType.BLOCKED_ROAD,
}

#: Terrain types that also get a discrete, trackable Obstacle entity
#: generated for every tile of that type (see :meth:`CityMap.from_config`).
_OBSTACLE_TERRAIN_TYPES = frozenset(
    {
        TerrainType.COLLAPSED_BUILDING,
        TerrainType.RUBBLE,
        TerrainType.TREE,
        TerrainType.BLOCKED_ROAD,
    }
)


@dataclass
class CityMap:
    """The disaster city: its tile grid plus every entity placed on it."""

    width: int
    height: int
    terrain: dict[Position, TerrainType]
    vehicle: Vehicle
    safe_zone: SafeZone
    victims: list[Victim] = field(default_factory=list)
    fires: list[FireSource] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    default_terrain: TerrainType = TerrainType.OPEN_GROUND

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> CityMap:
        """Build a validated ``CityMap`` from a parsed map YAML mapping.

        The terrain grid is an ASCII map: ``grid`` is a list of ``height``
        strings, each ``width`` characters long, one character per tile.
        Characters are resolved through :data:`DEFAULT_TERRAIN_LEGEND`,
        optionally extended/overridden by a ``terrain_legend`` mapping of
        ``{symbol: terrain_type_name}``. See
        ``configs/maps/city_default.yaml`` for a full example.

        Raises:
            DomainValidationError: If a required key is missing, the grid
                doesn't match the declared dimensions, an unknown terrain
                symbol appears, or the resulting map violates a domain
                invariant (see :meth:`validate`).
        """
        try:
            width = int(data["width"])
            height = int(data["height"])
        except KeyError as exc:
            raise DomainValidationError(f"Map config missing required key: {exc}") from exc

        legend = dict(DEFAULT_TERRAIN_LEGEND)
        for symbol, type_name in data.get("terrain_legend", {}).items():
            try:
                legend[symbol] = TerrainType(type_name)
            except ValueError as exc:
                raise DomainValidationError(
                    f"terrain_legend symbol '{symbol}' names unknown terrain type '{type_name}'"
                ) from exc

        grid = data.get("grid", [])
        if len(grid) != height:
            raise DomainValidationError(
                f"Map grid has {len(grid)} row(s) but height is declared as {height}"
            )

        terrain: dict[Position, TerrainType] = {}
        obstacles: list[Obstacle] = []
        obstacle_sequence: dict[TerrainType, int] = {}

        for y, row in enumerate(grid):
            if len(row) != width:
                raise DomainValidationError(
                    f"Map grid row {y} has length {len(row)} but width is declared as {width}"
                )
            for x, symbol in enumerate(row):
                terrain_type = legend.get(symbol)
                if terrain_type is None:
                    raise DomainValidationError(f"Unknown terrain symbol '{symbol}' at ({x}, {y})")
                if terrain_type is TerrainType.OPEN_GROUND:
                    continue  # default terrain; storing it explicitly would be redundant
                position = Position(x=x, y=y)
                terrain[position] = terrain_type
                if terrain_type in _OBSTACLE_TERRAIN_TYPES:
                    obstacle_sequence[terrain_type] = obstacle_sequence.get(terrain_type, 0) + 1
                    obstacle_id = f"{terrain_type.value}_{obstacle_sequence[terrain_type]}"
                    obstacles.append(
                        Obstacle(obstacle_id=obstacle_id, position=position, kind=terrain_type)
                    )

        if "safe_zone" not in data:
            raise DomainValidationError("Map config missing required key: 'safe_zone'")
        safe_zone_data = data["safe_zone"]
        safe_zone_position = _position_from_coord(safe_zone_data["position"], context="safe_zone")
        safe_zone = SafeZone(
            position=safe_zone_position,
            radius=int(safe_zone_data.get("radius", 2)),
            capacity=int(safe_zone_data.get("capacity", 4)),
        )
        terrain[safe_zone_position] = TerrainType.SAFE_ZONE

        if "vehicle_start" not in data:
            raise DomainValidationError("Map config missing required key: 'vehicle_start'")
        vehicle_position = _position_from_coord(data["vehicle_start"], context="vehicle_start")
        vehicle = Vehicle(position=vehicle_position)

        victims = [
            Victim(
                victim_id=str(item["id"]),
                position=_position_from_coord(item["position"], context="victims"),
            )
            for item in data.get("victims", [])
        ]
        fires = [
            FireSource(
                fire_id=str(item["id"]),
                position=_position_from_coord(item["position"], context="fires"),
                intensity=float(item.get("intensity", 0.5)),
                radius=int(item.get("radius", 1)),
            )
            for item in data.get("fires", [])
        ]

        return cls(
            width=width,
            height=height,
            terrain=terrain,
            vehicle=vehicle,
            safe_zone=safe_zone,
            victims=victims,
            fires=fires,
            obstacles=obstacles,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def in_bounds(self, position: Position) -> bool:
        """Whether ``position`` lies within the map's width/height."""
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def tile_at(self, position: Position) -> TerrainType:
        """The terrain type at ``position``, falling back to the map's default."""
        return self.terrain.get(position, self.default_terrain)

    def is_walkable(self, position: Position) -> bool:
        """Whether the vehicle could occupy ``position`` right now."""
        return self.in_bounds(position) and not self.tile_at(position).blocks_movement

    def entities_near(self, position: Position, radius: float) -> list[MapEntity]:
        """All victims, fires, obstacles, and the vehicle within ``radius`` tiles."""
        candidates: list[MapEntity] = [*self.victims, *self.fires, *self.obstacles, self.vehicle]
        return [entity for entity in candidates if entity.position.distance_to(position) <= radius]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check every cross-entity invariant this map must satisfy.

        Raises:
            DomainValidationError: On the first violation found — bad
                dimensions, an out-of-bounds entity, a duplicate id, or the
                vehicle starting on non-walkable terrain.
        """
        if self.width <= 0 or self.height <= 0:
            raise DomainValidationError(
                f"CityMap dimensions must be positive, got {self.width}x{self.height}"
            )

        self._require_in_bounds(self.vehicle.position, "vehicle")
        self._require_in_bounds(self.safe_zone.position, "safe_zone")
        for position in self.terrain:
            self._require_in_bounds(position, "terrain tile")

        self._require_unique_ids(self.victims, "victim_id", "victim")
        self._require_unique_ids(self.fires, "fire_id", "fire")
        self._require_unique_ids(self.obstacles, "obstacle_id", "obstacle")

        if self.tile_at(self.vehicle.position).blocks_movement:
            raise DomainValidationError(
                f"Vehicle start {self.vehicle.position.as_tuple()} sits on "
                f"non-walkable terrain ({self.tile_at(self.vehicle.position).value})"
            )

    def _require_in_bounds(self, position: Position, label: str) -> None:
        if not self.in_bounds(position):
            raise DomainValidationError(
                f"{label} position {position.as_tuple()} is outside the "
                f"{self.width}x{self.height} map"
            )

    @staticmethod
    def _require_unique_ids(
        entities: list[Any], id_attr: str, label: str
    ) -> None:
        seen: set[str] = set()
        for entity in entities:
            entity_id = getattr(entity, id_attr)
            if entity_id in seen:
                raise DomainValidationError(f"Duplicate {label} id: '{entity_id}'")
            seen.add(entity_id)


def _position_from_coord(coord: Any, *, context: str) -> Position:
    """Parse a ``[x, y]`` (or ``(x, y)``) YAML sequence into a ``Position``."""
    try:
        x, y = coord
    except (TypeError, ValueError) as exc:
        raise DomainValidationError(
            f"Expected a 2-element [x, y] coordinate in '{context}', got {coord!r}"
        ) from exc
    return Position(x=int(x), y=int(y))
