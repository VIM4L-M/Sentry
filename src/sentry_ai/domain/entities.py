"""Domain entities: the nouns of the disaster city.

Every entity validates its own invariants in ``__post_init__`` (non-empty
ids, values within legal ranges, ...). Invariants that depend on the wider
map — bounds, tile occupancy, uniqueness of ids — are the responsibility
of :class:`~sentry_ai.domain.map.CityMap`, not of the entities themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sentry_ai.common.exceptions import DomainValidationError
from sentry_ai.common.types import GridCoordinate
from sentry_ai.domain.enums import EntityKind, Heading, TerrainType, VictimStatus


@dataclass(frozen=True)
class Position:
    """An integer grid coordinate. Immutable and hashable (usable as a dict key)."""

    x: int
    y: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise DomainValidationError(
                f"Position coordinates must be non-negative, got ({self.x}, {self.y})"
            )

    def distance_to(self, other: Position) -> float:
        """Euclidean distance to another position, in tile units."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> GridCoordinate:
        """This position as a plain ``(x, y)`` tuple."""
        return (self.x, self.y)


@dataclass
class Victim:
    """A person trapped somewhere in the disaster city, awaiting rescue."""

    victim_id: str
    position: Position
    status: VictimStatus = VictimStatus.TRAPPED
    health: int = 100

    def __post_init__(self) -> None:
        if not self.victim_id.strip():
            raise DomainValidationError("Victim.victim_id must not be empty")
        if not 0 <= self.health <= 100:
            raise DomainValidationError(f"Victim.health must be within 0-100, got {self.health}")

    @property
    def entity_kind(self) -> EntityKind:
        return EntityKind.VICTIM


@dataclass
class FireSource:
    """An active fire with an intensity and area of effect."""

    fire_id: str
    position: Position
    intensity: float = 0.5
    radius: int = 1

    def __post_init__(self) -> None:
        if not self.fire_id.strip():
            raise DomainValidationError("FireSource.fire_id must not be empty")
        if not 0.0 <= self.intensity <= 1.0:
            raise DomainValidationError(
                f"FireSource.intensity must be within 0.0-1.0, got {self.intensity}"
            )
        if self.radius < 0:
            raise DomainValidationError(
                f"FireSource.radius must be non-negative, got {self.radius}"
            )

    @property
    def entity_kind(self) -> EntityKind:
        return EntityKind.FIRE


@dataclass
class Obstacle:
    """A discrete, detectable obstacle placed on top of the terrain grid.

    Distinct from a plain :class:`~sentry_ai.domain.enums.TerrainType` tile:
    an ``Obstacle`` has an identity (``obstacle_id``) so it can be tracked,
    referenced, and — from Phase 3 onward — used as a detection target.
    """

    obstacle_id: str
    position: Position
    kind: TerrainType

    def __post_init__(self) -> None:
        if not self.obstacle_id.strip():
            raise DomainValidationError("Obstacle.obstacle_id must not be empty")

    @property
    def blocks_movement(self) -> bool:
        return self.kind.blocks_movement

    @property
    def entity_kind(self) -> EntityKind:
        return EntityKind.OBSTACLE


@dataclass
class SafeZone:
    """The area victims must be delivered to for a rescue to count."""

    position: Position
    radius: int = 2
    capacity: int = 4

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise DomainValidationError(f"SafeZone.radius must be non-negative, got {self.radius}")
        if self.capacity < 1:
            raise DomainValidationError(
                f"SafeZone.capacity must be at least 1, got {self.capacity}"
            )

    def contains(self, position: Position) -> bool:
        """Whether ``position`` lies within this safe zone's radius."""
        return self.position.distance_to(position) <= self.radius


@dataclass
class Vehicle:
    """The rescue vehicle: position, resources, and onboard victims.

    Movement, battery drain, and health loss are simulation-engine
    concerns (Phase 2) — this entity only holds state and the invariants
    that must always hold regardless of who mutates it.
    """

    position: Position
    heading: Heading = Heading.NORTH
    battery_percent: float = 100.0
    health_percent: float = 100.0
    capacity: int = 2
    onboard_victims: list[Victim] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.battery_percent <= 100.0:
            raise DomainValidationError(
                f"Vehicle.battery_percent must be within 0.0-100.0, got {self.battery_percent}"
            )
        if not 0.0 <= self.health_percent <= 100.0:
            raise DomainValidationError(
                f"Vehicle.health_percent must be within 0.0-100.0, got {self.health_percent}"
            )
        if self.capacity < 1:
            raise DomainValidationError(f"Vehicle.capacity must be at least 1, got {self.capacity}")
        if len(self.onboard_victims) > self.capacity:
            raise DomainValidationError(
                f"Vehicle has {len(self.onboard_victims)} onboard victims but capacity "
                f"{self.capacity}"
            )

    def is_operational(self) -> bool:
        """Whether the vehicle has enough battery and health to keep running."""
        return self.battery_percent > 0.0 and self.health_percent > 0.0

    @property
    def entity_kind(self) -> EntityKind:
        return EntityKind.VEHICLE
