"""Port for processes that change the disaster city on their own.

Everything else in the simulation moves because the vehicle acted. A
:class:`IWorldProcess` moves because *time passed* — fire spreads, a
weakened facade finally comes down. That distinction matters to the
command center: a change the vehicle did not cause is exactly the event
that invalidates a plan it made a moment ago.

Each process reports a :class:`WorldChange` rather than mutating silently,
so the engine can tell the mission controller which tiles to stop trusting
without knowing what kind of hazard produced them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sentry_ai.domain.entities import Position
from sentry_ai.domain.map import CityMap


@dataclass(frozen=True)
class WorldChange:
    """Which tiles a world process altered during one tick, and why.

    Attributes:
        changed_tiles: Every tile whose occupancy may now differ from what
            the command center last believed. Reported whether the tile
            became blocked or became clear — both invalidate a plan.
        description: Short human-readable summary for logs and the HUD.
    """

    changed_tiles: frozenset[Position] = field(default_factory=frozenset)
    description: str = ""

    @property
    def is_empty(self) -> bool:
        """Whether nothing actually changed this tick."""
        return not self.changed_tiles

    @classmethod
    def none(cls) -> WorldChange:
        """The change reported by a process that did nothing this tick."""
        return cls()

    def merged_with(self, other: WorldChange) -> WorldChange:
        """Combine two changes into one, concatenating their descriptions."""
        descriptions = [text for text in (self.description, other.description) if text]
        return WorldChange(
            changed_tiles=self.changed_tiles | other.changed_tiles,
            description="; ".join(descriptions),
        )


class IWorldProcess(ABC):
    """Something that evolves the city independently of the vehicle.

    Implementations live in :mod:`sentry_ai.simulation.hazards`. They are
    driven by the engine once per tick, before the vehicle observes, so the
    vehicle always reacts to the world as it is now rather than as it was.
    """

    @abstractmethod
    def advance(self, city_map: CityMap, delta_seconds: float) -> WorldChange:
        """Advance this process by ``delta_seconds`` of simulated time.

        Implementations must be deterministic given the same seed and the
        same call sequence — the engine's replayability guarantee depends
        on it. Return :meth:`WorldChange.none` when nothing happened.
        """
        raise NotImplementedError
