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
from sentry_ai.domain.occupancy import OccupancyGrid


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


class IOccupancyGridSource(ABC):
    """Where the command center's belief map comes from.

    The one seam Phase 3 exists to move. Until Phase 3 there was a single
    producer — ``OccupancyGrid.from_city_map``, the simulator handing over
    its own truth — and
    :class:`~sentry_ai.simulation.mission.MissionController` called it
    directly. Behind this port it becomes a choice:

    * :class:`~sentry_ai.simulation.grid_source.GroundTruthGridSource` —
      the Phase 2 behaviour, and the reference a detector is scored against.
    * :class:`~sentry_ai.perception.grid_source.DetectedGridSource` — the
      camera pipeline, which is wrong in interesting ways.

    Swapping them must not change a line of ``navigation/`` or the mission
    state machine. That is the claim ADR 0002 makes and Phase 3.5 tests.
    """

    @abstractmethod
    def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
        """Produce the grid the planner should reason over right now.

        Args:
            city_map: The world. A ground-truth source reads it directly; a
                perception source is only allowed to look at it *through*
                cameras, which is the distinction the two implementations
                embody.
            vehicle_position: Where the vehicle is. Supplied rather than
                detected — the command center knows where its own vehicle
                is, and no detector class reports one.

        Returns:
            A fresh grid. Callers mutate what they are given, so returning a
            cached instance would let one tick corrupt the next.
        """
        raise NotImplementedError


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
