"""Other road users: the vehicles, people and animals sharing the streets (Phase 9).

Kept apart from the rescue entities in :mod:`sentry_ai.domain.entities`:
nothing about the mission depends on them, the command center's map never
contains them, and the perception models were never shown them. They exist
so the vehicle has to share the road — the unstructured road of an Indian
city, where cars, autorickshaws, two-wheelers, people and cattle all use the
same street (the traffic mix the India Driving Dataset, IIIT Hyderabad,
documents) — and so the emergency brake and the traffic-aware DQN have
something to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading


class AgentKind(Enum):
    """What sort of road user an agent is."""

    CAR = "car"
    AUTO_RICKSHAW = "auto"
    TWO_WHEELER = "two-wheeler"
    PEDESTRIAN = "pedestrian"
    COW = "cow"

    @property
    def is_vehicle(self) -> bool:
        """Drives the road tiles, going straight and turning at junctions."""
        return self in _VEHICLES

    @property
    def label(self) -> str:
        """What the display calls it."""
        return _LABELS[self]


_VEHICLES = frozenset({AgentKind.CAR, AgentKind.AUTO_RICKSHAW, AgentKind.TWO_WHEELER})

_LABELS = {
    AgentKind.CAR: "CAR",
    AgentKind.AUTO_RICKSHAW: "AUTO",
    AgentKind.TWO_WHEELER: "BIKE",
    AgentKind.PEDESTRIAN: "PERSON",
    AgentKind.COW: "COW",
}


@dataclass
class TrafficAgent:
    """One road user, moving one tile at a time.

    Attributes:
        agent_id: Unique id, for logs and tests.
        kind: Which sort of road user.
        position: The tile it occupies now.
        heading: The way it last moved (or faces, before its first move).
        previous: The tile it occupied before its last move, so a renderer
            can draw it gliding between the two.
        moved_at: The traffic clock tick of its last move.
        waiting: Consecutive move attempts it has been blocked for.
    """

    agent_id: str
    kind: AgentKind
    position: Position
    heading: Heading
    previous: Position
    moved_at: int = 0
    waiting: int = 0
