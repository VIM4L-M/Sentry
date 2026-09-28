"""Other road users: the cars and pedestrians sharing the streets (Phase 9).

Kept apart from the rescue entities in :mod:`sentry_ai.domain.entities`:
nothing about the mission depends on them, the command center's map never
contains them, and the trained models were never shown them. They exist so
the vehicle has to share the road — and so the one component that does
watch for them, the emergency brake, has something to do.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading


class AgentKind(Enum):
    """What sort of road user an agent is."""

    CAR = "car"
    PEDESTRIAN = "pedestrian"


@dataclass
class TrafficAgent:
    """One car or pedestrian, moving one tile at a time.

    Attributes:
        agent_id: Unique id, for logs and tests.
        kind: Car or pedestrian.
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
