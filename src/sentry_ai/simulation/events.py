"""A running record of what happened during a mission, and when.

The logger already writes everything to a file, but a file is the wrong
surface for two audiences this project has: an operator watching the window
and a reviewer watching a demo. Both need the last handful of events, in
order, on screen, phrased for a human rather than for grep.

The log is bounded. A long mission must not grow memory without limit, and
nothing on screen can show more than a dozen lines anyway.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum


class EventKind(Enum):
    """What sort of thing happened, so the HUD can colour it.

    Coarser than the set of things that *can* happen, on purpose: the point
    is to let a viewer tell good news from bad at a glance, not to
    reconstruct the mission from the log.
    """

    MISSION = "mission"
    ROUTE = "route"
    RESCUE = "rescue"
    HAZARD = "hazard"
    FAILURE = "failure"


@dataclass(frozen=True)
class MissionEvent:
    """One thing that happened, stamped with the simulated time it happened at.

    Attributes:
        at_seconds: Simulated mission time, not wall-clock — so a replay and
            a live run produce identical logs.
        kind: Category, used for colour.
        message: Short human-readable description, written for a screen.
    """

    at_seconds: float
    kind: EventKind
    message: str

    def as_row(self) -> tuple[str, str]:
        """Timestamp and message, formatted for direct rendering."""
        return (f"{self.at_seconds:5.1f}s", self.message)


class EventLog:
    """A bounded, append-only record of mission events."""

    def __init__(self, capacity: int = 200) -> None:
        """Create a log.

        Args:
            capacity: How many events to keep. Older ones are discarded
                once it is reached.

        Raises:
            ValueError: If ``capacity`` is not positive.
        """
        if capacity <= 0:
            raise ValueError(f"EventLog capacity must be positive, got {capacity}")
        self._events: deque[MissionEvent] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        """How many events this log retains."""
        return self._events.maxlen or 0

    def record(self, at_seconds: float, kind: EventKind, message: str) -> MissionEvent:
        """Append an event and return it."""
        event = MissionEvent(at_seconds=at_seconds, kind=kind, message=message)
        self._events.append(event)
        return event

    def recent(self, count: int) -> list[MissionEvent]:
        """The last ``count`` events, oldest first.

        Returns fewer than ``count`` when the log is shorter, and an empty
        list for a non-positive ``count`` — callers sizing a panel from
        available pixels should not have to special-case a tiny window.
        """
        if count <= 0:
            return []
        return list(self._events)[-count:]

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[MissionEvent]:
        return iter(self._events)
