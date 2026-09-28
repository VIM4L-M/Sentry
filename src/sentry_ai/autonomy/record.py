"""One mission, written down: outcome, timing, the vehicle's path, the event log.

PROJECT.md's Phase 8 asks for "mission statistics logged"; Phase 9's
dashboard is specified as "reading mission logs / replays". This is that
log. It is plain JSON on purpose — a record from the laptop GPU and one from
a CI run are the same kind of file, and either opens in a text editor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentry_ai.common.exceptions import AssetNotFoundError

#: Bumped when the record layout changes.
RECORD_FORMAT = 1


@dataclass(frozen=True)
class MissionRecord:
    """Everything worth keeping about one mission.

    Attributes:
        label: Who drove, e.g. ``"full stack"`` or ``"waypoint follower"``.
        seed: The hazard seed, when there was one.
        outcome: The final mission phase, e.g. ``"completed"``.
        failure_reason: Why it failed, or ``""``.
        stats: The mission's counters (``MissionStats`` as a dict).
        path: The vehicle's tile at the start and after every tick.
        timings: Per stage — calls, mean milliseconds, milliseconds per tick.
        events: The mission event log: time, kind, message.
        settings: The configuration it ran under, for reproducing it.
        written_at: When the record was made, UTC.
    """

    label: str
    seed: int | None
    outcome: str
    failure_reason: str
    stats: dict[str, Any]
    path: list[tuple[int, int]]
    timings: list[dict[str, Any]]
    events: list[dict[str, Any]]
    settings: dict[str, Any] = field(default_factory=dict)
    written_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    def save(self, directory: Path) -> Path:
        """Write the record as ``<time>_<label>_seed<n>.json`` in ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.written_at.replace(":", "").replace("-", "").replace("+0000", "Z")
        slug = self.label.replace(" ", "-")
        suffix = f"_seed{self.seed}" if self.seed is not None else ""
        path = directory / f"{stamp}_{slug}{suffix}.json"
        document = {"format": RECORD_FORMAT, **asdict(self)}
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> MissionRecord:
        """Read a record written by :meth:`save`.

        Raises:
            AssetNotFoundError: If the file does not exist.
            ValueError: If it is from an incompatible format.
        """
        if not path.is_file():
            raise AssetNotFoundError(f"Mission record not found: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.pop("format", None) != RECORD_FORMAT:
            raise ValueError(f"Unsupported mission record format in {path}")
        document["path"] = [tuple(tile) for tile in document["path"]]
        return cls(**document)

    @property
    def ticks(self) -> int:
        """Ticks the mission ran for."""
        return int(self.stats.get("ticks", 0))

    @property
    def mean_tick_ms(self) -> float:
        """Average wall-clock milliseconds per tick, when a tick total was timed."""
        for timing in self.timings:
            if timing["stage"] == "tick (total)":
                return float(timing["mean_ms"])
        return 0.0
