"""Unit tests for sentry_ai.simulation.events."""

from __future__ import annotations

import pytest

from sentry_ai.simulation.events import EventKind, EventLog, MissionEvent


class TestMissionEvent:
    def test_a_row_pairs_a_timestamp_with_the_message(self) -> None:
        event = MissionEvent(at_seconds=12.34, kind=EventKind.RESCUE, message="picked up alice")
        assert event.as_row() == (" 12.3s", "picked up alice")

    def test_timestamps_are_padded_so_rows_line_up(self) -> None:
        early = MissionEvent(at_seconds=1.0, kind=EventKind.ROUTE, message="x").as_row()[0]
        late = MissionEvent(at_seconds=123.0, kind=EventKind.ROUTE, message="x").as_row()[0]
        assert len(early) == len(late)


class TestEventLog:
    def test_a_new_log_is_empty(self) -> None:
        assert len(EventLog()) == 0
        assert EventLog().recent(5) == []

    def test_recording_returns_the_event_it_stored(self) -> None:
        log = EventLog()
        event = log.record(1.5, EventKind.HAZARD, "collapse at (3, 4)")
        assert list(log) == [event]
        assert event.at_seconds == 1.5

    def test_recent_returns_the_newest_events_oldest_first(self) -> None:
        log = EventLog()
        for index in range(5):
            log.record(float(index), EventKind.ROUTE, f"event {index}")
        assert [event.message for event in log.recent(3)] == ["event 2", "event 3", "event 4"]

    def test_recent_copes_with_asking_for_more_than_exists(self) -> None:
        log = EventLog()
        log.record(0.0, EventKind.MISSION, "only one")
        assert len(log.recent(10)) == 1

    def test_recent_of_zero_is_empty_rather_than_everything(self) -> None:
        """A panel sized to zero rows must not accidentally render the lot."""
        log = EventLog()
        log.record(0.0, EventKind.MISSION, "x")
        assert log.recent(0) == []
        assert log.recent(-3) == []

    def test_the_log_is_bounded_and_drops_the_oldest(self) -> None:
        log = EventLog(capacity=3)
        for index in range(6):
            log.record(float(index), EventKind.ROUTE, f"event {index}")
        assert len(log) == 3
        assert [event.message for event in log] == ["event 3", "event 4", "event 5"]

    def test_capacity_is_reported(self) -> None:
        assert EventLog(capacity=42).capacity == 42

    @pytest.mark.parametrize("capacity", [0, -1])
    def test_a_non_positive_capacity_is_rejected(self, capacity: int) -> None:
        with pytest.raises(ValueError, match="capacity must be positive"):
            EventLog(capacity=capacity)
