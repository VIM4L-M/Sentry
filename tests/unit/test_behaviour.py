"""Unit tests for sentry_ai.sequence.behaviour — what the four classes mean.

Everything in Phase 5 rests on this rule: the LSTM's training labels, the
persistence baseline, and therefore the M5 verdict. So its geometry is
pinned down case by case, in every heading, rather than left to be
inferred from a training curve.
"""

from __future__ import annotations

import math

import pytest

from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading
from sentry_ai.interfaces.sequence import BehaviourClass, VehicleState
from sentry_ai.sequence.behaviour import (
    BEHAVIOUR_ORDER,
    classify_motion,
    facing_vector,
    heading_to_degrees,
    persistence_baseline,
)


def _state(x: int, y: int, heading: Heading = Heading.EAST) -> VehicleState:
    return VehicleState(
        position=Position(x, y), battery_percent=80.0, heading_degrees=heading_to_degrees(heading)
    )


class TestBearings:
    @pytest.mark.parametrize("heading", list(Heading))
    def test_the_facing_vector_matches_the_headings_own_step(self, heading: Heading) -> None:
        """Bearing -> vector must agree with Heading.delta, or every label is rotated."""
        dx, dy = facing_vector(heading_to_degrees(heading))
        assert (round(dx), round(dy)) == heading.delta

    def test_bearings_run_clockwise_from_north(self) -> None:
        assert [heading_to_degrees(h) for h in (Heading.NORTH, Heading.EAST)] == [0.0, 90.0]
        assert heading_to_degrees(Heading.SOUTH) == 180.0
        assert heading_to_degrees(Heading.WEST) == 270.0


class TestClassifyMotion:
    def test_standing_still_is_hold(self) -> None:
        assert classify_motion(_state(3, 3), _state(3, 3, Heading.NORTH)) is BehaviourClass.HOLD

    @pytest.mark.parametrize("heading", list(Heading))
    def test_straight_ahead_is_advance_in_every_heading(self, heading: Heading) -> None:
        dx, dy = heading.delta
        start = _state(10, 10, heading)
        assert classify_motion(start, _state(10 + 3 * dx, 10 + 3 * dy)) is BehaviourClass.ADVANCE

    @pytest.mark.parametrize("heading", list(Heading))
    def test_straight_back_is_retreat_in_every_heading(self, heading: Heading) -> None:
        dx, dy = heading.delta
        start = _state(10, 10, heading)
        assert classify_motion(start, _state(10 - 2 * dx, 10 - 2 * dy)) is BehaviourClass.RETREAT

    @pytest.mark.parametrize("heading", list(Heading))
    def test_sideways_is_divert_in_every_heading(self, heading: Heading) -> None:
        dx, dy = heading.delta
        start = _state(10, 10, heading)
        # Perpendicular: rotate the heading's step by 90 degrees.
        assert classify_motion(start, _state(10 - 2 * dy, 10 + 2 * dx)) is BehaviourClass.DIVERT

    def test_mostly_forward_is_still_advance(self) -> None:
        """Three ahead, one across: 18 degrees off, inside the 30-degree cone."""
        assert classify_motion(_state(0, 5), _state(3, 4)) is BehaviourClass.ADVANCE

    def test_two_and_two_is_a_divert(self) -> None:
        """45 degrees off — a corner taken, not a street driven."""
        assert classify_motion(_state(0, 5), _state(2, 3)) is BehaviourClass.DIVERT

    def test_the_cone_is_a_parameter(self) -> None:
        start, end = _state(0, 5), _state(2, 3)
        assert classify_motion(start, end, cone_degrees=50.0) is BehaviourClass.ADVANCE

    def test_only_the_start_heading_matters(self) -> None:
        """Egocentric to where the vehicle *was* facing, not where it ended up facing."""
        start = _state(0, 0, Heading.EAST)
        assert classify_motion(start, _state(3, 0, Heading.WEST)) is BehaviourClass.ADVANCE

    @pytest.mark.parametrize("cone", [0.0, 90.0, -5.0, 120.0])
    def test_degenerate_cones_are_rejected(self, cone: float) -> None:
        with pytest.raises(ValueError):
            classify_motion(_state(0, 0), _state(1, 0), cone_degrees=cone)


class TestPersistenceBaseline:
    def test_it_predicts_the_recent_past_continues(self) -> None:
        history = [_state(x, 0) for x in range(5)]
        assert persistence_baseline(history, horizon=4) is BehaviourClass.ADVANCE

    def test_it_looks_back_exactly_horizon_steps(self) -> None:
        """Held for the last 2 ticks after moving: horizon 2 says HOLD, 3 says ADVANCE."""
        history = [_state(0, 0), _state(1, 0), _state(1, 0), _state(1, 0)]
        assert persistence_baseline(history, horizon=2) is BehaviourClass.HOLD
        assert persistence_baseline(history, horizon=3) is BehaviourClass.ADVANCE

    def test_too_short_a_history_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 5"):
            persistence_baseline([_state(0, 0)] * 4, horizon=4)

    def test_a_non_positive_horizon_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            persistence_baseline([_state(0, 0)] * 4, horizon=0)


class TestOrdering:
    def test_every_class_has_exactly_one_index(self) -> None:
        assert sorted(BEHAVIOUR_ORDER, key=lambda b: b.value) == sorted(
            BehaviourClass, key=lambda b: b.value
        )
        assert len(set(BEHAVIOUR_ORDER)) == len(BEHAVIOUR_ORDER)

    def test_facing_vectors_are_unit_length(self) -> None:
        for degrees in (0.0, 37.0, 90.0, 211.0):
            assert math.hypot(*facing_vector(degrees)) == pytest.approx(1.0)
