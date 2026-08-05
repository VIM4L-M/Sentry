"""Unit tests for sentry_ai.simulation.waypoint_follower.WaypointFollower."""

from __future__ import annotations

import pytest

from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading
from sentry_ai.interfaces.navigation import LOCAL_ACTION_ORDER, LocalAction, LocalObservation
from sentry_ai.simulation.waypoint_follower import WaypointFollower


def _observation(
    *,
    position: tuple[int, int] = (1, 1),
    heading: Heading = Heading.NORTH,
    waypoint: tuple[int, int] | None = (1, 0),
    blocked_ahead: bool = False,
    battery_percent: float = 100.0,
) -> LocalObservation:
    return LocalObservation(
        position=Position(*position),
        heading=heading,
        battery_percent=battery_percent,
        next_waypoint=Position(*waypoint) if waypoint is not None else None,
        blocked_ahead=blocked_ahead,
        fire_proximity=0.0,
    )


@pytest.fixture
def follower() -> WaypointFollower:
    return WaypointFollower()


class TestActionChoice:
    def test_drives_forward_when_already_facing_the_waypoint(
        self, follower: WaypointFollower
    ) -> None:
        decision = follower.decide(_observation(heading=Heading.NORTH, waypoint=(1, 0)))
        assert decision.action is LocalAction.MOVE_FORWARD

    def test_turns_right_toward_a_waypoint_to_the_east(self, follower: WaypointFollower) -> None:
        decision = follower.decide(_observation(heading=Heading.NORTH, waypoint=(2, 1)))
        assert decision.action is LocalAction.TURN_RIGHT

    def test_turns_left_toward_a_waypoint_to_the_west(self, follower: WaypointFollower) -> None:
        decision = follower.decide(_observation(heading=Heading.NORTH, waypoint=(0, 1)))
        assert decision.action is LocalAction.TURN_LEFT

    def test_a_180_degree_correction_resolves_as_a_right_turn(
        self, follower: WaypointFollower
    ) -> None:
        decision = follower.decide(_observation(heading=Heading.NORTH, waypoint=(1, 2)))
        assert decision.action is LocalAction.TURN_RIGHT

    def test_stops_when_there_is_no_waypoint(self, follower: WaypointFollower) -> None:
        assert follower.decide(_observation(waypoint=None)).action is LocalAction.STOP

    def test_stops_when_already_standing_on_the_waypoint(self, follower: WaypointFollower) -> None:
        decision = follower.decide(_observation(position=(1, 1), waypoint=(1, 1)))
        assert decision.action is LocalAction.STOP

    def test_stops_rather_than_ramming_a_blocked_tile(self, follower: WaypointFollower) -> None:
        decision = follower.decide(
            _observation(heading=Heading.NORTH, waypoint=(1, 0), blocked_ahead=True)
        )
        assert decision.action is LocalAction.STOP

    def test_turns_even_when_blocked_ahead(self, follower: WaypointFollower) -> None:
        """Being blocked must not prevent rotating away from the obstacle."""
        decision = follower.decide(
            _observation(heading=Heading.NORTH, waypoint=(2, 1), blocked_ahead=True)
        )
        assert decision.action is LocalAction.TURN_RIGHT


class TestDecisionShape:
    def test_q_values_cover_every_action_and_flag_the_chosen_one(
        self, follower: WaypointFollower
    ) -> None:
        decision = follower.decide(_observation())
        assert set(decision.q_values) == set(LOCAL_ACTION_ORDER)
        assert decision.q_values[decision.action] == 1.0
        assert sum(decision.q_values.values()) == 1.0
