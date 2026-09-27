"""Unit tests for sentry_ai.decision.dqn_controller.

The egocentric features are the whole interface between the simulation and
the policy, so their geometry is pinned down in every heading: a waypoint
straight ahead must read as "ahead" whichever way the vehicle faces, or the
policy learns a different city for each compass point. The adapter is
tested against a stand-in model — the translation from Q-values to a
decision is what matters, and it needs no trained weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.decision.dqn_controller import (
    POLICY_FEATURES,
    DqnLocalController,
    action_at,
    policy_features,
)
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading
from sentry_ai.interfaces.navigation import (
    LOCAL_ACTION_ORDER,
    ILocalController,
    LocalAction,
    LocalObservation,
)

_CENTRE = Position(10, 10)


def _observation(
    heading: Heading,
    waypoint: Position | None,
    position: Position = _CENTRE,
    **overrides: object,
) -> LocalObservation:
    values: dict[str, object] = {
        "position": position,
        "heading": heading,
        "battery_percent": 80.0,
        "next_waypoint": waypoint,
        "blocked_ahead": False,
        "fire_proximity": 0.0,
    }
    values.update(overrides)
    return LocalObservation(**values)  # type: ignore[arg-type]


def _offset(heading: Heading, forward: int, right: int) -> Position:
    """The tile ``forward`` ahead and ``right`` to the right of (10, 10)."""
    hx, hy = heading.delta
    rx, ry = -hy, hx
    return Position(10 + forward * hx + right * rx, 10 + forward * hy + right * ry)


class TestFeatures:
    def test_one_vector_of_the_declared_length(self) -> None:
        features = policy_features(_observation(Heading.NORTH, Position(10, 9)))
        assert features.shape == (POLICY_FEATURES,)
        assert features.dtype == np.float32

    @pytest.mark.parametrize("heading", list(Heading))
    def test_straight_ahead_reads_as_ahead_in_every_heading(self, heading: Heading) -> None:
        features = policy_features(_observation(heading, _offset(heading, 1, 0)))
        assert features[0] == pytest.approx(1 / 3)
        assert features[1] == pytest.approx(0.0)

    @pytest.mark.parametrize("heading", list(Heading))
    def test_to_the_right_reads_as_right_in_every_heading(self, heading: Heading) -> None:
        features = policy_features(_observation(heading, _offset(heading, 0, 2)))
        assert features[0] == pytest.approx(0.0)
        assert features[1] == pytest.approx(2 / 3)

    def test_behind_is_negative_forward(self) -> None:
        features = policy_features(_observation(Heading.EAST, _offset(Heading.EAST, -1, 0)))
        assert features[0] == pytest.approx(-1 / 3)

    def test_far_waypoints_are_clipped(self) -> None:
        features = policy_features(_observation(Heading.NORTH, _offset(Heading.NORTH, 9, -7)))
        assert features[0] == 1.0 and features[1] == -1.0

    def test_no_waypoint_is_flagged_and_zeroed(self) -> None:
        features = policy_features(_observation(Heading.NORTH, None))
        assert features[:3].tolist() == [0.0, 0.0, 0.0]

    def test_standing_on_the_waypoint_is_flagged(self) -> None:
        features = policy_features(_observation(Heading.NORTH, Position(10, 10)))
        assert features[2] == 1.0 and features[6] == 1.0

    def test_the_rest_are_passed_through(self) -> None:
        observation = _observation(
            Heading.WEST,
            Position(9, 10),
            blocked_ahead=True,
            fire_proximity=0.4,
            battery_percent=25.0,
        )
        features = policy_features(observation)
        assert features[3:6].tolist() == pytest.approx([1.0, 0.4, 0.25])

    def test_absolute_position_is_not_an_input(self) -> None:
        """Egocentric by design: the same situation elsewhere looks the same."""
        here = policy_features(_observation(Heading.SOUTH, Position(3, 5), Position(3, 4)))
        there = policy_features(_observation(Heading.SOUTH, Position(20, 15), Position(20, 14)))
        assert np.array_equal(here, there)


class TestActionIndex:
    def test_indices_follow_the_shared_order(self) -> None:
        assert [action_at(i) for i in range(len(LOCAL_ACTION_ORDER))] == list(LOCAL_ACTION_ORDER)

    @pytest.mark.parametrize("index", [-1, len(LOCAL_ACTION_ORDER)])
    def test_out_of_range_is_rejected(self, index: int) -> None:
        with pytest.raises(ValueError):
            action_at(index)


class _Values:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def squeeze(self, _dim: int) -> _Values:
        return self

    def cpu(self) -> _Values:
        return self

    def tolist(self) -> list[float]:
        return self._values


@dataclass
class _Policy:
    seen: list[np.ndarray]

    def obs_to_tensor(self, observation: np.ndarray) -> tuple[np.ndarray, bool]:
        self.seen.append(observation)
        return observation, False


class _FakeDqn:
    """Just enough of an SB3 DQN: a policy that tensorises, a Q-network."""

    def __init__(self, q_values: list[float]) -> None:
        self.policy = _Policy(seen=[])
        self._q_values = q_values

    def q_net(self, _tensor: object) -> _Values:
        return _Values(self._q_values)


class TestController:
    def test_it_is_an_ilocalcontroller(self) -> None:
        assert isinstance(DqnLocalController(_FakeDqn([0.0] * 5)), ILocalController)

    def test_it_takes_the_highest_valued_action(self) -> None:
        values = [0.1, 0.2, 0.9, 0.3, 0.0]
        decision = DqnLocalController(_FakeDqn(values)).decide(
            _observation(Heading.NORTH, Position(10, 9))
        )
        assert decision.action is LocalAction.TURN_LEFT

    def test_every_q_value_is_reported(self) -> None:
        values = [0.5, -1.0, 0.2, 0.1, 0.0]
        decision = DqnLocalController(_FakeDqn(values)).decide(
            _observation(Heading.NORTH, Position(10, 9))
        )
        assert decision.q_values == dict(zip(LOCAL_ACTION_ORDER, values, strict=True))

    def test_the_model_sees_the_policy_features(self) -> None:
        model = _FakeDqn([1.0, 0, 0, 0, 0])
        observation = _observation(Heading.EAST, Position(11, 10))
        DqnLocalController(model).decide(observation)
        assert np.array_equal(model.policy.seen[0], policy_features(observation))

    def test_missing_weights_fail_before_anything_loads(self, tmp_path: Path) -> None:
        with pytest.raises(AssetNotFoundError):
            DqnLocalController.from_file(tmp_path / "absent.zip")
