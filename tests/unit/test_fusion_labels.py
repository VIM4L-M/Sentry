"""Unit tests for the two fusion labels (``FusionTrainingConfig.label_mode``)."""

from __future__ import annotations

import random

import pytest

from sentry_ai.common.exceptions import ConfigValidationError
from sentry_ai.config.schema import FusionTrainingConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.interfaces.decision import SceneEvidence
from sentry_ai.interfaces.navigation import (
    LOCAL_ACTION_ORDER,
    ILocalController,
    LocalAction,
    LocalDecision,
    LocalObservation,
)
from sentry_ai.training.fusion import FusionRecorder

_OBSERVATION = LocalObservation(
    position=Position(2, 2),
    heading=Heading.EAST,
    battery_percent=90.0,
    next_waypoint=Position(4, 2),
    blocked_ahead=False,
    fire_proximity=0.0,
)


class _Script(ILocalController):
    """The stale-map DQN says ``stale``; on the true map it would say ``truth``."""

    def __init__(self, stale: LocalAction, truth: LocalAction) -> None:
        self._answers = [stale, truth]

    def decide(self, observation: LocalObservation) -> LocalDecision:
        action = self._answers[0]
        self._answers.reverse()
        return LocalDecision(action, {a: 0.0 for a in LOCAL_ACTION_ORDER})


class _Engine:
    """Just what the recorder asks of an engine: the true grid and an observation on it."""

    def __init__(self, blocked: Position | None) -> None:
        self.grid = OccupancyGrid.empty(6, 6)
        if blocked is not None:
            self.grid.mark(blocked, OccupancyCode.DEBRIS)

    def physics_grid(self) -> OccupancyGrid:
        return self.grid

    def observe(self, grid: OccupancyGrid | None = None) -> LocalObservation:
        return _OBSERVATION


def _label(mode: str, stale: LocalAction, truth: LocalAction, blocked: Position | None) -> int:
    recorder = FusionRecorder(_Script(stale, truth), None, 0.0, random.Random(0), label_mode=mode)
    recorder.bind(_Engine(blocked), SceneEvidence.empty, seed=1)  # type: ignore[arg-type]
    recorder.decide(_OBSERVATION)
    return int(recorder.dataset().labels[0])


def _index(action: LocalAction) -> int:
    return LOCAL_ACTION_ORDER.index(action)


def test_veto_keeps_a_safe_dqn_move() -> None:
    label = _label("veto", LocalAction.MOVE_FORWARD, LocalAction.TURN_LEFT, blocked=None)
    assert label == _index(LocalAction.MOVE_FORWARD)


def test_veto_stops_a_move_into_real_debris() -> None:
    label = _label("veto", LocalAction.MOVE_FORWARD, LocalAction.TURN_LEFT, Position(3, 2))
    assert label == _index(LocalAction.STOP)


def test_veto_stops_a_reverse_into_real_debris() -> None:
    label = _label("veto", LocalAction.REVERSE, LocalAction.MOVE_FORWARD, Position(1, 2))
    assert label == _index(LocalAction.STOP)


def test_truth_records_the_true_map_decision() -> None:
    label = _label("truth", LocalAction.MOVE_FORWARD, LocalAction.TURN_LEFT, Position(3, 2))
    assert label == _index(LocalAction.TURN_LEFT)


def test_an_unknown_label_mode_is_rejected() -> None:
    with pytest.raises(ConfigValidationError):
        FusionTrainingConfig(label_mode="guess")
