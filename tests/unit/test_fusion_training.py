"""Unit tests for sentry_ai.training.fusion — the label, the scores, the recorder, the trainer.

The label ("the DQN's move, made safe") is what fusion learns, and the
veto metrics are what M7's mission-level claim rests on, so both are pinned
down on hand-built cases. The camera-veto rule is scored from recorded
features by a fast path; a test proves it matches the live rule exactly.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="fusion training needs torch")

from sentry_ai.common.exceptions import AssetNotFoundError, ConfigValidationError  # noqa: E402
from sentry_ai.config.loader import ConfigLoader  # noqa: E402
from sentry_ai.config.schema import FusionTrainingConfig  # noqa: E402
from sentry_ai.decision.fusion import (  # noqa: E402
    FUSION_FEATURES,
    CameraVetoFusion,
    FusionArchitecture,
    MlpFusion,
    fusion_features,
)
from sentry_ai.domain.entities import Position  # noqa: E402
from sentry_ai.domain.enums import EntityKind, Heading  # noqa: E402
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid  # noqa: E402
from sentry_ai.interfaces.navigation import (  # noqa: E402
    LOCAL_ACTION_ORDER,
    LocalAction,
    LocalDecision,
    LocalObservation,
)
from sentry_ai.interfaces.perception import WorldDetection  # noqa: E402
from sentry_ai.interfaces.sequence import BehaviourClass, BehaviourSignal  # noqa: E402
from sentry_ai.training.fusion import (  # noqa: E402
    DecisionReport,
    FusionSamples,
    safe_action,
    score_reference,
    train_model,
)

F, R, L, T, S = (
    LOCAL_ACTION_ORDER.index(a)
    for a in (
        LocalAction.MOVE_FORWARD,
        LocalAction.REVERSE,
        LocalAction.TURN_LEFT,
        LocalAction.TURN_RIGHT,
        LocalAction.STOP,
    )
)


def _grid_with_debris(x: int, y: int) -> OccupancyGrid:
    grid = OccupancyGrid.empty(10, 10)
    grid.mark(Position(x, y), OccupancyCode.DEBRIS)
    return grid


class TestSafeAction:
    def test_driving_into_debris_becomes_stop(self) -> None:
        grid = _grid_with_debris(5, 4)
        assert safe_action(LocalAction.MOVE_FORWARD, Position(5, 5), Heading.NORTH, grid) is (
            LocalAction.STOP
        )

    def test_reversing_into_debris_becomes_stop(self) -> None:
        grid = _grid_with_debris(5, 6)
        assert safe_action(LocalAction.REVERSE, Position(5, 5), Heading.NORTH, grid) is (
            LocalAction.STOP
        )

    def test_a_clear_move_is_kept(self) -> None:
        grid = _grid_with_debris(9, 9)
        assert safe_action(LocalAction.MOVE_FORWARD, Position(5, 5), Heading.EAST, grid) is (
            LocalAction.MOVE_FORWARD
        )

    @pytest.mark.parametrize(
        "action", [LocalAction.TURN_LEFT, LocalAction.TURN_RIGHT, LocalAction.STOP]
    )
    def test_actions_that_do_not_move_are_always_safe(self, action: LocalAction) -> None:
        grid = _grid_with_debris(5, 4)
        assert safe_action(action, Position(5, 5), Heading.NORTH, grid) is action

    def test_driving_off_the_map_is_stopped(self) -> None:
        grid = OccupancyGrid.empty(10, 10)
        assert safe_action(LocalAction.MOVE_FORWARD, Position(0, 0), Heading.WEST, grid) is (
            LocalAction.STOP
        )


class TestDecisionReport:
    def test_perfect_decisions(self) -> None:
        labels = np.array([F, S, T])
        report = DecisionReport.score(labels, labels, np.array([F, F, T]))
        assert report.accuracy == 1.0 and report.macro_f1 == 1.0
        assert report.veto_recall == 1.0 and report.false_veto_rate == 0.0

    def test_passing_the_dqn_through_vetoes_nothing(self) -> None:
        proposed = np.array([F, F, T, F])
        labels = np.array([F, S, T, S])  # two of the DQN's moves were unsafe
        report = DecisionReport.score(labels, proposed, proposed)
        assert report.veto_recall == 0.0 and report.false_veto_rate == 0.0

    def test_stopping_everywhere_is_all_false_vetoes(self) -> None:
        proposed = np.array([F, F, F, F])
        labels = np.array([F, S, F, F])
        report = DecisionReport.score(labels, np.full(4, S), proposed)
        assert report.veto_recall == 1.0
        assert report.false_veto_rate == 1.0

    def test_no_unsafe_moves_leaves_recall_undefined(self) -> None:
        proposed = np.array([F, T])
        assert math.isnan(DecisionReport.score(proposed, proposed, proposed).veto_recall)


def _samples(features: np.ndarray, labels: list[int], proposed: list[int]) -> FusionSamples:
    return FusionSamples(
        features=features.astype(np.float32),
        labels=np.array(labels, dtype=np.int64),
        proposed=np.array(proposed, dtype=np.int64),
        seeds=np.zeros(len(labels), dtype=np.int64),
    )


def _live_sample(
    action: LocalAction, sightings: list[WorldDetection]
) -> tuple[np.ndarray, LocalObservation, LocalDecision, BehaviourSignal]:
    observation = LocalObservation(
        position=Position(5, 5),
        heading=Heading.EAST,
        battery_percent=80.0,
        next_waypoint=Position(6, 5),
        blocked_ahead=False,
        fire_proximity=0.0,
    )
    decision = LocalDecision(action, {a: float(a is action) for a in LOCAL_ACTION_ORDER})
    behaviour = BehaviourSignal(BehaviourClass.ADVANCE, 0.9)
    return (
        fusion_features(observation, sightings, behaviour, decision),
        observation,
        decision,
        behaviour,
    )


class TestReferences:
    def test_the_dqn_reference_is_the_proposal(self) -> None:
        samples = _samples(np.zeros((3, FUSION_FEATURES)), [F, S, T], [F, F, T])
        report = score_reference(samples, None)
        assert report.veto_recall == 0.0

    def test_the_recorded_rule_matches_the_live_rule(self) -> None:
        """The fast path over recorded features must decide exactly as CameraVetoFusion does."""
        cases = [
            (
                LocalAction.MOVE_FORWARD,
                [WorldDetection(EntityKind.OBSTACLE, frozenset({Position(6, 5)}), 0.9)],
            ),
            (
                LocalAction.MOVE_FORWARD,
                [WorldDetection(EntityKind.FIRE, frozenset({Position(6, 5)}), 0.4)],
            ),
            (
                LocalAction.REVERSE,
                [WorldDetection(EntityKind.OBSTACLE, frozenset({Position(4, 5)}), 0.8)],
            ),
            (
                LocalAction.TURN_LEFT,
                [WorldDetection(EntityKind.OBSTACLE, frozenset({Position(6, 5)}), 0.9)],
            ),
            (
                LocalAction.MOVE_FORWARD,
                [WorldDetection(EntityKind.VICTIM, frozenset({Position(6, 5)}), 0.9)],
            ),
        ]
        rule = CameraVetoFusion()
        rows, proposed, live = [], [], []
        for action, sightings in cases:
            features, observation, decision, behaviour = _live_sample(action, sightings)
            rows.append(features)
            proposed.append(LOCAL_ACTION_ORDER.index(action))
            live.append(
                LOCAL_ACTION_ORDER.index(
                    rule.fuse(observation, sightings, behaviour, decision).action
                )
            )
        samples = _samples(np.stack(rows), live, proposed)
        report = score_reference(samples, rule)
        assert report.accuracy == 1.0


class TestSamples:
    def test_they_round_trip(self, tmp_path: Path) -> None:
        samples = _samples(
            np.random.default_rng(0).random((4, FUSION_FEATURES)), [F, S, T, R], [F, F, T, R]
        )
        samples.save(tmp_path / "s.npz")
        loaded = FusionSamples.load(tmp_path / "s.npz")
        assert np.array_equal(loaded.features, samples.features)
        assert loaded.vetoes == 1

    def test_missing_samples_say_how_to_make_them(self, tmp_path: Path) -> None:
        with pytest.raises(AssetNotFoundError, match="--record"):
            FusionSamples.load(tmp_path / "absent.npz")


class TestConfig:
    def test_the_shipped_file_loads(self) -> None:
        loader = ConfigLoader(project_root=Path(__file__).resolve().parents[2])
        config = loader.load_fusion_config("configs/training/fusion.yaml")
        assert config.belief_lag > 0
        assert not set(config.training_seeds) & set(config.validation_seeds)

    @pytest.mark.parametrize(
        "overrides",
        [{"belief_lag": -1}, {"epochs": 0}, {"dropout": 1.0}, {"hidden_sizes": ()}],
    )
    def test_invalid_values_are_rejected(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ConfigValidationError):
            FusionTrainingConfig(**overrides)  # type: ignore[arg-type]


class TestTraining:
    def test_it_learns_to_veto_what_the_camera_sees(self, tmp_path: Path) -> None:
        """Synthetic but realistic: DQN says forward; stop iff debris is sighted ahead."""
        rng = np.random.default_rng(0)
        rows, labels, proposed = [], [], []
        for _ in range(600):
            debris = rng.random() < 0.3
            sightings = (
                [WorldDetection(EntityKind.OBSTACLE, frozenset({Position(6, 5)}), 0.9)]
                if debris
                else []
            )
            features, *_ = _live_sample(LocalAction.MOVE_FORWARD, sightings)
            rows.append(features)
            labels.append(S if debris else F)
            proposed.append(F)
        samples = _samples(np.stack(rows), labels, proposed)
        config = replace(FusionTrainingConfig(), epochs=40, batch_size=64, device="cpu", patience=0)

        fusion = train_model(
            "fusion", FusionArchitecture(), samples, samples, config, tmp_path / "a"
        )
        dqn_only = train_model(
            "dqn only",
            FusionArchitecture(groups=("dqn",)),
            samples,
            samples,
            config,
            tmp_path / "b",
        )
        assert fusion.report.veto_recall == 1.0 and fusion.report.false_veto_rate == 0.0
        assert fusion.report.macro_f1 > dqn_only.report.macro_f1
        MlpFusion.from_checkpoint(fusion.weights_path)
