"""Unit tests for Phase 5's data and training: trajectories, windows, reports, trainer.

The trainer runs end to end on trajectories written by hand — a vehicle
driving a square, so every window's label is known in advance — with a
network a few units wide. That is enough to prove the plumbing: windows are
cut and labelled by the shared rule, the baselines are scored on the same
windows as the model, the checkpoint loads into the live adapter. Whether
a real run beats the baselines is recorded in
docs/architecture/phase5-sequence.md.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from sentry_ai.common.exceptions import AssetNotFoundError, ConfigValidationError
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import LstmTrainingConfig
from sentry_ai.domain.entities import Position
from sentry_ai.interfaces.sequence import BehaviourClass, VehicleState
from sentry_ai.training.trajectories import Trajectory, TrajectorySet

torch = pytest.importorskip("torch", reason="the trainer needs torch (requirements-ml.txt)")

from sentry_ai.sequence.lstm_predictor import LstmMotionPredictor  # noqa: E402
from sentry_ai.training.motion import (  # noqa: E402
    ClassificationReport,
    LstmTrainer,
    labelled_windows,
    majority_class,
    novel_windows,
    score_majority,
    score_persistence,
    score_predictor,
)

A, R, H, D = (
    BehaviourClass.ADVANCE,
    BehaviourClass.RETREAT,
    BehaviourClass.HOLD,
    BehaviourClass.DIVERT,
)


def _state(x: int, y: int, heading: float) -> VehicleState:
    return VehicleState(Position(x, y), battery_percent=90.0, heading_degrees=heading)


def _square(seed: int, side: int = 4, laps: int = 2) -> Trajectory:
    """Drive clockwise round a square: east, south, west, north, turning on the spot."""
    states = [_state(1, 1, 90.0)]
    x, y = 1, 1
    legs = ((90.0, (1, 0)), (180.0, (0, 1)), (270.0, (-1, 0)), (0.0, (0, -1)))
    for _ in range(laps):
        for heading, (dx, dy) in legs:
            states.append(_state(x, y, heading))  # the turn: a tick without moving
            for _ in range(side):
                x, y = x + dx, y + dy
                states.append(_state(x, y, heading))
    return Trajectory(seed=seed, outcome="completed", states=tuple(states))


def _write_sets(root: Path) -> None:
    TrajectorySet(12, 12, tuple(_square(seed) for seed in range(3))).save(root / "train.json")
    TrajectorySet(12, 12, (_square(99),)).save(root / "val.json")


def _config(root: Path, **overrides: object) -> LstmTrainingConfig:
    base = LstmTrainingConfig(
        trajectories_dir=root,
        runs_dir=root / "runs",
        window=5,
        horizon=3,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
        epochs=2,
        batch_size=8,
        device="cpu",
        seed=1,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


class TestTrajectorySet:
    def test_it_round_trips_through_json(self, tmp_path: Path) -> None:
        original = TrajectorySet(12, 12, (_square(4),))
        original.save(tmp_path / "t.json")
        assert TrajectorySet.load(tmp_path / "t.json") == original

    def test_it_counts_states(self) -> None:
        assert TrajectorySet(12, 12, (_square(0), _square(1))).total_states == 2 * 41

    def test_a_missing_file_says_how_to_make_one(self, tmp_path: Path) -> None:
        with pytest.raises(AssetNotFoundError, match="record_trajectories"):
            TrajectorySet.load(tmp_path / "absent.json")

    def test_an_unknown_format_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        path.write_text('{"format": 99}', encoding="utf-8")
        with pytest.raises(ValueError, match="format"):
            TrajectorySet.load(path)


class TestWindows:
    def test_windows_are_labelled_by_what_happens_next(self) -> None:
        windows = labelled_windows([_square(0)], window=3, horizon=2, cone_degrees=30.0)
        # First window ends on state 2: (2,1) facing east; two ticks later (4,1).
        assert windows[0].label is A
        assert len(windows[0].states) == 3

    def test_every_window_has_its_full_horizon(self) -> None:
        trajectory = _square(0)
        windows = labelled_windows([trajectory], window=4, horizon=3, cone_degrees=30.0)
        assert len(windows) == len(trajectory.states) - 4 - 3 + 1

    def test_windows_never_straddle_trajectories(self) -> None:
        one = labelled_windows([_square(0)], window=4, horizon=2, cone_degrees=30.0)
        two = labelled_windows([_square(0), _square(1)], window=4, horizon=2, cone_degrees=30.0)
        assert len(two) == 2 * len(one)

    def test_a_turning_corner_is_a_divert(self) -> None:
        windows = labelled_windows([_square(0)], window=3, horizon=3, cone_degrees=30.0)
        assert D in {window.label for window in windows}

    def test_a_too_short_trajectory_yields_nothing(self) -> None:
        short = Trajectory(0, "completed", (_state(0, 0, 90.0),) * 3)
        assert labelled_windows([short], window=3, horizon=2, cone_degrees=30.0) == []


class TestNovelWindows:
    def test_a_repeated_route_has_nothing_novel(self) -> None:
        train = labelled_windows([_square(0)], window=4, horizon=2, cone_degrees=30.0)
        held_out = labelled_windows([_square(1)], window=4, horizon=2, cone_degrees=30.0)
        assert novel_windows(held_out, train) == []

    def test_a_new_street_is_novel(self) -> None:
        train = labelled_windows([_square(0)], window=4, horizon=2, cone_degrees=30.0)
        elsewhere = Trajectory(1, "completed", tuple(_state(x, 9, 90.0) for x in range(8)))
        held_out = labelled_windows([elsewhere], window=4, horizon=2, cone_degrees=30.0)
        assert novel_windows(held_out, train) == held_out

    def test_battery_does_not_make_a_window_new(self) -> None:
        """Same streets, same headings, different charge: the same situation."""
        train = labelled_windows([_square(0)], window=4, horizon=2, cone_degrees=30.0)
        drained = Trajectory(
            1,
            "completed",
            tuple(
                VehicleState(s.position, battery_percent=10.0, heading_degrees=s.heading_degrees)
                for s in _square(0).states
            ),
        )
        held_out = labelled_windows([drained], window=4, horizon=2, cone_degrees=30.0)
        assert novel_windows(held_out, train) == []


class TestClassificationReport:
    def test_a_perfect_predictor_scores_one(self) -> None:
        report = ClassificationReport.from_predictions([A, D, R], [A, D, R])
        assert report.accuracy == 1.0 and report.macro_f1 == 1.0

    def test_always_the_majority_is_punished_by_macro_f1(self) -> None:
        """Seven in ten right, but two classes never predicted: macro-F1 sees it."""
        truth = [A] * 7 + [D] * 2 + [R]
        report = ClassificationReport.from_predictions(truth, [A] * 10)
        assert report.accuracy == pytest.approx(0.7)
        assert report.macro_f1 == pytest.approx((2 * 0.7 / 1.7) / 3)

    def test_absent_classes_do_not_count_against_anyone(self) -> None:
        report = ClassificationReport.from_predictions([A, D], [A, D])
        assert report.macro_f1 == 1.0  # HOLD and RETREAT never occurred

    def test_per_class_precision_and_recall(self) -> None:
        report = ClassificationReport.from_predictions([A, A, D, D], [A, D, D, D])
        advance, _, _, divert = report.per_class
        assert (advance.precision, advance.recall) == (1.0, 0.5)
        assert divert.precision == pytest.approx(2 / 3) and divert.recall == 1.0

    @pytest.mark.parametrize(("truth", "predicted"), [([], []), ([A], [A, D])])
    def test_malformed_input_is_rejected(
        self, truth: list[BehaviourClass], predicted: list[BehaviourClass]
    ) -> None:
        with pytest.raises(ValueError):
            ClassificationReport.from_predictions(truth, predicted)


class TestBaselines:
    def test_majority_is_the_most_common_label(self) -> None:
        windows = labelled_windows([_square(0)], window=3, horizon=2, cone_degrees=30.0)
        assert majority_class(windows) is A

    def test_majority_of_nothing_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            majority_class([])

    def test_baselines_are_scored_on_the_given_windows(self) -> None:
        windows = labelled_windows([_square(0)], window=5, horizon=3, cone_degrees=30.0)
        assert score_majority(windows, A).total == len(windows)
        assert score_persistence(windows, 3, 30.0).total == len(windows)

    def test_the_persistence_baseline_is_right_on_a_straight(self) -> None:
        straight = Trajectory(0, "completed", tuple(_state(x, 0, 90.0) for x in range(12)))
        windows = labelled_windows([straight], window=5, horizon=3, cone_degrees=30.0)
        assert score_persistence(windows, 3, 30.0).accuracy == 1.0


class TestConfig:
    def test_the_shipped_file_loads(self, project_root: Path) -> None:
        loader = ConfigLoader(project_root=project_root)
        config = loader.load_lstm_config("configs/training/lstm.yaml")
        assert config.trajectories_dir == project_root / "data/trajectories"
        assert config.horizon < config.window

    @pytest.mark.parametrize(
        "overrides",
        [
            {"window": 4, "horizon": 4},
            {"cone_degrees": 90.0},
            {"dropout": 1.0},
            {"hidden_size": 0},
            {"learning_rate": 0.0},
            {"patience": -1},
        ],
    )
    def test_invalid_values_are_rejected(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ConfigValidationError):
            LstmTrainingConfig(**overrides)  # type: ignore[arg-type]


class TestTrainer:
    def test_a_run_writes_loadable_weights_and_a_log(self, tmp_path: Path) -> None:
        _write_sets(tmp_path)
        rows: list[dict[str, float]] = []
        outcome = LstmTrainer(_config(tmp_path)).train("tiny", on_epoch=rows.append)

        assert outcome.epochs_run == 2
        assert [row["epoch"] for row in rows] == [1.0, 2.0]
        assert (outcome.run_dir / "last.pt").is_file()
        with (outcome.run_dir / "metrics.csv").open(newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 2

        predictor = LstmMotionPredictor.from_checkpoint(outcome.weights_path)
        predictor.predict(_square(5).states[:5])

    def test_both_baselines_are_reported(self, tmp_path: Path) -> None:
        _write_sets(tmp_path)
        outcome = LstmTrainer(_config(tmp_path, epochs=1)).train("tiny")
        assert set(outcome.baselines) == {"majority", "persistence"}
        assert outcome.baselines["majority"].total == outcome.report.total

    def test_the_report_matches_the_saved_weights(self, tmp_path: Path) -> None:
        """Scoring the best checkpoint through the port reproduces the trainer's report."""
        _write_sets(tmp_path)
        config = _config(tmp_path)
        outcome = LstmTrainer(config).train("tiny")
        val = TrajectorySet.load(tmp_path / "val.json")
        windows = labelled_windows(val.trajectories, config.window, config.horizon, 30.0)
        predictor = LstmMotionPredictor.from_checkpoint(outcome.weights_path)
        assert score_predictor(predictor, windows).confusion == outcome.report.confusion

    def test_it_can_learn_a_square(self, tmp_path: Path) -> None:
        """A route with no randomness at all should be learnable to beat 'always advance'."""
        _write_sets(tmp_path)
        config = _config(tmp_path, epochs=150, hidden_size=16, learning_rate=0.02, patience=0)
        outcome = LstmTrainer(config).train("square")
        assert outcome.report.macro_f1 > outcome.baselines["majority"].macro_f1

    def test_the_same_seed_trains_the_same_network(self, tmp_path: Path) -> None:
        _write_sets(tmp_path)
        first = LstmTrainer(_config(tmp_path)).train("a")
        second = LstmTrainer(_config(tmp_path)).train("b")
        assert first.report.confusion == second.report.confusion

    def test_maps_of_different_sizes_are_refused(self, tmp_path: Path) -> None:
        TrajectorySet(12, 12, (_square(0),)).save(tmp_path / "train.json")
        TrajectorySet(30, 20, (_square(1),)).save(tmp_path / "val.json")
        with pytest.raises(ValueError, match="map sizes"):
            LstmTrainer(_config(tmp_path)).train()

    def test_missing_trajectories_fail_before_training(self, tmp_path: Path) -> None:
        with pytest.raises(AssetNotFoundError):
            LstmTrainer(_config(tmp_path)).train()
