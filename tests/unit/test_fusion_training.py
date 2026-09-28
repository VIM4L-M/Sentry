"""Unit tests for sentry_ai.training.fusion and the fusion config (Phase 7)."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sentry_ai.common.exceptions import AssetNotFoundError, ConfigValidationError
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import FusionTrainingConfig
from sentry_ai.decision.mlp_fusion import FUSION_FEATURES, FeatureGroup
from sentry_ai.interfaces.navigation import LOCAL_ACTION_ORDER
from sentry_ai.simulation.waypoint_follower import WaypointFollower
from sentry_ai.training.fusion import ActionReport, FusionDataset, FusionRecorder, FusionTrainer


def _dataset(n: int, seed: int = 0) -> FusionDataset:
    """Ticks where the DQN is right unless 'debris ahead' is set; then STOP is right."""
    rng = np.random.default_rng(seed)
    features = np.zeros((n, FUSION_FEATURES), dtype=np.float32)
    dqn_choice = rng.integers(0, len(LOCAL_ACTION_ORDER), n)
    features[:, : len(LOCAL_ACTION_ORDER)] = -1.0
    features[np.arange(n), dqn_choice] = 0.0
    debris = rng.random(n) < 0.3
    features[debris, -5] = 0.9
    stop = LOCAL_ACTION_ORDER.index(LOCAL_ACTION_ORDER[-1])
    labels = np.where(debris, stop, dqn_choice).astype(np.int64)
    return FusionDataset(features, labels, labels != dqn_choice, np.zeros(n, dtype=np.int64))


class TestActionReport:
    def test_perfect_predictions_score_one(self) -> None:
        labels = np.array([0, 1, 4, 4])
        report = ActionReport.score(labels, labels, np.array([False, False, True, True]))
        assert (report.accuracy, report.macro_f1, report.critical_accuracy) == (1.0, 1.0, 1.0)
        assert report.critical_count == 2

    def test_critical_weight_shifts_weighted_accuracy(self) -> None:
        labels = np.array([0, 0, 4])
        guesses = np.array([0, 0, 0])
        critical = np.array([False, False, True])
        report = ActionReport.score(labels, guesses, critical, critical_weight=2.0)
        assert report.accuracy == pytest.approx(2 / 3)
        assert report.weighted_accuracy == pytest.approx(2 / 4)
        assert report.critical_accuracy == 0.0

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            ActionReport.score(np.array([0]), np.array([0, 1]), np.array([False]))


class TestFusionDataset:
    def test_dqn_predictions_are_the_policy_argmax(self) -> None:
        data = _dataset(50)
        expected = data.features[:, : len(LOCAL_ACTION_ORDER)].argmax(axis=1)
        assert np.array_equal(data.dqn_predictions, expected)

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        data = _dataset(20)
        data.save(tmp_path / "d.npz")
        loaded = FusionDataset.load(tmp_path / "d.npz")
        assert np.array_equal(loaded.features, data.features)
        assert np.array_equal(loaded.critical, data.critical)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AssetNotFoundError):
            FusionDataset.load(tmp_path / "absent.npz")


class TestFusionTrainer:
    def test_learns_to_stop_for_debris_the_dqn_cannot_see(self, tmp_path: Path) -> None:
        config = FusionTrainingConfig(
            runs_dir=tmp_path, epochs=40, batch_size=64, patience=0, device="cpu", dropout=0.0
        )
        outcome = FusionTrainer(config).train(
            _dataset(1500, 1), _dataset(400, 2), tuple(FeatureGroup), "t"
        )
        assert outcome.weights_path.is_file()
        assert outcome.report.critical_accuracy > 0.9
        assert outcome.report.accuracy > 0.9

    def test_a_policy_only_network_cannot_see_the_debris(self, tmp_path: Path) -> None:
        config = FusionTrainingConfig(runs_dir=tmp_path, epochs=10, patience=0, device="cpu")
        outcome = FusionTrainer(config).train(
            _dataset(800, 1), _dataset(200, 2), (FeatureGroup.POLICY,), "p"
        )
        assert outcome.report.critical_accuracy < 0.5


class TestFusionRecorder:
    def test_refuses_to_decide_before_bind(self) -> None:
        recorder = FusionRecorder(WaypointFollower(), None, 0.0, random.Random(0))
        with pytest.raises(RuntimeError):
            recorder.decide(None)  # type: ignore[arg-type]


class TestFusionConfig:
    def test_shipped_config_loads(self, project_root: Path) -> None:
        config = ConfigLoader(project_root=project_root).load_fusion_config(
            "configs/training/fusion.yaml"
        )
        assert config.belief_lag > 0
        assert config.hidden_sizes

    def test_seed_ranges_do_not_overlap(self) -> None:
        config = FusionTrainingConfig()
        train, val, test = (
            set(config.training_seeds),
            set(config.validation_seeds),
            set(config.evaluation_seeds),
        )
        assert not (train & val or train & test or val & test)

    def test_rejects_out_of_range_values(self) -> None:
        with pytest.raises(ConfigValidationError):
            replace(FusionTrainingConfig(), oracle_drive_probability=1.5)
