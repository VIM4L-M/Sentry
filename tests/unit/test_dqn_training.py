"""Unit tests for the DQN config, the controller comparison, and the trainer (Phase 6).

The comparison logic decides M6, so it is tested on hand-built results.
The trainer runs end to end for a few hundred steps: far too few to learn
anything, but enough to prove it evaluates on held-out missions, keeps a
checkpoint, and that the checkpoint loads into the live adapter.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from sentry_ai.common.exceptions import ConfigurationError, ConfigValidationError
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import DqnTrainingConfig, RewardConfig
from sentry_ai.simulation.mission import MissionPhase
from sentry_ai.training.dqn import (
    ControllerScore,
    MissionResult,
    PolicyComparison,
    drive_missions,
)
from sentry_ai.training.missions import MissionFactory


def _result(
    rescued: int, phase: MissionPhase = MissionPhase.COMPLETED, **kw: object
) -> MissionResult:
    values: dict[str, object] = {
        "seed": 0,
        "phase": phase,
        "rescued": rescued,
        "lost": 0,
        "collisions": 0,
        "ticks": 100,
    }
    values.update(kw)
    return MissionResult(**values)  # type: ignore[arg-type]


def _score(*results: MissionResult) -> ControllerScore:
    return ControllerScore(results=results)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def factory() -> MissionFactory:
    loader = ConfigLoader(project_root=_PROJECT_ROOT)
    return MissionFactory.from_app_config(loader, loader.load_app_config("configs/app.yaml"))


class TestConfig:
    def test_the_shipped_file_loads(self, project_root: Path) -> None:
        loader = ConfigLoader(project_root=project_root)
        config = loader.load_dqn_config("configs/training/dqn.yaml")
        assert config.runs_dir == project_root / "models/dqn"
        assert config.reward.reverse < 0.0
        assert config.hidden_sizes == (64, 64)

    def test_training_and_evaluation_seeds_never_overlap(self) -> None:
        config = DqnTrainingConfig(train_seeds=50, eval_seeds=10, seed_base=100)
        assert not set(config.training_seeds) & set(config.evaluation_seeds)
        assert len(config.evaluation_seeds) == 10

    @pytest.mark.parametrize(
        "overrides",
        [
            {"total_timesteps": 0},
            {"gamma": 1.5},
            {"rescue_threshold": 0.0},
            {"hidden_sizes": ()},
            {"hidden_sizes": (64, 0)},
            {"learning_starts": -1},
        ],
    )
    def test_invalid_values_are_rejected(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ConfigValidationError):
            DqnTrainingConfig(**overrides)  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["collision", "step", "reverse", "failure"])
    def test_a_penalty_cannot_be_a_reward(self, field: str) -> None:
        with pytest.raises(ConfigValidationError, match="penalty"):
            RewardConfig(**{field: 1.0})  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["progress", "pickup", "delivery", "completion"])
    def test_a_reward_cannot_be_a_penalty(self, field: str) -> None:
        with pytest.raises(ConfigValidationError):
            RewardConfig(**{field: -1.0})  # type: ignore[arg-type]

    def test_hidden_sizes_must_be_a_list_of_ints(self, tmp_path: Path) -> None:
        (tmp_path / "dqn.yaml").write_text("hidden_sizes: [64, 'wide']\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="hidden_sizes"):
            ConfigLoader(project_root=tmp_path).load_dqn_config("dqn.yaml")

    def test_reward_keys_are_optional(self, tmp_path: Path) -> None:
        (tmp_path / "dqn.yaml").write_text("reward:\n  pickup: 7.0\n", encoding="utf-8")
        config = ConfigLoader(project_root=tmp_path).load_dqn_config("dqn.yaml")
        assert config.reward.pickup == 7.0
        assert config.reward.delivery == RewardConfig().delivery


class TestScores:
    def test_totals_add_up_across_missions(self) -> None:
        score = _score(_result(3, lost=1, collisions=2), _result(4, ticks=300))
        assert (score.rescued, score.lost, score.collisions) == (7, 1, 2)
        assert score.mean_ticks == pytest.approx(200.0)

    def test_completion_rate_counts_completed_missions(self) -> None:
        score = _score(_result(4), _result(2, MissionPhase.FAILED, failure_reason="timer"))
        assert score.completion_rate == pytest.approx(0.5)

    def test_failures_are_grouped_by_reason(self) -> None:
        score = _score(
            _result(0, MissionPhase.FAILED, failure_reason="hospital unreachable"),
            _result(0, MissionPhase.FAILED, failure_reason="hospital unreachable"),
            _result(4),
        )
        assert score.failure_reasons() == {"hospital unreachable": 2}

    def test_an_empty_score_is_all_zero(self) -> None:
        assert _score().completion_rate == 0.0 and _score().mean_ticks == 0.0


class TestComparison:
    def test_the_ratio_is_candidate_over_baseline(self) -> None:
        baseline = _score(_result(4), _result(4))
        comparison = PolicyComparison(baseline, _score(_result(4), _result(3)))
        assert comparison.rescue_ratio == pytest.approx(7 / 8)

    def test_m6_is_a_threshold_on_the_ratio(self) -> None:
        comparison = PolicyComparison(_score(_result(10)), _score(_result(9)))
        assert comparison.meets(0.9)
        assert not comparison.meets(0.91)

    def test_nobody_rescued_by_either_is_parity(self) -> None:
        assert PolicyComparison(_score(_result(0)), _score(_result(0))).rescue_ratio == 1.0


class TestDriveMissions:
    def test_the_follower_rescues_everyone_on_a_clear_mission(
        self, factory: MissionFactory
    ) -> None:
        score = drive_missions(factory.build, lambda: None, [1], max_ticks=1500)
        assert score.results[0].phase is MissionPhase.COMPLETED
        assert score.rescued == 4

    def test_one_result_per_seed_in_order(self, factory: MissionFactory) -> None:
        score = drive_missions(factory.build, lambda: None, [5, 2, 9], max_ticks=50)
        assert [result.seed for result in score.results] == [5, 2, 9]


class TestTrainer:
    def test_a_short_run_keeps_a_loadable_checkpoint(
        self, factory: MissionFactory, tmp_path: Path
    ) -> None:
        pytest.importorskip("stable_baselines3")
        from sentry_ai.decision.dqn_controller import DqnLocalController
        from sentry_ai.training.dqn import DqnTrainer

        config = replace(
            DqnTrainingConfig(),
            runs_dir=tmp_path,
            train_seeds=5,
            eval_seeds=2,
            eval_episodes=2,
            total_timesteps=400,
            learning_starts=100,
            eval_every=200,
            max_episode_steps=100,
            buffer_size=1000,
            device="cpu",
        )
        rows: list[dict[str, float]] = []
        outcome = DqnTrainer(config, factory.build).train("tiny", on_evaluation=rows.append)

        assert [row["step"] for row in rows] == [200.0, 400.0]
        assert outcome.weights_path.is_file()
        assert (outcome.run_dir / "best.json").is_file()
        assert (outcome.run_dir / "last.zip").is_file()
        with (outcome.run_dir / "metrics.csv").open(newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 2
        assert outcome.comparison.baseline.missions == 2
        DqnLocalController.from_file(outcome.weights_path)
