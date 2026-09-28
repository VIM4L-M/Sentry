"""Trains the DQN local controller and scores it against the waypoint follower (Unit V).

**What counts is the mission, not the reward.** Episode return is how the
agent learns; it is not how the result is judged. A reward can be gamed —
circling to farm progress, stopping to avoid collision penalties — while
the mission quietly fails. So both during training (to pick the best
checkpoint) and at the end (for M6), the policy drives *real missions*
through :class:`~sentry_ai.decision.dqn_controller.DqnLocalController`, the
same adapter the live pipeline uses, and is scored on rescues, losses,
collisions and completion — next to the waypoint follower on exactly the
same missions.

**M6** is met when, on held-out missions, the DQN rescues at least
``rescue_threshold`` of the victims the follower rescues. The follower
rescues everyone on this map, so M6 asks a learned policy to match a
hand-written one that is already perfect at the task, having been told
nothing about driving but the reward.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.schema import DqnTrainingConfig
from sentry_ai.decision.dqn_controller import DqnLocalController
from sentry_ai.interfaces.navigation import ILocalController
from sentry_ai.simulation.factory import Mission
from sentry_ai.simulation.mission import MissionPhase
from sentry_ai.training.checkpoint import write_metadata
from sentry_ai.training.metrics import CsvMetricLogger
from sentry_ai.training.missions import DrivenMissionSource
from sentry_ai.training.seed import resolve_device, seed_everything
from sentry_ai.training.sentry_env import SentryEnv

logger = get_logger(__name__)

#: Called after every held-out evaluation during training.
EvaluationCallback = Callable[[dict[str, float]], None]


@dataclass(frozen=True)
class MissionResult:
    """How one mission went under one controller."""

    seed: int
    phase: MissionPhase
    rescued: int
    lost: int
    collisions: int
    ticks: int
    failure_reason: str = ""
    damage: float = 0.0


@dataclass(frozen=True)
class ControllerScore:
    """A controller's results over a set of missions."""

    results: tuple[MissionResult, ...]

    @property
    def missions(self) -> int:
        """Missions driven."""
        return len(self.results)

    @property
    def rescued(self) -> int:
        """Victims delivered to the hospital, across every mission."""
        return sum(result.rescued for result in self.results)

    @property
    def lost(self) -> int:
        """Victims lost, across every mission."""
        return sum(result.lost for result in self.results)

    @property
    def collisions(self) -> int:
        """Refused moves, across every mission."""
        return sum(result.collisions for result in self.results)

    @property
    def mean_damage(self) -> float:
        """Average vehicle health lost per mission, in percentage points."""
        if not self.missions:
            return 0.0
        return sum(result.damage for result in self.results) / self.missions

    @property
    def completion_rate(self) -> float:
        """Share of missions that ended COMPLETED."""
        completed = sum(result.phase is MissionPhase.COMPLETED for result in self.results)
        return completed / self.missions if self.missions else 0.0

    @property
    def mean_ticks(self) -> float:
        """Average mission length in ticks — how quickly the job gets done."""
        if not self.missions:
            return 0.0
        return sum(result.ticks for result in self.results) / self.missions

    def failure_reasons(self) -> dict[str, int]:
        """How many missions failed, by reason."""
        reasons: dict[str, int] = {}
        for result in self.results:
            if result.phase is MissionPhase.FAILED:
                reason = result.failure_reason or "unknown"
                reasons[reason] = reasons.get(reason, 0) + 1
        return reasons

    def as_row(self) -> dict[str, float]:
        """The headline numbers, for a metric log."""
        return {
            "rescued": float(self.rescued),
            "lost": float(self.lost),
            "collisions": float(self.collisions),
            "completion_rate": self.completion_rate,
            "mean_ticks": self.mean_ticks,
        }


@dataclass(frozen=True)
class PolicyComparison:
    """A candidate controller next to the baseline on identical missions."""

    baseline: ControllerScore
    candidate: ControllerScore

    @property
    def rescue_ratio(self) -> float:
        """Candidate rescues as a share of baseline rescues; 1.0 if both rescued nobody."""
        if self.baseline.rescued == 0:
            return 1.0 if self.candidate.rescued == 0 else float("inf")
        return self.candidate.rescued / self.baseline.rescued

    def meets(self, threshold: float) -> bool:
        """Milestone M6: rescues at least ``threshold`` of what the baseline rescues."""
        return self.rescue_ratio >= threshold


def drive_missions(
    factory: DrivenMissionSource,
    controller: Callable[[], ILocalController | None],
    seeds: Sequence[int],
    max_ticks: int,
    prepare: Callable[[Mission, ILocalController | None], None] | None = None,
) -> ControllerScore:
    """Run one mission per seed with a fresh controller and collect the results.

    ``controller`` returning ``None`` means the factory's default driver —
    the waypoint follower. ``prepare`` runs on each built mission before it
    starts — how a controller whose camera needs the mission's city gets it.
    """
    results = []
    for seed in seeds:
        driver = controller()
        mission = factory(seed, driver)
        if prepare is not None:
            prepare(mission, driver)
        stats = mission.engine.run(max_ticks=max_ticks)
        results.append(
            MissionResult(
                seed=seed,
                phase=mission.controller.phase,
                rescued=stats.victims_rescued,
                lost=stats.victims_lost,
                collisions=stats.collisions,
                ticks=stats.ticks,
                failure_reason=stats.failure_reason,
                damage=100.0 - mission.city_map.vehicle.health_percent,
            )
        )
    return ControllerScore(results=tuple(results))


@dataclass(frozen=True)
class DqnTrainingOutcome:
    """What a finished training run produced.

    Attributes:
        weights_path: The best checkpoint, chosen on held-out missions.
        run_dir: Holds ``best.zip``, ``last.zip``, metadata and ``metrics.csv``.
        comparison: The best checkpoint against the follower on the
            evaluation missions used during training.
        best_step: The training step the best checkpoint was taken at.
    """

    weights_path: Path
    run_dir: Path
    comparison: PolicyComparison
    best_step: int


class DqnTrainer:
    """Trains a Stable-Baselines3 DQN in :class:`SentryEnv`."""

    def __init__(self, config: DqnTrainingConfig, factory: DrivenMissionSource) -> None:
        """Create a trainer.

        Args:
            config: Environment, hyperparameters and reward.
            factory: Builds a mission for a seed and a driver —
                :meth:`MissionFactory.build`.
        """
        self._config = config
        self._factory = factory

    def resolve_device(self) -> str:
        """The device to train on. Public so a script can report it up front."""
        return resolve_device(self._config.device)

    def train(
        self, run_name: str = "sentry", on_evaluation: EvaluationCallback | None = None
    ) -> DqnTrainingOutcome:
        """Train, evaluate on held-out missions periodically, keep the best."""
        from stable_baselines3 import DQN  # noqa: PLC0415 - keeps SB3 off the import path
        from stable_baselines3.common.callbacks import BaseCallback  # noqa: PLC0415
        from stable_baselines3.common.monitor import Monitor  # noqa: PLC0415

        config = self._config
        seed_everything(config.seed)
        run_dir = config.runs_dir / run_name
        env: Any = Monitor(
            SentryEnv(
                self._factory, config.training_seeds, config.reward, config.max_episode_steps
            )
        )
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            gamma=config.gamma,
            train_freq=config.train_freq,
            target_update_interval=config.target_update_interval,
            exploration_fraction=config.exploration_fraction,
            exploration_final_eps=config.exploration_final_eps,
            policy_kwargs={"net_arch": list(config.hidden_sizes)},
            seed=config.seed,
            device=self.resolve_device(),
            verbose=0,
        )

        eval_seeds = list(config.evaluation_seeds)[: config.eval_episodes]
        baseline = drive_missions(
            self._factory, lambda: None, eval_seeds, config.max_episode_steps
        )
        logger.info(
            "Baseline (waypoint follower) on %d held-out missions: %d rescued, %.0f%% completed",
            len(eval_seeds),
            baseline.rescued,
            100 * baseline.completion_rate,
        )
        metrics = CsvMetricLogger(run_dir / "metrics.csv")
        best: dict[str, Any] = {"key": None, "comparison": None, "step": 0}

        def evaluate(step: int) -> None:
            candidate = drive_missions(
                self._factory,
                lambda: DqnLocalController(model),
                eval_seeds,
                config.max_episode_steps,
            )
            comparison = PolicyComparison(baseline=baseline, candidate=candidate)
            row = {
                "step": float(step),
                **candidate.as_row(),
                "rescue_ratio": comparison.rescue_ratio,
            }
            metrics.log(row)
            if on_evaluation is not None:
                on_evaluation(row)
            # Rescues first, then finishing missions, then fewer collisions,
            # then speed — the order the mission itself cares about them.
            key = (
                candidate.rescued,
                candidate.completion_rate,
                -candidate.collisions,
                -candidate.mean_ticks,
            )
            if best["key"] is None or key > best["key"]:
                best.update(key=key, comparison=comparison, step=step)
                model.save(str(run_dir / "best.zip"))
                write_metadata(run_dir / "best.zip", config, {"step": float(step), **row})

        class _Evaluate(BaseCallback):
            def _on_step(self) -> bool:
                if self.num_timesteps % config.eval_every == 0:
                    evaluate(self.num_timesteps)
                return True

        run_dir.mkdir(parents=True, exist_ok=True)
        model.learn(total_timesteps=config.total_timesteps, callback=_Evaluate())
        model.save(str(run_dir / "last.zip"))
        write_metadata(run_dir / "last.zip", config, {"step": float(config.total_timesteps)})
        if best["comparison"] is None:
            evaluate(config.total_timesteps)

        return DqnTrainingOutcome(
            weights_path=run_dir / "best.zip",
            run_dir=run_dir,
            comparison=best["comparison"],
            best_step=int(best["step"]),
        )
