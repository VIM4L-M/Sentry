"""Collecting data for, training, and scoring the fusion MLP (Phase 7, Unit I).

**Where the labels come from.** The simulator knows the truth, so every
tick can be labelled exactly: *the action the DQN would have chosen had the
command center's map been current*. That is the DQN run once on the stale
map it really drives on, and once on the true world. On most ticks the two
agree. Where they differ — the map says the street is clear, a building
has just collapsed across it — the tick is **critical**, and the stale
choice is the mistake fusion exists to catch.

**Why split by mission.** Validation missions are disasters the network has
seen no tick of, the same rule every dataset in this project follows.

**What M7 measures.** Networks of identical shape are trained on every
signal and on each one alone (:class:`FeatureGroup`), and scored on held-out
ticks: overall accuracy and macro-F1 against the ground-truth decision, and
accuracy on the critical ticks. The DQN alone scores 0 on critical ticks by
definition; fusion has to win there without losing the easy ticks.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.schema import FusionTrainingConfig
from sentry_ai.decision.fused_controller import HISTORY_LENGTH, NO_BEHAVIOUR
from sentry_ai.decision.mlp_fusion import (
    FUSION_FEATURES,
    FeatureGroup,
    FusionArchitecture,
    FusionNet,
    fusion_features,
)
from sentry_ai.domain.entities import Position
from sentry_ai.interfaces.decision import SceneEvidence
from sentry_ai.interfaces.navigation import (
    LOCAL_ACTION_ORDER,
    ILocalController,
    LocalAction,
    LocalDecision,
    LocalObservation,
)
from sentry_ai.interfaces.sequence import IMotionPredictor, VehicleState
from sentry_ai.sequence.behaviour import heading_to_degrees
from sentry_ai.simulation.engine import SimulationEngine
from sentry_ai.simulation.factory import Mission
from sentry_ai.training.checkpoint import save_checkpoint
from sentry_ai.training.metrics import CsvMetricLogger
from sentry_ai.training.seed import resolve_device, seed_everything

logger = get_logger(__name__)

_ACTION_INDEX = {action: index for index, action in enumerate(LOCAL_ACTION_ORDER)}

#: Builds a fresh mission for a seed, driven by the given controller.
MissionBuilder = Callable[[int, ILocalController], Mission]

#: Given a mission, the onboard camera's evidence for it, called once per tick.
EvidenceFactory = Callable[[Mission], Callable[[], SceneEvidence]]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FusionDataset:
    """Collected ticks, as arrays.

    Attributes:
        features: ``(n, FUSION_FEATURES)`` network inputs.
        labels: ``(n,)`` ground-truth action indices.
        critical: ``(n,)`` whether the stale-map DQN disagreed with the label.
        missions: ``(n,)`` the hazard seed each tick came from.
    """

    features: NDArray[np.float32]
    labels: NDArray[np.int64]
    critical: NDArray[np.bool_]
    missions: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @property
    def dqn_predictions(self) -> NDArray[np.int64]:
        """What the DQN alone chose: the policy block's argmax (its 0.0)."""
        return self.features[:, : len(LOCAL_ACTION_ORDER)].argmax(axis=1).astype(np.int64)

    def save(self, path: Path) -> None:
        """Write to a compressed ``.npz``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            labels=self.labels,
            critical=self.critical,
            missions=self.missions,
        )

    @classmethod
    def load(cls, path: Path) -> FusionDataset:
        """Read what :meth:`save` wrote.

        Raises:
            AssetNotFoundError: If the file does not exist.
        """
        if not path.is_file():
            raise AssetNotFoundError(f"fusion dataset not found: {path} (run train_fusion.py)")
        with np.load(path) as data:
            return cls(
                features=data["features"].astype(np.float32),
                labels=data["labels"].astype(np.int64),
                critical=data["critical"].astype(np.bool_),
                missions=data["missions"].astype(np.int64),
            )


@dataclass
class _Buffer:
    features: list[NDArray[np.float32]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    critical: list[bool] = field(default_factory=list)
    missions: list[int] = field(default_factory=list)

    def to_dataset(self) -> FusionDataset:
        return FusionDataset(
            features=np.stack(self.features).astype(np.float32)
            if self.features
            else np.zeros((0, FUSION_FEATURES), dtype=np.float32),
            labels=np.array(self.labels, dtype=np.int64),
            critical=np.array(self.critical, dtype=np.bool_),
            missions=np.array(self.missions, dtype=np.int64),
        )


class FusionRecorder(ILocalController):
    """Drives a collection mission and records one sample per tick.

    Installed as the engine's controller, then :meth:`bind`-ed to that
    engine — it needs the engine to ask what the DQN would see on the true
    map. Drives the DQN's stale-map choice, or with probability
    ``oracle_probability`` the ground-truth one.
    """

    def __init__(
        self,
        policy: ILocalController,
        predictor: IMotionPredictor | None,
        oracle_probability: float,
        rng: random.Random,
        label_mode: str = "truth",
    ) -> None:
        """Create a recorder that writes into its own buffer.

        ``label_mode`` picks the label (see ``FusionTrainingConfig``):
        ``"truth"`` records the DQN's decision on the true map, ``"veto"``
        records the DQN's own decision, turned into STOP when it would move
        into a tile the real city has blocked.
        """
        self._label_mode = label_mode
        self._policy = policy
        self._predictor = predictor
        self._oracle_probability = oracle_probability
        self._rng = rng
        self._buffer = _Buffer()
        self._engine: SimulationEngine | None = None
        self._evidence: Callable[[], SceneEvidence] = SceneEvidence.empty
        self._history: deque[VehicleState] = deque(maxlen=HISTORY_LENGTH)
        self._mission_seed = 0

    def bind(
        self, engine: SimulationEngine, evidence: Callable[[], SceneEvidence], seed: int
    ) -> None:
        """Attach to a freshly built mission before its first tick."""
        self._engine = engine
        self._evidence = evidence
        self._history.clear()
        self._mission_seed = seed

    def decide(self, observation: LocalObservation) -> LocalDecision:
        """Record this tick's inputs and label, then drive."""
        if self._engine is None:
            raise RuntimeError("FusionRecorder.bind() must be called before the first tick")
        self._history.append(_vehicle_state(observation))
        stale = self._policy.decide(observation)
        truth = self._policy.decide(self._engine.observe(self._engine.physics_grid()))
        behaviour = (
            self._predictor.predict(list(self._history)) if self._predictor else NO_BEHAVIOUR
        )
        label = truth.action if self._label_mode == "truth" else self._veto(observation, stale)
        self._buffer.features.append(fusion_features(self._evidence(), behaviour, stale))
        self._buffer.labels.append(_ACTION_INDEX[label])
        self._buffer.critical.append(label is not stale.action)
        self._buffer.missions.append(self._mission_seed)
        return truth if self._rng.random() < self._oracle_probability else stale

    def _veto(self, observation: LocalObservation, stale: LocalDecision) -> LocalAction:
        """The DQN's action, or STOP if it would drive into something really there."""
        assert self._engine is not None
        dx, dy = observation.heading.delta
        step = {LocalAction.MOVE_FORWARD: 1, LocalAction.REVERSE: -1}.get(stale.action)
        if step is None:
            return stale.action
        x, y = observation.position.x + step * dx, observation.position.y + step * dy
        if x < 0 or y < 0 or not self._engine.physics_grid().is_traversable(Position(x, y)):
            return LocalAction.STOP
        return stale.action

    def dataset(self) -> FusionDataset:
        """Everything recorded so far."""
        return self._buffer.to_dataset()


def collect(
    build: MissionBuilder,
    evidence_for: EvidenceFactory,
    seeds: Iterable[int],
    recorder: FusionRecorder,
    max_ticks: int,
) -> FusionDataset:
    """Run one mission per seed through ``recorder`` and return what it saw."""
    for seed in seeds:
        mission = build(seed, recorder)
        recorder.bind(mission.engine, evidence_for(mission), seed)
        mission.engine.run(max_ticks)
    return recorder.dataset()


def _vehicle_state(observation: LocalObservation) -> VehicleState:
    return VehicleState(
        position=observation.position,
        battery_percent=observation.battery_percent,
        heading_degrees=heading_to_degrees(observation.heading),
    )


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ActionReport:
    """How a chooser of actions did against the ground-truth decision.

    Attributes:
        accuracy: Share of all ticks chosen right.
        weighted_accuracy: Accuracy with critical ticks counted
            ``critical_weight`` times — the quantity training optimises, and
            the one checkpoints are selected on.
        macro_f1: Mean per-action F1 over the actions that occur.
        critical_accuracy: Share of critical ticks chosen right — the ticks
            where the DQN on the stale map was wrong.
        critical_count: How many critical ticks there were.
        false_override_rate: Share of the *non-critical* ticks — where the
            DQN was already right — chosen wrong. What fusion costs.
    """

    accuracy: float
    weighted_accuracy: float
    macro_f1: float
    critical_accuracy: float
    critical_count: int
    false_override_rate: float

    @classmethod
    def score(
        cls,
        labels: NDArray[np.int64],
        predictions: NDArray[np.int64],
        critical: NDArray[np.bool_],
        critical_weight: float = 1.0,
    ) -> ActionReport:
        """Score ``predictions`` against ``labels``.

        Raises:
            ValueError: If the arrays are empty or differ in length.
        """
        if len(labels) == 0 or len(labels) != len(predictions):
            raise ValueError(f"cannot score {len(predictions)} predictions on {len(labels)} labels")
        f1s = []
        for index in range(len(LOCAL_ACTION_ORDER)):
            actual, guessed = labels == index, predictions == index
            if not actual.any():
                continue
            hits = float((actual & guessed).sum())
            precision = hits / guessed.sum() if guessed.any() else 0.0
            recall = hits / actual.sum()
            f1s.append(2 * precision * recall / (precision + recall) if hits else 0.0)
        right = labels == predictions
        count = int(critical.sum())
        weights = np.where(critical, critical_weight, 1.0)
        return cls(
            accuracy=float(right.mean()),
            weighted_accuracy=float((right * weights).sum() / weights.sum()),
            macro_f1=float(np.mean(f1s)),
            critical_accuracy=float(right[critical].mean()) if count else 0.0,
            critical_count=count,
            false_override_rate=float((~right[~critical]).mean()) if (~critical).any() else 0.0,
        )


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FusionTrainingOutcome:
    """What one training run produced."""

    groups: tuple[FeatureGroup, ...]
    report: ActionReport
    weights_path: Path
    epochs_run: int
    best_epoch: int
    parameters: int


class FusionTrainer:
    """Trains a :class:`FusionNet` on collected ticks with Adam."""

    def __init__(self, config: FusionTrainingConfig) -> None:
        """Create a trainer for ``config``'s hyperparameters."""
        self._config = config

    def resolve_device(self) -> str:
        """The device training actually runs on."""
        return resolve_device(self._config.device)

    def train(
        self,
        train_set: FusionDataset,
        val_set: FusionDataset,
        groups: Sequence[FeatureGroup],
        run_name: str,
        on_epoch: Callable[[dict[str, float]], None] | None = None,
    ) -> FusionTrainingOutcome:
        """Train on ``groups`` only; keep the epoch with the best weighted accuracy.

        Selection is on validation accuracy with critical ticks weighted as
        in the loss. Plain accuracy is dominated by easy ticks where the
        DQN is already right; critical accuracy alone rewards a network
        that swerves at everything.
        """
        config = self._config
        seed_everything(config.seed)
        device = torch.device(self.resolve_device())
        architecture = FusionArchitecture(config.hidden_sizes, config.dropout, tuple(groups))
        network = FusionNet(architecture).to(device)
        weights = config.runs_dir / run_name / "best.pt"
        epochs_run, best_epoch = self._fit(network, train_set, val_set, weights, on_epoch)
        final = FusionNet.from_checkpoint(
            torch.load(weights, map_location="cpu", weights_only=True)
        )
        return FusionTrainingOutcome(
            groups=tuple(groups),
            report=evaluate_network(final.to(device), val_set, device, config.critical_weight),
            weights_path=weights,
            epochs_run=epochs_run,
            best_epoch=best_epoch,
            parameters=network.parameter_count(),
        )

    def _fit(
        self,
        network: FusionNet,
        train_set: FusionDataset,
        val_set: FusionDataset,
        weights: Path,
        on_epoch: Callable[[dict[str, float]], None] | None,
    ) -> tuple[int, int]:
        """The epoch loop with early stopping. Returns (epochs run, best epoch)."""
        config = self._config
        device = next(network.parameters()).device
        optimiser = torch.optim.Adam(
            network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        metrics = CsvMetricLogger(weights.parent / "metrics.csv")
        batches = _batches(train_set, config, device)
        best, best_epoch, stale_epochs, epoch = -1.0, 0, 0, 0
        for epoch in range(1, config.epochs + 1):
            loss = _train_epoch(network, optimiser, batches)
            report = evaluate_network(network, val_set, device, config.critical_weight)
            row = _epoch_row(epoch, loss, report)
            metrics.log(row)
            if on_epoch is not None:
                on_epoch(row)
            if report.weighted_accuracy > best:
                best, best_epoch, stale_epochs = report.weighted_accuracy, epoch, 0
                save_checkpoint(weights, network.to_checkpoint({"epoch": epoch}), config, row)
                continue
            stale_epochs += 1
            if config.patience and stale_epochs >= config.patience:
                break
        return epoch, best_epoch


def evaluate_network(
    network: FusionNet, dataset: FusionDataset, device: torch.device, critical_weight: float = 1.0
) -> ActionReport:
    """Score ``network``'s greedy choices on ``dataset``."""
    network.eval()
    with torch.inference_mode():
        logits = network(torch.from_numpy(dataset.features).to(device))
        predictions = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
    return ActionReport.score(dataset.labels, predictions, dataset.critical, critical_weight)


def _batches(
    dataset: FusionDataset, config: FusionTrainingConfig, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Shuffled mini-batches of (features, labels, per-sample weights)."""
    weights = np.where(dataset.critical, config.critical_weight, 1.0).astype(np.float32)
    order = np.random.default_rng(config.seed).permutation(len(dataset))
    batches = []
    for start in range(0, len(order), config.batch_size):
        index = order[start : start + config.batch_size]
        batches.append(
            (
                torch.from_numpy(dataset.features[index]).to(device),
                torch.from_numpy(dataset.labels[index]).to(device),
                torch.from_numpy(weights[index]).to(device),
            )
        )
    return batches


def _train_epoch(
    network: FusionNet,
    optimiser: torch.optim.Optimizer,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    """One pass of weighted cross-entropy; returns the mean loss."""
    network.train()
    criterion = nn.CrossEntropyLoss(reduction="none")
    total, seen = 0.0, 0
    for features, labels, weights in random.sample(batches, len(batches)):
        optimiser.zero_grad()
        loss = (criterion(network(features), labels) * weights).sum() / weights.sum()
        loss.backward()
        optimiser.step()
        total += loss.item() * len(labels)
        seen += len(labels)
    return total / max(1, seen)


def _epoch_row(epoch: int, loss: float, report: ActionReport) -> dict[str, float]:
    return {
        "epoch": float(epoch),
        "train_loss": loss,
        "val_accuracy": report.accuracy,
        "val_weighted_accuracy": report.weighted_accuracy,
        "val_macro_f1": report.macro_f1,
        "val_critical_accuracy": report.critical_accuracy,
    }


def action_at(index: int) -> LocalAction:
    """The action a label index stands for."""
    return LOCAL_ACTION_ORDER[index]
