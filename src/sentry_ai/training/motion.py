"""Trains and scores the behaviour LSTM against the baselines it must beat (Unit III).

**Every window is labelled by one rule.** A window is ``window`` consecutive
states; its label is :func:`~sentry_ai.sequence.behaviour.classify_motion`
applied to its last state and the state ``horizon`` ticks later. The
persistence baseline uses the same rule looking *backwards*, so the model
and the baseline answer exactly the same question.

**The headline metric is macro-F1, not accuracy.** About seven windows in
ten are ADVANCE, so "always advance" scores ~0.71 accuracy while never once
predicting a turn — and turns are the part worth predicting. Macro-F1
averages the per-class F1 of every class that actually occurs, so a class
the model never predicts costs it a full share. Accuracy is still reported,
because the model must not buy its macro-F1 by getting worse at the common
case.

**Two baselines**, both scored on the same validation windows:

* *majority* — always the training split's most common class;
* *persistence* — whatever the vehicle did over the last ``horizon`` ticks.

Milestone M5 is met when the LSTM beats both on macro-F1.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.schema import LstmTrainingConfig
from sentry_ai.interfaces.sequence import BehaviourClass, IMotionPredictor, VehicleState
from sentry_ai.sequence.behaviour import BEHAVIOUR_ORDER, classify_motion, persistence_baseline
from sentry_ai.sequence.lstm_predictor import BehaviourLstmNet, LstmArchitecture, encode_states
from sentry_ai.training.checkpoint import save_checkpoint
from sentry_ai.training.metrics import CsvMetricLogger
from sentry_ai.training.seed import resolve_device, seed_everything
from sentry_ai.training.trajectories import Trajectory, TrajectorySet

logger = get_logger(__name__)

#: Called after every epoch with that epoch's metric row.
EpochCallback = Callable[[dict[str, float]], None]


@dataclass(frozen=True)
class LabelledWindow:
    """``window`` consecutive states and what the vehicle did next."""

    states: tuple[VehicleState, ...]
    label: BehaviourClass


def labelled_windows(
    trajectories: Iterable[Trajectory], window: int, horizon: int, cone_degrees: float
) -> list[LabelledWindow]:
    """Every full window in every trajectory that has ``horizon`` ticks after it.

    Windows never straddle two trajectories, and none is padded: padding is
    for the first ticks of a live mission, not for training data.
    """
    windows: list[LabelledWindow] = []
    for trajectory in trajectories:
        states = trajectory.states
        for last in range(window - 1, len(states) - horizon):
            windows.append(
                LabelledWindow(
                    states=states[last - window + 1 : last + 1],
                    label=classify_motion(states[last], states[last + horizon], cone_degrees),
                )
            )
    return windows


@dataclass(frozen=True)
class ClassScore:
    """Precision, recall and F1 for one behaviour class."""

    behaviour: BehaviourClass
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationReport:
    """How a predictor did on a set of labelled windows.

    Attributes:
        confusion: ``confusion[true][predicted]`` counts, indexed in
            :data:`~sentry_ai.sequence.behaviour.BEHAVIOUR_ORDER`.
    """

    confusion: tuple[tuple[int, ...], ...]

    @classmethod
    def from_predictions(
        cls, truth: Sequence[BehaviourClass], predicted: Sequence[BehaviourClass]
    ) -> ClassificationReport:
        """Tally predictions against the truth.

        Raises:
            ValueError: If the sequences differ in length or are empty.
        """
        if len(truth) != len(predicted):
            raise ValueError(f"{len(truth)} labels but {len(predicted)} predictions")
        if not truth:
            raise ValueError("cannot score zero predictions")
        index = {behaviour: position for position, behaviour in enumerate(BEHAVIOUR_ORDER)}
        matrix = [[0] * len(BEHAVIOUR_ORDER) for _ in BEHAVIOUR_ORDER]
        for actual, guess in zip(truth, predicted, strict=True):
            matrix[index[actual]][index[guess]] += 1
        return cls(confusion=tuple(tuple(row) for row in matrix))

    @property
    def total(self) -> int:
        """Windows scored."""
        return sum(sum(row) for row in self.confusion)

    @property
    def accuracy(self) -> float:
        """Share of windows predicted exactly right."""
        correct = sum(self.confusion[i][i] for i in range(len(BEHAVIOUR_ORDER)))
        return correct / self.total

    @property
    def per_class(self) -> tuple[ClassScore, ...]:
        """Scores for every class, in :data:`BEHAVIOUR_ORDER`."""
        scores = []
        for i, behaviour in enumerate(BEHAVIOUR_ORDER):
            true_positive = self.confusion[i][i]
            support = sum(self.confusion[i])
            predicted = sum(row[i] for row in self.confusion)
            precision = true_positive / predicted if predicted else 0.0
            recall = true_positive / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            scores.append(ClassScore(behaviour, precision, recall, f1, support))
        return tuple(scores)

    @property
    def macro_f1(self) -> float:
        """Mean F1 over the classes that actually occur.

        A class with no true instances is left out rather than scored
        zero: it would drag every predictor down equally and say nothing.
        """
        present = [score.f1 for score in self.per_class if score.support]
        return sum(present) / len(present)

    def as_display_rows(self) -> list[tuple[str, str]]:
        """Label/value pairs for direct printing."""
        rows = [("accuracy", f"{self.accuracy:.4f}"), ("macro-F1", f"{self.macro_f1:.4f}")]
        for score in self.per_class:
            if score.support:
                rows.append(
                    (
                        f"  {score.behaviour.value}",
                        f"P {score.precision:.3f}  R {score.recall:.3f}  "
                        f"F1 {score.f1:.3f}  (n={score.support})",
                    )
                )
        return rows


def score_persistence(
    windows: Sequence[LabelledWindow], horizon: int, cone_degrees: float
) -> ClassificationReport:
    """Score "the vehicle keeps doing what it just did"."""
    return ClassificationReport.from_predictions(
        [window.label for window in windows],
        [persistence_baseline(window.states, horizon, cone_degrees) for window in windows],
    )


def score_majority(
    windows: Sequence[LabelledWindow], majority: BehaviourClass
) -> ClassificationReport:
    """Score "always predict ``majority``"."""
    return ClassificationReport.from_predictions(
        [window.label for window in windows], [majority] * len(windows)
    )


def score_predictor(
    predictor: IMotionPredictor, windows: Sequence[LabelledWindow]
) -> ClassificationReport:
    """Score any :class:`IMotionPredictor` through its public port, one window at a time."""
    return ClassificationReport.from_predictions(
        [window.label for window in windows],
        [predictor.predict(window.states).predicted_class for window in windows],
    )


def novel_windows(
    windows: Sequence[LabelledWindow], seen: Iterable[LabelledWindow]
) -> list[LabelledWindow]:
    """The windows whose exact state sequence appears nowhere in ``seen``.

    The honest test of generalisation on a fixed map. Most held-out windows
    repeat a training window verbatim — the city has a handful of streets,
    and every mission drives some of them — so a score over all of them
    mixes recall with prediction. Scored on these alone, a model is being
    asked about situations it has genuinely never been shown.
    """
    known = {_fingerprint(window) for window in seen}
    return [window for window in windows if _fingerprint(window) not in known]


def majority_class(windows: Iterable[LabelledWindow]) -> BehaviourClass:
    """The most common label, ties broken by :data:`BEHAVIOUR_ORDER`.

    Raises:
        ValueError: If there are no windows.
    """
    counts = Counter(window.label for window in windows)
    if not counts:
        raise ValueError("no windows to take a majority over")
    return max(BEHAVIOUR_ORDER, key=lambda behaviour: counts[behaviour])


@dataclass(frozen=True)
class LstmTrainingOutcome:
    """What a finished run produced, next to the baselines it had to beat.

    Attributes:
        weights_path: The best checkpoint (highest validation macro-F1).
        run_dir: Holds ``best.pt``, ``last.pt``, metadata and ``metrics.csv``.
        epochs_run: Epochs actually trained.
        best_epoch: The epoch ``weights_path`` was saved at.
        report: The best checkpoint's validation report.
        baselines: Validation reports for ``"majority"`` and ``"persistence"``.
        train_windows: Training windows seen per epoch.
        parameters: Trainable parameter count.
    """

    weights_path: Path
    run_dir: Path
    epochs_run: int
    best_epoch: int
    report: ClassificationReport
    baselines: dict[str, ClassificationReport]
    train_windows: int
    parameters: int

    @property
    def beats_baselines(self) -> bool:
        """Milestone M5: higher validation macro-F1 than every baseline."""
        return all(self.report.macro_f1 > base.macro_f1 for base in self.baselines.values())


class LstmTrainer:
    """Runs one training job over recorded trajectories."""

    def __init__(self, config: LstmTrainingConfig) -> None:
        """Create a trainer from validated hyperparameters."""
        self._config = config

    def resolve_device(self) -> str:
        """The device to train on. Public so a script can report it up front."""
        return resolve_device(self._config.device)

    def train(
        self, run_name: str = "sentry", on_epoch: EpochCallback | None = None
    ) -> LstmTrainingOutcome:
        """Train, keep the checkpoint with the best validation macro-F1, report baselines.

        Raises:
            AssetNotFoundError: If the trajectory files are missing.
            ValueError: If train and validation were recorded on different maps.
        """
        config = self._config
        train_set = TrajectorySet.load(config.trajectories_dir / "train.json")
        val_set = TrajectorySet.load(config.trajectories_dir / "val.json")
        if (train_set.grid_width, train_set.grid_height) != (
            val_set.grid_width,
            val_set.grid_height,
        ):
            raise ValueError("train and val trajectories were recorded on different map sizes")

        rule = (config.window, config.horizon, config.cone_degrees)
        train_windows = labelled_windows(train_set.trajectories, *rule)
        val_windows = labelled_windows(val_set.trajectories, *rule)
        if not train_windows or not val_windows:
            raise AssetNotFoundError(
                f"No windows of {config.window}+{config.horizon} ticks in "
                f"{config.trajectories_dir}; record longer or more missions."
            )

        majority = majority_class(train_windows)
        baselines = {
            "majority": score_majority(val_windows, majority),
            "persistence": score_persistence(val_windows, config.horizon, config.cone_degrees),
        }

        seed_everything(config.seed)
        device = torch.device(self.resolve_device())
        architecture = LstmArchitecture(
            window=config.window,
            grid_width=train_set.grid_width,
            grid_height=train_set.grid_height,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            horizon=config.horizon,
            cone_degrees=config.cone_degrees,
        )
        network = BehaviourLstmNet(architecture).to(device)
        optimizer = torch.optim.Adam(
            network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        train_x, train_y = _tensors(train_windows, architecture)
        val_x, val_y = _tensors(val_windows, architecture)
        loss_fn = nn.CrossEntropyLoss(
            weight=_class_weights(train_y).to(device) if config.class_weighting else None
        )
        loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
        )
        logger.info(
            "Training behaviour LSTM: %d parameters, %d train / %d val windows, "
            "majority macro-F1 %.4f, persistence macro-F1 %.4f, on %s",
            network.parameter_count(),
            len(train_windows),
            len(val_windows),
            baselines["majority"].macro_f1,
            baselines["persistence"].macro_f1,
            device,
        )

        run_dir = config.runs_dir / run_name
        metrics = CsvMetricLogger(run_dir / "metrics.csv")
        best_epoch, best_score, best_report = 0, -math.inf, baselines["majority"]
        epoch = 0
        for epoch in range(1, config.epochs + 1):
            train_loss = _train_epoch(network, loader, optimizer, loss_fn, device)
            val_loss, predictions = _validate(network, val_x, val_y, loss_fn, device)
            report = ClassificationReport.from_predictions(
                [window.label for window in val_windows], predictions
            )
            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": report.accuracy,
                "val_macro_f1": report.macro_f1,
            }
            metrics.log(row)
            if on_epoch is not None:
                on_epoch(row)

            scores = {"val_macro_f1": report.macro_f1, "val_accuracy": report.accuracy}
            payload = network.to_checkpoint({"epoch": epoch, **scores})
            save_checkpoint(run_dir / "last.pt", payload, config, scores)
            if report.macro_f1 > best_score:
                best_epoch, best_score, best_report = epoch, report.macro_f1, report
                save_checkpoint(run_dir / "best.pt", payload, config, scores)
            elif config.patience and epoch - best_epoch >= config.patience:
                logger.info("No improvement for %d epochs — stopping", config.patience)
                break

        return LstmTrainingOutcome(
            weights_path=run_dir / "best.pt",
            run_dir=run_dir,
            epochs_run=epoch,
            best_epoch=best_epoch,
            report=best_report,
            baselines=baselines,
            train_windows=len(train_windows),
            parameters=network.parameter_count(),
        )


def _fingerprint(window: LabelledWindow) -> tuple[tuple[int, int, float], ...]:
    """What makes two windows the same situation: positions and headings, in order."""
    return tuple(
        (state.position.x, state.position.y, state.heading_degrees) for state in window.states
    )


def _tensors(
    windows: Sequence[LabelledWindow], architecture: LstmArchitecture
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode windows into ``(n, window, 7)`` features and ``(n,)`` class indices."""
    index = {behaviour: position for position, behaviour in enumerate(BEHAVIOUR_ORDER)}
    features: NDArray[np.float32] = np.stack(
        [
            encode_states(window.states, architecture.grid_width, architecture.grid_height)
            for window in windows
        ]
    )
    labels = np.array([index[window.label] for window in windows], dtype=np.int64)
    return torch.from_numpy(features), torch.from_numpy(labels)


def sqrt_inverse_frequency_weights(labels: torch.Tensor, classes: int) -> torch.Tensor:
    """Square-root inverse-frequency class weights; see :func:`_class_weights`."""
    counts = torch.bincount(labels, minlength=classes).float()
    present = counts > 0
    weights = torch.zeros_like(counts)
    weights[present] = torch.sqrt(counts.sum() / (present.sum() * counts[present]))
    return weights


def _class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Square-root inverse-frequency weights: rare classes count for more, not for everything.

    Plain inverse frequency was tried first and failed in a specific way.
    HOLD is about one training window in a thousand, which made each HOLD
    window weigh as much as five hundred ADVANCE windows (212 against 0.4).
    The model learned that "hold" was worth guessing whenever in doubt:
    86% HOLD recall at 6% precision, about a hundred false alarms. The
    square root keeps the ordering — rarer still weighs more — while
    capping how far one tiny class can bend the loss (HOLD ~15, not 212).

    A class absent from training gets weight zero — there is nothing to
    weight, and an infinite weight would be a NaN waiting to happen.
    """
    return sqrt_inverse_frequency_weights(labels, len(BEHAVIOUR_ORDER))


def _train_epoch(
    network: BehaviourLstmNet,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """One pass over the training windows; returns the mean loss per window."""
    network.train()
    total, count = 0.0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(network(features), labels)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * labels.shape[0]
        count += labels.shape[0]
    return total / count


def _validate(
    network: BehaviourLstmNet,
    features: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, list[BehaviourClass]]:
    """Validation loss and every window's predicted class, in one batch."""
    network.eval()
    with torch.inference_mode():
        logits = network(features.to(device))
        loss = float(loss_fn(logits, labels.to(device)).item())
        predicted = logits.argmax(dim=-1).cpu().tolist()
    return loss, [BEHAVIOUR_ORDER[index] for index in predicted]
