"""Records, trains and scores decision fusion (Unit I, milestone M7).

**The label is the DQN's own move, made safe.** Every tick the DQN proposes
an action on the command center's (lagging) map. The label is that action,
unless it would drive the vehicle into a tile that is *really* impassable —
debris or fire the map has not caught up with — in which case it is STOP.
Fusion therefore learns one thing: keep the DQN's driving, veto the moves
the camera shows are unsafe. Stopping is the conservative veto: the map
catches up within ``belief_lag`` refreshes, the command center then routes
around the obstacle (``MissionController`` re-checks every route against
every fresh map), and the vehicle moves on.

**Recorded with the DQN driving and fusion passing it through.** A recording
run is the full pipeline with the fusion slot filled by a recorder that
returns the DQN's action unchanged — so the DQN really does drive into the
debris, and those are the moments the dataset needs.

**Scored three ways, all on held-out missions:**

1. *Per decision*, for the MLP and for the same MLP restricted to one signal
   group (DQN only, LSTM only, camera only). M7 is met when fusion's
   macro-F1 beats every single-signal model.
2. *Per decision*, for two untrained references: the DQN as-is, and
   :class:`~sentry_ai.decision.fusion.CameraVetoFusion`, the obvious
   hand-written rule. A network that cannot beat a two-line rule has only
   earned its place as a Unit I exercise, and the report says so.
3. *Per mission*: rescues, losses and collisions with each driving the
   vehicle on a lagging map. Decision accuracy is a proxy; crashes avoided
   is the point.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import FusionTrainingConfig
from sentry_ai.decision.fused_controller import FusedLocalController
from sentry_ai.decision.fusion import (
    FUSION_FEATURES,
    FusionArchitecture,
    FusionNet,
    fusion_features,
)
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.interfaces.decision import FinalAction, IDecisionFusion
from sentry_ai.interfaces.navigation import (
    LOCAL_ACTION_ORDER,
    ILocalController,
    LocalAction,
    LocalDecision,
    LocalObservation,
)
from sentry_ai.interfaces.perception import WorldDetection
from sentry_ai.interfaces.sequence import BehaviourSignal, IMotionPredictor
from sentry_ai.perception.onboard import OnboardSensing
from sentry_ai.simulation.grid_source import GroundTruthGridSource, LaggedGridSource
from sentry_ai.training.checkpoint import save_checkpoint
from sentry_ai.training.metrics import CsvMetricLogger
from sentry_ai.training.missions import MissionFactory
from sentry_ai.training.motion import sqrt_inverse_frequency_weights
from sentry_ai.training.seed import resolve_device, seed_everything

logger = get_logger(__name__)

#: The models M7 compares: fusion over everything, and each signal alone.
ABLATIONS: dict[str, tuple[str, ...]] = {
    "fusion": ("dqn", "lstm", "camera", "observation"),
    "dqn only": ("dqn",),
    "lstm only": ("lstm",),
    "camera only": ("camera",),
}

_STOP = LOCAL_ACTION_ORDER.index(LocalAction.STOP)


def safe_action(
    action: LocalAction, position: Position, heading: Heading, truth: OccupancyGrid
) -> LocalAction:
    """``action``, or STOP if it would move the vehicle into a truly impassable tile."""
    dx, dy = heading.delta
    if action is LocalAction.REVERSE:
        dx, dy = -dx, -dy
    elif action is not LocalAction.MOVE_FORWARD:
        return action
    x, y = position.x + dx, position.y + dy
    if x < 0 or y < 0 or not truth.is_traversable(Position(x, y)):
        return LocalAction.STOP
    return action


@dataclass(frozen=True)
class FusionSamples:
    """Recorded fusion inputs with their labels.

    Attributes:
        features: ``(n, FUSION_FEATURES)`` fusion inputs.
        labels: ``(n,)`` indices into ``LOCAL_ACTION_ORDER`` — the safe action.
        proposed: ``(n,)`` the DQN's own action, same indexing.
        seeds: ``(n,)`` the mission each sample came from.
    """

    features: NDArray[np.float32]
    labels: NDArray[np.int64]
    proposed: NDArray[np.int64]
    seeds: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @property
    def vetoes(self) -> int:
        """Samples where the DQN's move was unsafe."""
        return int((self.labels != self.proposed).sum())

    def save(self, path: Path) -> None:
        """Write the samples as one ``.npz``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            labels=self.labels,
            proposed=self.proposed,
            seeds=self.seeds,
        )

    @classmethod
    def load(cls, path: Path) -> FusionSamples:
        """Read samples written by :meth:`save`.

        Raises:
            AssetNotFoundError: If the file does not exist.
        """
        if not path.is_file():
            raise AssetNotFoundError(
                f"Fusion samples not found: {path}. Run scripts/train_fusion.py --record first."
            )
        with np.load(path) as data:
            return cls(
                features=data["features"].astype(np.float32),
                labels=data["labels"].astype(np.int64),
                proposed=data["proposed"].astype(np.int64),
                seeds=data["seeds"].astype(np.int64),
            )


class RecordingFusion(IDecisionFusion):
    """Fills the fusion slot during recording: logs a labelled sample, passes the DQN through."""

    def __init__(self) -> None:
        self._city_map: CityMap | None = None
        self._seed = 0
        self.features: list[NDArray[np.float32]] = []
        self.labels: list[int] = []
        self.proposed: list[int] = []
        self.seeds: list[int] = []

    def attach(self, city_map: CityMap, seed: int) -> None:
        """Label against this city's real state from now on."""
        self._city_map = city_map
        self._seed = seed

    def fuse(
        self,
        observation: LocalObservation,
        sightings: Sequence[WorldDetection],
        behaviour: BehaviourSignal,
        local_decision: LocalDecision,
    ) -> FinalAction:
        """Record the inputs and the safe action; return the DQN's action unchanged."""
        if self._city_map is None:
            raise RuntimeError("attach() a city before recording")
        truth = OccupancyGrid.from_city_map(self._city_map)
        label = safe_action(local_decision.action, observation.position, observation.heading, truth)
        self.features.append(fusion_features(observation, sightings, behaviour, local_decision))
        self.labels.append(LOCAL_ACTION_ORDER.index(label))
        self.proposed.append(LOCAL_ACTION_ORDER.index(local_decision.action))
        self.seeds.append(self._seed)
        return FinalAction(action=local_decision.action, rationale_score=1.0)

    def samples(self) -> FusionSamples:
        """Everything recorded so far."""
        return FusionSamples(
            features=(
                np.stack(self.features)
                if self.features
                else np.zeros((0, FUSION_FEATURES), np.float32)
            ),
            labels=np.array(self.labels, dtype=np.int64),
            proposed=np.array(self.proposed, dtype=np.int64),
            seeds=np.array(self.seeds, dtype=np.int64),
        )


@dataclass(frozen=True)
class Pipeline:
    """The trained parts a fused controller is assembled from, for one mission at a time."""

    local: ILocalController
    predictor: IMotionPredictor
    sensing_factory: Callable[[], OnboardSensing]


def record_samples(
    factory: MissionFactory,
    pipeline: Pipeline,
    seeds: Sequence[int],
    belief_lag: int,
    max_ticks: int,
) -> FusionSamples:
    """Drive each mission with the DQN on a lagging map; record every tick."""
    recorder = RecordingFusion()
    for index, seed in enumerate(seeds, start=1):
        sensing = pipeline.sensing_factory()
        controller = FusedLocalController(
            pipeline.local, pipeline.predictor, recorder, sensing.sightings
        )
        mission = factory.build(
            seed, controller, LaggedGridSource(GroundTruthGridSource(), belief_lag)
        )
        sensing.attach(mission.city_map)
        recorder.attach(mission.city_map, seed)
        mission.engine.run(max_ticks=max_ticks)
        if index % 10 == 0:
            logger.info(
                "recorded %d/%d missions, %d samples", index, len(seeds), len(recorder.labels)
            )
    return recorder.samples()


@dataclass(frozen=True)
class DecisionReport:
    """How a set of decisions compared with the safe labels.

    Attributes:
        confusion: ``confusion[true][predicted]`` over ``LOCAL_ACTION_ORDER``.
        veto_recall: Of the ticks where the DQN's move was unsafe, the share
            where the decision was *not* that unsafe move — crashes avoided.
        false_veto_rate: Of the ticks where the DQN's move was safe, the
            share where the decision overrode it anyway — needless stops.
    """

    confusion: tuple[tuple[int, ...], ...]
    veto_recall: float
    false_veto_rate: float

    @classmethod
    def score(
        cls,
        labels: NDArray[np.int64],
        predicted: NDArray[np.int64],
        proposed: NDArray[np.int64],
    ) -> DecisionReport:
        """Compare ``predicted`` with ``labels``, given what the DQN ``proposed``."""
        size = len(LOCAL_ACTION_ORDER)
        confusion = np.zeros((size, size), dtype=np.int64)
        np.add.at(confusion, (labels, predicted), 1)
        unsafe = labels != proposed
        safe = ~unsafe
        veto_recall = (
            float((predicted[unsafe] != proposed[unsafe]).mean()) if unsafe.any() else math.nan
        )
        false_veto = (
            float((predicted[safe] != proposed[safe]).mean()) if safe.any() else math.nan
        )
        return cls(
            confusion=tuple(tuple(int(v) for v in row) for row in confusion),
            veto_recall=veto_recall,
            false_veto_rate=false_veto,
        )

    @property
    def accuracy(self) -> float:
        """Share of decisions equal to the safe label."""
        matrix = np.array(self.confusion)
        return float(np.trace(matrix) / matrix.sum())

    @property
    def macro_f1(self) -> float:
        """Mean F1 over the actions that occur in the labels."""
        matrix = np.array(self.confusion, dtype=np.float64)
        scores = []
        for i in range(matrix.shape[0]):
            support = matrix[i].sum()
            if support == 0:
                continue
            predicted = matrix[:, i].sum()
            precision = matrix[i, i] / predicted if predicted else 0.0
            recall = matrix[i, i] / support
            harmonic = precision + recall
            scores.append(2 * precision * recall / harmonic if harmonic else 0.0)
        return float(np.mean(scores))


@dataclass(frozen=True)
class FusionModelOutcome:
    """One trained model: where it is and how it scored on validation."""

    name: str
    weights_path: Path
    report: DecisionReport
    best_epoch: int


def train_model(
    name: str,
    architecture: FusionArchitecture,
    train: FusionSamples,
    val: FusionSamples,
    config: FusionTrainingConfig,
    run_dir: Path,
) -> FusionModelOutcome:
    """Train one fusion network; keep the epoch with the best validation macro-F1."""
    seed_everything(config.seed)
    device = torch.device(resolve_device(config.device))
    network = FusionNet(architecture).to(device)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    labels = torch.from_numpy(train.labels)
    loss_fn = nn.CrossEntropyLoss(
        weight=sqrt_inverse_frequency_weights(labels, len(LOCAL_ACTION_ORDER)).to(device)
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train.features), labels),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_features = torch.from_numpy(val.features).to(device)
    metrics = CsvMetricLogger(run_dir / "metrics.csv")
    best_epoch, best_score, best_report = 0, -math.inf, None
    for epoch in range(1, config.epochs + 1):
        network.train()
        total, count = 0.0, 0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(network(features), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * targets.shape[0]
            count += targets.shape[0]
        network.eval()
        with torch.inference_mode():
            predicted = network(val_features).argmax(dim=-1).cpu().numpy().astype(np.int64)
        report = DecisionReport.score(val.labels, predicted, val.proposed)
        row = {
            "epoch": float(epoch),
            "train_loss": total / count,
            "val_accuracy": report.accuracy,
            "val_macro_f1": report.macro_f1,
            "val_veto_recall": report.veto_recall,
            "val_false_veto_rate": report.false_veto_rate,
        }
        metrics.log(row)
        if report.macro_f1 > best_score:
            best_epoch, best_score, best_report = epoch, report.macro_f1, report
            save_checkpoint(
                run_dir / "best.pt",
                network.to_checkpoint({"epoch": epoch, "name": name}),
                config,
                {k: v for k, v in row.items() if math.isfinite(v)},
            )
        elif config.patience and epoch - best_epoch >= config.patience:
            break
    assert best_report is not None  # epochs >= 1, so at least one report exists
    logger.info("%s: best epoch %d, macro-F1 %.4f", name, best_epoch, best_score)
    return FusionModelOutcome(name, run_dir / "best.pt", best_report, best_epoch)


def score_reference(
    samples: FusionSamples, fusion: IDecisionFusion | None
) -> DecisionReport:
    """Score an untrained reference: the DQN as-is (``None``), or a rule.

    Rules are applied to the recorded feature vectors' camera and DQN parts
    through a reconstructed decision — no simulation needed.
    """
    if fusion is None:
        return DecisionReport.score(samples.labels, samples.proposed, samples.proposed)
    from sentry_ai.decision.fusion import CameraVetoFusion  # noqa: PLC0415

    if not isinstance(fusion, CameraVetoFusion):
        raise TypeError("only the DQN and CameraVetoFusion are scored from recorded features")
    predicted = _camera_veto_from_features(samples, fusion)
    return DecisionReport.score(samples.labels, predicted, samples.proposed)


def _camera_veto_from_features(samples: FusionSamples, rule: object) -> NDArray[np.int64]:
    """Apply the camera-veto rule to recorded features directly.

    Equivalent to running :class:`CameraVetoFusion` live: the camera block of
    the feature vector is exactly what it reads, and the proposed action is
    recorded alongside.
    """
    from sentry_ai.decision.fusion import CAMERA_TILES, FEATURE_GROUPS  # noqa: PLC0415
    from sentry_ai.domain.enums import EntityKind  # noqa: PLC0415
    from sentry_ai.sensors.frame import YOLO_CLASSES  # noqa: PLC0415

    threshold = getattr(rule, "_threshold", 0.5)
    offset = 0
    for group, width in FEATURE_GROUPS:
        if group == "camera":
            break
        offset += width
    camera = samples.features[:, offset : offset + len(CAMERA_TILES) * len(YOLO_CLASSES)]
    camera = camera.reshape(len(samples), len(CAMERA_TILES), len(YOLO_CLASSES))
    rows = {name: index for index, (name, _) in enumerate(CAMERA_TILES)}
    hazards = [YOLO_CLASSES.index(EntityKind.FIRE), YOLO_CLASSES.index(EntityKind.OBSTACLE)]
    danger_ahead = camera[:, rows["ahead"], hazards].max(axis=1) >= threshold
    danger_behind = camera[:, rows["behind"], hazards].max(axis=1) >= threshold
    forward = samples.proposed == LOCAL_ACTION_ORDER.index(LocalAction.MOVE_FORWARD)
    reverse = samples.proposed == LOCAL_ACTION_ORDER.index(LocalAction.REVERSE)
    veto = (forward & danger_ahead) | (reverse & danger_behind)
    return np.where(veto, _STOP, samples.proposed).astype(np.int64)


def assemble_pipeline(
    loader: ConfigLoader,
    sensor_config_path: Path,
    dqn_weights: Path,
    lstm_weights: Path,
    detector_weights: Path,
    device: str = "cpu",
    seed: int = 0,
) -> Pipeline:
    """Load the trained DQN, LSTM and onboard detector into a :class:`Pipeline`.

    The onboard camera's frames are degraded with the shipped corruption, as
    the CCTV frames are, and read by the given detector. Each call of the
    sensing factory makes a fresh sensing chain with its own seeded
    degrader, so every mission's smoke is reproducible.
    """
    from sentry_ai.decision.dqn_controller import DqnLocalController  # noqa: PLC0415
    from sentry_ai.perception.grid_source import ModelObserver  # noqa: PLC0415
    from sentry_ai.perception.yolo_detector import YoloDetector  # noqa: PLC0415
    from sentry_ai.sensors.degradation import FrameDegrader  # noqa: PLC0415
    from sentry_ai.sensors.palette import SensorPalette  # noqa: PLC0415
    from sentry_ai.sensors.rig import SensorRig  # noqa: PLC0415
    from sentry_ai.sequence.lstm_predictor import LstmMotionPredictor  # noqa: PLC0415

    sensor_config = loader.load_sensor_config(sensor_config_path)
    palette = SensorPalette.from_config(loader, sensor_config_path)
    rig = SensorRig.from_config(sensor_config, palette)
    observer = ModelObserver(YoloDetector(weights_path=detector_weights, device=device))
    counter = itertools.count(seed)

    def sensing() -> OnboardSensing:
        return OnboardSensing(
            rig=rig,
            observer=observer,
            degrader=FrameDegrader(sensor_config.degradation, np.random.default_rng(next(counter))),
        )

    return Pipeline(
        local=DqnLocalController.from_file(dqn_weights, device=device),
        predictor=LstmMotionPredictor.from_checkpoint(lstm_weights, device=device),
        sensing_factory=sensing,
    )
