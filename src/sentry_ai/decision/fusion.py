"""MLP decision fusion implementing :class:`IDecisionFusion` (Unit I).

The last stage before the vehicle acts. It reads four things and picks the
move:

========================  ==================================================
**local controller**      the DQN's Q-values, as a probability over actions
**sequence model**        the LSTM's predicted behaviour, one-hot x confidence
**onboard camera**        what the vehicle's own camera sees on the tiles
                          around it right now, per class, egocentric
**observation**           the same egocentric view the DQN acts on
========================  ==================================================

**Why it exists.** The DQN drives on the command center's map. When that map
lags reality (``LaggedGridSource``) the DQN confidently drives into debris
that fell after the map was made. The onboard camera sees the debris; the
DQN cannot. Fusion's job is to combine the two — trust the DQN about *where
to go*, trust the camera about *what is in the way*.

**The camera, egocentrically.** A sighting is a set of map tiles. For five
tiles around the vehicle — one and two ahead, behind, left, right — and each
of the three detector classes, the feature is the highest confidence of any
sighting covering that tile. "Ahead" is relative to the vehicle's heading,
so a fire in front reads the same in every direction.

The network is deliberately small — two hidden layers of 64, ReLU, Dropout,
trained with Adam — per Unit I, and because the decision it has to learn is
narrow.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger
from sentry_ai.decision.dqn_controller import POLICY_FEATURES, action_at, policy_features
from sentry_ai.domain.enums import EntityKind
from sentry_ai.interfaces.decision import FinalAction, IDecisionFusion
from sentry_ai.interfaces.navigation import (
    LOCAL_ACTION_ORDER,
    LocalAction,
    LocalDecision,
    LocalObservation,
)
from sentry_ai.interfaces.perception import WorldDetection
from sentry_ai.interfaces.sequence import BehaviourSignal
from sentry_ai.sensors.frame import YOLO_CLASSES
from sentry_ai.sequence.behaviour import BEHAVIOUR_ORDER

logger = get_logger(__name__)

#: Egocentric tiles the camera features describe, as (forward, right) offsets.
CAMERA_TILES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("ahead", (1, 0)),
    ("ahead2", (2, 0)),
    ("behind", (-1, 0)),
    ("left", (0, -1)),
    ("right", (0, 1)),
)

#: Feature groups, in vector order, with their widths. Ablations switch
#: whole groups off; this is the one place their layout is defined.
FEATURE_GROUPS: tuple[tuple[str, int], ...] = (
    ("dqn", len(LOCAL_ACTION_ORDER)),
    ("lstm", len(BEHAVIOUR_ORDER)),
    ("camera", len(CAMERA_TILES) * len(YOLO_CLASSES)),
    ("observation", POLICY_FEATURES),
)

#: Total width of :func:`fusion_features`.
FUSION_FEATURES = sum(width for _, width in FEATURE_GROUPS)

#: Bumped whenever the checkpoint layout changes.
CHECKPOINT_FORMAT = 1


def fusion_features(
    observation: LocalObservation,
    sightings: Sequence[WorldDetection],
    behaviour: BehaviourSignal,
    local_decision: LocalDecision,
) -> NDArray[np.float32]:
    """Every signal as one vector, in :data:`FEATURE_GROUPS` order."""
    return np.concatenate(
        [
            _dqn_features(local_decision),
            _lstm_features(behaviour),
            camera_features(observation, sightings),
            policy_features(observation),
        ]
    ).astype(np.float32)


def camera_features(
    observation: LocalObservation, sightings: Sequence[WorldDetection]
) -> NDArray[np.float32]:
    """Highest confidence per class on each egocentric tile around the vehicle."""
    tiles = _egocentric_tiles(observation)
    classes = {kind: index for index, kind in enumerate(YOLO_CLASSES)}
    grid = np.zeros((len(CAMERA_TILES), len(YOLO_CLASSES)), dtype=np.float32)
    for sighting in sightings:
        column = classes.get(sighting.label)
        if column is None:
            continue
        covered = {tile.as_tuple() for tile in sighting.tiles}
        for row, tile in enumerate(tiles):
            if tile in covered:
                grid[row, column] = max(grid[row, column], sighting.confidence)
    return grid.reshape(-1)


def group_mask(groups: Sequence[str]) -> NDArray[np.float32]:
    """1.0 over the features belonging to ``groups``, 0.0 elsewhere.

    Raises:
        ValueError: If a group name is unknown.
    """
    known = {name for name, _ in FEATURE_GROUPS}
    unknown = set(groups) - known
    if unknown:
        raise ValueError(f"unknown feature groups {sorted(unknown)}; known: {sorted(known)}")
    return np.concatenate(
        [
            np.full(width, float(name in groups), dtype=np.float32)
            for name, width in FEATURE_GROUPS
        ]
    )


@dataclass(frozen=True)
class FusionArchitecture:
    """The network's shape, and which signal groups it is allowed to see.

    Attributes:
        groups: Feature groups the network reads. The rest are zeroed —
            which is how the single-signal baselines M7 compares against are
            built from the same code.
        hidden_sizes: Hidden layer widths.
        dropout: Dropout after every hidden layer.
    """

    groups: tuple[str, ...] = tuple(name for name, _ in FEATURE_GROUPS)
    hidden_sizes: tuple[int, ...] = (64, 64)
    dropout: float = 0.2

    def __post_init__(self) -> None:
        group_mask(self.groups)  # validates the names
        if not self.groups:
            raise ValueError("a fusion network must see at least one signal group")
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must be a non-empty tuple of positives")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be within [0, 1), got {self.dropout}")

    def as_dict(self) -> dict[str, Any]:
        """Plain values, for a checkpoint."""
        values = asdict(self)
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in values.items()
        }


class FusionNet(nn.Module):
    """``(n, FUSION_FEATURES)`` -> ``(n, 5)`` action logits: ReLU, Dropout."""

    def __init__(self, architecture: FusionArchitecture) -> None:
        """Build the layers; register the group mask as a buffer so it saves with the weights."""
        super().__init__()
        self.architecture = architecture
        self.register_buffer("mask", torch.from_numpy(group_mask(architecture.groups)))
        layers: list[nn.Module] = []
        width = FUSION_FEATURES
        for hidden in architecture.hidden_sizes:
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(architecture.dropout)]
            width = hidden
        layers.append(nn.Linear(width, len(LOCAL_ACTION_ORDER)))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Logits for each action, from the permitted features only."""
        mask = self.get_buffer("mask")
        return self.layers(features * mask)

    def to_checkpoint(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Everything needed to rebuild this network with its weights."""
        return {
            "format": CHECKPOINT_FORMAT,
            "architecture": self.architecture.as_dict(),
            "state_dict": self.state_dict(),
            "metadata": dict(metadata or {}),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> FusionNet:
        """Rebuild a network from :meth:`to_checkpoint`'s output.

        Raises:
            ValueError: If the checkpoint is from an incompatible format.
        """
        found = checkpoint.get("format")
        if found != CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported fusion checkpoint format {found!r}; expected "
                f"{CHECKPOINT_FORMAT}. Retrain with scripts/train_fusion.py."
            )
        spec = checkpoint["architecture"]
        network = cls(
            FusionArchitecture(
                groups=tuple(spec["groups"]),
                hidden_sizes=tuple(spec["hidden_sizes"]),
                dropout=float(spec["dropout"]),
            )
        )
        network.load_state_dict(checkpoint["state_dict"])
        return network


class MlpFusion(IDecisionFusion):
    """Fuses the upstream signals with a trained :class:`FusionNet`."""

    def __init__(self, network: FusionNet, device: str = "cpu") -> None:
        """Wrap a network for inference, in evaluation mode on ``device``."""
        self._device = torch.device(device)
        self._network = network.to(self._device).eval()

    @classmethod
    def from_checkpoint(cls, weights_path: Path, device: str = "cpu") -> MlpFusion:
        """Load a ``.pt`` file written by ``scripts/train_fusion.py``.

        Raises:
            AssetNotFoundError: If the file does not exist.
        """
        if not weights_path.is_file():
            raise AssetNotFoundError(f"Fusion weights not found: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        logger.info("Loaded decision fusion from %s on %s", weights_path, device)
        return cls(FusionNet.from_checkpoint(checkpoint), device=device)

    def fuse(
        self,
        observation: LocalObservation,
        sightings: Sequence[WorldDetection],
        behaviour: BehaviourSignal,
        local_decision: LocalDecision,
    ) -> FinalAction:
        """The most probable action, and that probability as the rationale score."""
        features = fusion_features(observation, sightings, behaviour, local_decision)
        with torch.inference_mode():
            logits = self._network(torch.from_numpy(features).unsqueeze(0).to(self._device))
            probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        return FinalAction(action=action_at(best), rationale_score=float(probabilities[best]))


class CameraVetoFusion(IDecisionFusion):
    """A hand-written fusion rule — the baseline the MLP must justify itself against.

    Take the local controller's action, unless it drives forward (or
    backward) into a tile where the onboard camera sees fire or debris with
    at least ``threshold`` confidence; then stop. This is the obvious rule a
    human would write for the problem fusion is solving. Reporting it next to
    the MLP is how the MLP's result is kept honest: if a two-line rule does as
    well, the network has earned its place only as a Unit I exercise.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Veto moves into tiles sighted as hazards at or above ``threshold``."""
        self._threshold = threshold

    def fuse(
        self,
        observation: LocalObservation,
        sightings: Sequence[WorldDetection],
        behaviour: BehaviourSignal,
        local_decision: LocalDecision,
    ) -> FinalAction:
        """The local action, or STOP if the camera shows it is unsafe."""
        action = local_decision.action
        camera = camera_features(observation, sightings).reshape(len(CAMERA_TILES), -1)
        hazards = [YOLO_CLASSES.index(EntityKind.FIRE), YOLO_CLASSES.index(EntityKind.OBSTACLE)]
        rows = {name: index for index, (name, _) in enumerate(CAMERA_TILES)}
        target = {LocalAction.MOVE_FORWARD: "ahead", LocalAction.REVERSE: "behind"}.get(action)
        if target is not None and camera[rows[target], hazards].max() >= self._threshold:
            return FinalAction(action=LocalAction.STOP, rationale_score=1.0)
        return FinalAction(action=action, rationale_score=1.0)


def _dqn_features(local_decision: LocalDecision) -> NDArray[np.float32]:
    """The Q-values as a softmax: scale-free, so only their differences matter."""
    values = [local_decision.q_values.get(action, 0.0) for action in LOCAL_ACTION_ORDER]
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    total = sum(exponentials)
    return np.array([value / total for value in exponentials], dtype=np.float32)


def _lstm_features(behaviour: BehaviourSignal) -> NDArray[np.float32]:
    """One-hot of the predicted behaviour, scaled by its confidence."""
    features = np.zeros(len(BEHAVIOUR_ORDER), dtype=np.float32)
    features[BEHAVIOUR_ORDER.index(behaviour.predicted_class)] = behaviour.confidence
    return features


def _egocentric_tiles(observation: LocalObservation) -> list[tuple[int, int]]:
    """The ``(x, y)`` of each :data:`CAMERA_TILES` entry, for where the vehicle faces now.

    Plain tuples, not ``Position``: at the map edge "behind" can be off the
    map, which ``Position`` rightly refuses to represent. Nothing is ever
    sighted there, so such a tile simply stays at zero.
    """
    heading_x, heading_y = observation.heading.delta
    right_x, right_y = -heading_y, heading_x
    x, y = observation.position.x, observation.position.y
    return [
        (x + forward * heading_x + right * right_x, y + forward * heading_y + right * right_y)
        for _, (forward, right) in CAMERA_TILES
    ]

