"""MLP decision fusion implementing :class:`IDecisionFusion` (Unit I).

The network is the syllabus unit to the letter — fully connected layers,
ReLU activations, Dropout, trained with Adam — and its job is to settle
disagreements. Three signals arrive every tick:

==============  =====  ===================================================
group           width  what it says
==============  =====  ===================================================
``POLICY``      5      the DQN's Q-values, rescaled to -1..0 (0 = its pick)
``BEHAVIOUR``   4      the LSTM's predicted behaviour, one-hot x confidence
``CAMERA``      18     onboard-camera confidence per class and region
==============  =====  ===================================================

Usually they agree and fusion simply passes the DQN's choice through. They
disagree when the command center's map has fallen behind the world: the
DQN, driving on that map, wants to go forward; the camera sees debris on
the tile ahead. The network learns from the simulator's ground truth which
signal to believe.

**Ablation is built in.** A network can be trained on any subset of the
groups — the others are zeroed on the way in — and the subset is saved in
the checkpoint, so "fusion vs each signal alone" (milestone M7) compares
networks of identical shape trained identically.

:func:`fusion_features` is the single definition of the input, used by
training and by this adapter, so the two cannot drift apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger
from sentry_ai.interfaces.decision import (
    EVIDENCE_KINDS,
    EVIDENCE_REGIONS,
    FinalAction,
    IDecisionFusion,
    SceneEvidence,
)
from sentry_ai.interfaces.navigation import LOCAL_ACTION_ORDER, LocalAction, LocalDecision
from sentry_ai.interfaces.sequence import BehaviourSignal
from sentry_ai.sequence.behaviour import BEHAVIOUR_ORDER

logger = get_logger(__name__)

#: Bumped whenever the checkpoint layout changes.
CHECKPOINT_FORMAT = 1


class FeatureGroup(Enum):
    """The three upstream signals, as switchable blocks of the input."""

    POLICY = "policy"
    BEHAVIOUR = "behaviour"
    CAMERA = "camera"


#: Width of each group, in input order.
GROUP_WIDTHS: dict[FeatureGroup, int] = {
    FeatureGroup.POLICY: len(LOCAL_ACTION_ORDER),
    FeatureGroup.BEHAVIOUR: len(BEHAVIOUR_ORDER),
    FeatureGroup.CAMERA: len(EVIDENCE_KINDS) * len(EVIDENCE_REGIONS),
}

#: Length of :func:`fusion_features`' output.
FUSION_FEATURES = sum(GROUP_WIDTHS.values())

#: Starting weight of the DQN skip connection. Policy features span -1..0,
#: so 4.0 starts the network agreeing with the DQN at about 98% confidence.
_INITIAL_PRIOR_SCALE = 4.0


def fusion_features(
    evidence: SceneEvidence, behaviour: BehaviourSignal, local_decision: LocalDecision
) -> NDArray[np.float32]:
    """The fixed-length vector the network reads; see the module docstring.

    Q-values are rescaled per tick to ``(q - max) / (max - min)``: their
    absolute size depends on the reward scale and drifts between training
    runs, while *which action the DQN prefers and by how much* does not.
    """
    q = np.array(
        [local_decision.q_values.get(action, 0.0) for action in LOCAL_ACTION_ORDER],
        dtype=np.float32,
    )
    spread = float(q.max() - q.min())
    policy = (q - q.max()) / spread if spread > 1e-6 else np.zeros_like(q)
    behaviour_part = np.array(
        [
            behaviour.confidence if behaviour.predicted_class is kind else 0.0
            for kind in BEHAVIOUR_ORDER
        ],
        dtype=np.float32,
    )
    camera = np.array(evidence.as_list(), dtype=np.float32)
    return np.concatenate([policy, behaviour_part, camera]).astype(np.float32)


def group_mask(groups: tuple[FeatureGroup, ...]) -> NDArray[np.float32]:
    """1.0 over the features of ``groups``, 0.0 elsewhere."""
    parts = [
        np.full(width, 1.0 if group in groups else 0.0, dtype=np.float32)
        for group, width in GROUP_WIDTHS.items()
    ]
    return np.concatenate(parts)


@dataclass(frozen=True)
class FusionArchitecture:
    """Everything needed to rebuild the network and feed it correctly.

    Attributes:
        hidden_sizes: Width of each hidden layer, input side first.
        dropout: Dropout after every hidden activation.
        groups: The signals this network was trained on; the rest are
            zeroed at the input.
    """

    hidden_sizes: tuple[int, ...] = (64, 32)
    dropout: float = 0.2
    groups: tuple[FeatureGroup, ...] = tuple(FeatureGroup)

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must be positive, got {self.hidden_sizes}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be within [0, 1), got {self.dropout}")
        if not self.groups:
            raise ValueError("a fusion network needs at least one feature group")

    def as_dict(self) -> dict[str, Any]:
        """Plain values, for a checkpoint."""
        values = asdict(self)
        values["hidden_sizes"] = list(self.hidden_sizes)
        values["groups"] = [group.value for group in self.groups]
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FusionArchitecture:
        """Inverse of :meth:`as_dict`."""
        return cls(
            hidden_sizes=tuple(int(size) for size in values["hidden_sizes"]),
            dropout=float(values["dropout"]),
            groups=tuple(FeatureGroup(name) for name in values["groups"]),
        )


class FusionNet(nn.Module):
    """``(n, 27)`` features -> ``(n, 5)`` action logits: Linear, ReLU, Dropout.

    The DQN's own preference is added to the output through a learned skip
    connection (``prior_scale`` x the policy block). Nearly every tick the
    right answer *is* the DQN's choice, and an MLP asked to rediscover that
    from scratch spends its capacity copying; with the skip it starts from
    "agree with the DQN" and the hidden layers only have to learn when not
    to. A network trained without the policy group has that block zeroed by
    the mask, so the skip contributes nothing and the comparison stays fair.
    """

    def __init__(self, architecture: FusionArchitecture) -> None:
        """Build the layers ``architecture`` describes."""
        super().__init__()
        self.architecture = architecture
        self.register_buffer("mask", torch.from_numpy(group_mask(architecture.groups)))
        layers: list[nn.Module] = []
        width = FUSION_FEATURES
        for size in architecture.hidden_sizes:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(architecture.dropout)]
            width = size
        layers.append(nn.Linear(width, len(LOCAL_ACTION_ORDER)))
        self.layers = nn.Sequential(*layers)
        self.prior_scale = nn.Parameter(torch.tensor(_INITIAL_PRIOR_SCALE))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Zero the groups this network does not use, classify, add the DQN prior."""
        masked = features * self.get_buffer("mask")
        prior = masked[:, : GROUP_WIDTHS[FeatureGroup.POLICY]]
        return self.layers(masked) + self.prior_scale * prior

    def parameter_count(self) -> int:
        """Trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

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
        network = cls(FusionArchitecture.from_dict(checkpoint["architecture"]))
        network.load_state_dict(checkpoint["state_dict"])
        return network


class MlpFusion(IDecisionFusion):
    """Fuses the three signals with a trained :class:`FusionNet`, greedily."""

    def __init__(
        self, network: FusionNet, device: str = "cpu", override_threshold: float = 0.0
    ) -> None:
        """Wrap a network for inference, in evaluation mode on ``device``.

        Args:
            network: The trained network.
            device: Where inference runs.
            override_threshold: The probability fusion's choice needs before
                it may overrule the DQN. ``0.0`` always takes the argmax. An
                override the network is unsure of costs more than it saves:
                every wrong one sends the vehicle off its route.
        """
        self._device = torch.device(device)
        self._network = network.to(self._device).eval()
        self._override_threshold = override_threshold

    @classmethod
    def from_checkpoint(
        cls, weights_path: Path, device: str = "cpu", override_threshold: float = 0.0
    ) -> MlpFusion:
        """Load a ``.pt`` file written by ``scripts/train_fusion.py``.

        Raises:
            AssetNotFoundError: If the file does not exist.
        """
        if not weights_path.is_file():
            raise AssetNotFoundError(f"Fusion weights not found: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        network = FusionNet.from_checkpoint(checkpoint)
        logger.info(
            "Loaded fusion MLP from %s (%d parameters, groups: %s) on %s",
            weights_path,
            network.parameter_count(),
            ", ".join(group.value for group in network.architecture.groups),
            device,
        )
        return cls(network, device=device, override_threshold=override_threshold)

    @property
    def groups(self) -> tuple[FeatureGroup, ...]:
        """The signals this fusion network listens to."""
        return self._network.architecture.groups

    def fuse(
        self,
        evidence: SceneEvidence,
        behaviour: BehaviourSignal,
        local_decision: LocalDecision,
    ) -> FinalAction:
        """The most probable action, with that probability as the rationale score.

        An action other than the DQN's is taken only when its probability
        reaches the override threshold; otherwise the DQN's choice stands.
        """
        probabilities = self.probabilities(evidence, behaviour, local_decision)
        best = max(probabilities, key=probabilities.__getitem__)
        if best is not local_decision.action and probabilities[best] < self._override_threshold:
            best = local_decision.action
        return FinalAction(action=best, rationale_score=min(1.0, max(0.0, probabilities[best])))

    def probabilities(
        self,
        evidence: SceneEvidence,
        behaviour: BehaviourSignal,
        local_decision: LocalDecision,
    ) -> dict[LocalAction, float]:
        """The full distribution over actions, for the mission-control display."""
        features = fusion_features(evidence, behaviour, local_decision)
        with torch.inference_mode():
            logits = self._network(torch.from_numpy(features).unsqueeze(0).to(self._device))
            scores = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        return {
            action: float(score) for action, score in zip(LOCAL_ACTION_ORDER, scores, strict=True)
        }
