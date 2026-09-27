"""LSTM implementing :class:`IMotionPredictor` (Unit III).

A window of recent :class:`~sentry_ai.interfaces.sequence.VehicleState` goes
in; a :class:`~sentry_ai.interfaces.sequence.BehaviourSignal` comes out —
which of ADVANCE, RETREAT, HOLD or DIVERT the vehicle is about to do over
the next few ticks, and how sure the model is.

**What each tick becomes.** Seven numbers, chosen so no feature needs the
network to learn arithmetic it could be handed:

====================  ================================================
``x``, ``y``          position, scaled to 0-1 by the map size
``sin h``, ``cos h``  heading as a point on a circle, so 350 degrees and
                      10 degrees are close, as they should be
``battery``           0-1
``dx``, ``dy``        the step just taken, clipped to -1..1 — motion,
                      which is what the classes are about
====================  ================================================

Position is included on purpose. There is one map, and "the vehicle is
about to turn" is largely a fact about *where corners are*. That is a real
limitation — see docs/architecture/phase5-sequence.md — and it is also
exactly what a sequence model over a fixed city should exploit.

**The checkpoint carries everything inference needs**: architecture, window
length, map size, and the behaviour definition (horizon and cone) the model
was trained to predict. A predictor can be loaded and run from the file
alone, and cannot be silently paired with the wrong map scale.
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
from sentry_ai.interfaces.sequence import (
    BehaviourClass,
    BehaviourSignal,
    IMotionPredictor,
    VehicleState,
)
from sentry_ai.sequence.behaviour import BEHAVIOUR_ORDER

logger = get_logger(__name__)

#: Numbers per tick; see the module docstring.
FEATURES_PER_STATE = 7

#: Bumped whenever the checkpoint layout changes.
CHECKPOINT_FORMAT = 1


@dataclass(frozen=True)
class LstmArchitecture:
    """Everything needed to rebuild the network and feed it correctly.

    Attributes:
        window: States per prediction.
        grid_width: Map width in tiles — the position scale.
        grid_height: Map height in tiles.
        hidden_size: LSTM hidden units per layer.
        num_layers: Stacked LSTM layers.
        dropout: Between LSTM layers and before the classifier.
        horizon: Ticks ahead the predicted behaviour spans. Not used by the
            network; recorded so a checkpoint says what it predicts.
        cone_degrees: The behaviour definition's cone, recorded likewise.
    """

    window: int
    grid_width: int
    grid_height: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    horizon: int = 4
    cone_degrees: float = 30.0

    def __post_init__(self) -> None:
        for name, value in (
            ("window", self.window),
            ("grid_width", self.grid_width),
            ("grid_height", self.grid_height),
            ("hidden_size", self.hidden_size),
            ("num_layers", self.num_layers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    def as_dict(self) -> dict[str, Any]:
        """Plain values, for a checkpoint."""
        return asdict(self)


class BehaviourLstmNet(nn.Module):
    """``(n, window, 7)`` features -> ``(n, 4)`` behaviour logits."""

    def __init__(self, architecture: LstmArchitecture) -> None:
        """Build the layers ``architecture`` describes."""
        super().__init__()
        self.architecture = architecture
        self.lstm = nn.LSTM(
            input_size=FEATURES_PER_STATE,
            hidden_size=architecture.hidden_size,
            num_layers=architecture.num_layers,
            batch_first=True,
            # PyTorch applies this between layers only, and warns if there
            # are none to apply it between.
            dropout=architecture.dropout if architecture.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(architecture.dropout)
        self.classifier = nn.Linear(architecture.hidden_size, len(BEHAVIOUR_ORDER))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Classify each window from the LSTM's output at its last tick."""
        outputs, _ = self.lstm(windows)
        return self.classifier(self.dropout(outputs[:, -1, :]))

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
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> BehaviourLstmNet:
        """Rebuild a network from :meth:`to_checkpoint`'s output.

        Raises:
            ValueError: If the checkpoint is from an incompatible format.
        """
        found = checkpoint.get("format")
        if found != CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported LSTM checkpoint format {found!r}; expected "
                f"{CHECKPOINT_FORMAT}. Retrain with scripts/train_lstm.py."
            )
        network = cls(LstmArchitecture(**checkpoint["architecture"]))
        network.load_state_dict(checkpoint["state_dict"])
        return network


class LstmMotionPredictor(IMotionPredictor):
    """Predicts near-term behaviour with a trained :class:`BehaviourLstmNet`."""

    def __init__(self, network: BehaviourLstmNet, device: str = "cpu") -> None:
        """Wrap a network for inference, in evaluation mode on ``device``."""
        self._device = torch.device(device)
        self._network = network.to(self._device).eval()
        self._architecture = network.architecture

    @classmethod
    def from_checkpoint(cls, weights_path: Path, device: str = "cpu") -> LstmMotionPredictor:
        """Load a ``.pt`` file written by ``scripts/train_lstm.py``.

        Raises:
            AssetNotFoundError: If the file does not exist.
        """
        if not weights_path.is_file():
            raise AssetNotFoundError(f"LSTM weights not found: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        network = BehaviourLstmNet.from_checkpoint(checkpoint)
        logger.info(
            "Loaded behaviour LSTM from %s (%d parameters) on %s",
            weights_path,
            network.parameter_count(),
            device,
        )
        return cls(network, device=device)

    @property
    def window(self) -> int:
        """How many past states a prediction uses."""
        return self._architecture.window

    def predict(self, state_history: Sequence[VehicleState]) -> BehaviourSignal:
        """Predict the behaviour over the next ``horizon`` ticks.

        Only the last ``window`` states are used. A shorter history — the
        first ticks of a mission — is padded at the front by repeating its
        oldest state, which reads as "the vehicle was standing still", the
        honest assumption about a time nobody recorded.

        Raises:
            ValueError: If ``state_history`` is empty.
        """
        probabilities = self.probabilities(state_history)
        best = max(probabilities, key=probabilities.__getitem__)
        return BehaviourSignal(predicted_class=best, confidence=probabilities[best])

    def probabilities(self, state_history: Sequence[VehicleState]) -> dict[BehaviourClass, float]:
        """The full distribution over behaviours, not just the winner.

        Phase 7's fusion network wants this: "70% advance, 25% divert" is
        more to go on than "advance".
        """
        features = encode_states(
            pad_history(state_history, self.window),
            self._architecture.grid_width,
            self._architecture.grid_height,
        )
        with torch.inference_mode():
            logits = self._network(torch.from_numpy(features).unsqueeze(0).to(self._device))
            scores = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        pairs = zip(BEHAVIOUR_ORDER, scores, strict=True)
        return {behaviour: float(score) for behaviour, score in pairs}


def pad_history(states: Sequence[VehicleState], window: int) -> list[VehicleState]:
    """The last ``window`` states, front-padded by repeating the oldest.

    Raises:
        ValueError: If ``states`` is empty.
    """
    if not states:
        raise ValueError("a behaviour prediction needs at least one vehicle state")
    recent = list(states[-window:])
    return [recent[0]] * (window - len(recent)) + recent


def encode_states(
    states: Sequence[VehicleState], grid_width: int, grid_height: int
) -> NDArray[np.float32]:
    """Turn states into the ``(len(states), 7)`` feature matrix the LSTM reads.

    The first state's step is zero: nothing before it is known.
    """
    x_scale = max(1, grid_width - 1)
    y_scale = max(1, grid_height - 1)
    rows = np.zeros((len(states), FEATURES_PER_STATE), dtype=np.float32)
    previous: VehicleState | None = None
    for index, state in enumerate(states):
        radians = math.radians(state.heading_degrees)
        step_x = 0 if previous is None else state.position.x - previous.position.x
        step_y = 0 if previous is None else state.position.y - previous.position.y
        rows[index] = (
            state.position.x / x_scale,
            state.position.y / y_scale,
            math.sin(radians),
            math.cos(radians),
            state.battery_percent / 100.0,
            max(-1, min(1, step_x)),
            max(-1, min(1, step_y)),
        )
        previous = state
    return rows
