"""Convolutional denoising autoencoder implementing :class:`IDenoiser` (Unit IV).

Two things live here, and they are kept together on purpose:

* :class:`DenoisingAutoencoderNet` — the PyTorch network. An encoder of
  stride-2 convolutions squeezes a frame into a low-resolution code; a
  decoder of upsampling convolutions expands that code back into an image.
  Trained on (corrupted, clean) pairs, the code has to keep what the scene
  *is* and drop what the smoke, blur and sensor did to it. That is the
  representation-learning half of the unit.
* :class:`ConvDenoisingAutoencoder` — the adapter the pipeline holds. uint8
  RGB in, uint8 RGB out, the same shape, nothing downstream aware of Torch.

The checkpoint format is defined here too, not in ``training/``, because the
live pipeline has to read it and ``training/`` is never imported at runtime.
A checkpoint carries its own architecture, so a model can never be loaded
into a network of the wrong shape — the denoiser's version of the
"``image_size`` must match training" trap the detector has.

**Torch is imported at module level.** Unlike ``yolo_detector``, this module
*is* the model; composition roots import it lazily, only when a denoiser has
been asked for, so nothing else pays for Torch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger
from sentry_ai.interfaces.perception import IDenoiser

logger = get_logger(__name__)

#: Bumped whenever the checkpoint layout changes, so an old file fails with
#: a clear message rather than a missing-key error from ``load_state_dict``.
CHECKPOINT_FORMAT = 1

_MAX_PIXEL = 255.0


@dataclass(frozen=True)
class AutoencoderArchitecture:
    """The shape of the network — everything needed to rebuild it.

    Attributes:
        base_channels: Feature maps at full resolution; doubled per level.
        depth: Number of stride-2 downsamplings between input and code.
        skip_connections: Concatenate each encoder level's features into
            the matching decoder level.
    """

    base_channels: int = 32
    depth: int = 3
    skip_connections: bool = True

    def __post_init__(self) -> None:
        if self.base_channels <= 0 or self.depth <= 0:
            raise ValueError(
                f"base_channels and depth must be positive, got "
                f"{self.base_channels} and {self.depth}"
            )

    @property
    def widths(self) -> tuple[int, ...]:
        """Channel count at each level, full resolution first."""
        return tuple(self.base_channels * 2**level for level in range(self.depth + 1))

    def as_dict(self) -> dict[str, Any]:
        """Plain values, for a checkpoint."""
        return asdict(self)


class DenoisingAutoencoderNet(nn.Module):
    """Encoder-decoder over ``(n, 3, h, w)`` images scaled to 0-1.

    Any frame size works: the decoder upsamples to the exact size of the
    level it is rebuilding rather than by a fixed factor of two, so a
    dimension that is not a multiple of ``2 ** depth`` (the 144 px onboard
    view at depth 5, say) round-trips without padding.

    Output passes through a sigmoid, so it is always a valid image.
    """

    def __init__(self, architecture: AutoencoderArchitecture) -> None:
        """Build the layers ``architecture`` describes."""
        super().__init__()
        self.architecture = architecture
        widths = architecture.widths

        self.stem = _conv_block(3, widths[0])
        self.encoder = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(widths[level], widths[level + 1], 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                _conv_block(widths[level + 1], widths[level + 1]),
            )
            for level in range(architecture.depth)
        )
        merge_factor = 2 if architecture.skip_connections else 1
        self.upsample = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(widths[level + 1], widths[level], 3, padding=1),
                nn.ReLU(inplace=True),
            )
            for level in reversed(range(architecture.depth))
        )
        self.decoder = nn.ModuleList(
            _conv_block(widths[level] * merge_factor, widths[level])
            for level in reversed(range(architecture.depth))
        )
        self.head = nn.Conv2d(widths[0], 3, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Reconstruct clean images from corrupted ones, same shape."""
        features = [self.stem(images)]
        for stage in self.encoder:
            features.append(stage(features[-1]))

        code = features[-1]
        for upsample, decode, skip in zip(
            self.upsample, self.decoder, reversed(features[:-1]), strict=True
        ):
            code = functional.interpolate(
                code, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            code = upsample(code)
            if self.architecture.skip_connections:
                code = torch.cat([code, skip], dim=1)
            code = decode(code)
        return torch.sigmoid(self.head(code))

    def parameter_count(self) -> int:
        """Trainable parameters — reported so model size is a number, not a guess."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def to_checkpoint(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Everything needed to rebuild this network with its weights.

        Args:
            metadata: Anything JSON-like worth keeping with the weights —
                the epoch, the validation score. Not read back by
                :meth:`from_checkpoint`.
        """
        return {
            "format": CHECKPOINT_FORMAT,
            "architecture": self.architecture.as_dict(),
            "state_dict": self.state_dict(),
            "metadata": dict(metadata or {}),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> DenoisingAutoencoderNet:
        """Rebuild a network from :meth:`to_checkpoint`'s output.

        Raises:
            ValueError: If the checkpoint is from an incompatible format.
        """
        found = checkpoint.get("format")
        if found != CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported denoiser checkpoint format {found!r}; expected "
                f"{CHECKPOINT_FORMAT}. Retrain with scripts/train_autoencoder.py."
            )
        network = cls(AutoencoderArchitecture(**checkpoint["architecture"]))
        network.load_state_dict(checkpoint["state_dict"])
        return network


class ConvDenoisingAutoencoder(IDenoiser):
    """Cleans camera frames with a trained :class:`DenoisingAutoencoderNet`."""

    def __init__(self, network: DenoisingAutoencoderNet, device: str = "cpu") -> None:
        """Wrap a network for inference.

        Args:
            network: Trained weights. Switched to evaluation mode and moved
                to ``device``.
            device: ``"cpu"``, ``"cuda"``, or a device index.
        """
        self._device = torch.device(device)
        self._network = network.to(self._device).eval()

    @classmethod
    def from_checkpoint(cls, weights_path: Path, device: str = "cpu") -> ConvDenoisingAutoencoder:
        """Load a ``.pt`` file written by ``scripts/train_autoencoder.py``.

        Raises:
            AssetNotFoundError: If the file does not exist — checked eagerly,
                because the alternative is a confusing failure mid-mission.
        """
        if not weights_path.is_file():
            raise AssetNotFoundError(f"Denoiser weights not found: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        network = DenoisingAutoencoderNet.from_checkpoint(checkpoint)
        logger.info(
            "Loaded denoiser from %s (%d parameters) on %s",
            weights_path,
            network.parameter_count(),
            device,
        )
        return cls(network, device=device)

    def denoise(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Return a cleaned copy of an ``(h, w, 3)`` RGB frame.

        Raises:
            ValueError: If ``frame`` is not an ``(h, w, 3)`` uint8 array.
        """
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"denoise expects an (h, w, 3) uint8 frame, got {frame.shape} {frame.dtype}"
            )
        with torch.inference_mode():
            cleaned = self._network(to_tensor(frame).unsqueeze(0).to(self._device))
        return to_image(cleaned.squeeze(0))


def to_tensor(image: NDArray[np.uint8]) -> torch.Tensor:
    """``(h, w, 3)`` uint8 -> ``(3, h, w)`` float in 0-1. Copies; never aliases."""
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / _MAX_PIXEL


def to_image(tensor: torch.Tensor) -> NDArray[np.uint8]:
    """``(3, h, w)`` float in 0-1 -> ``(h, w, 3)`` uint8, rounded rather than truncated."""
    scaled = (tensor.detach().clamp(0.0, 1.0) * _MAX_PIXEL).round().to(torch.uint8)
    return np.ascontiguousarray(scaled.permute(1, 2, 0).cpu().numpy())


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3 convolutions with ReLU — the unit every level is built from."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )
