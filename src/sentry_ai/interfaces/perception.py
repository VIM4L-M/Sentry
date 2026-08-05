"""Ports for the perception pipeline: denoising (Unit IV) and detection (Unit II).

Concrete adapters — ``ConvDenoisingAutoencoder`` (Phase 4) and
``YoloDetector`` (Phase 3) — will live in ``sentry_ai.perception`` and
implement these ABCs. This file defines the contract only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sentry_ai.domain.enums import EntityKind


@dataclass(frozen=True)
class BoundingBox:
    """A detection's location in image space, in pixel coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(
                f"BoundingBox must have positive width/height, got "
                f"({self.x_min}, {self.y_min}) -> ({self.x_max}, {self.y_max})"
            )


@dataclass(frozen=True)
class Detection:
    """One object found in a camera frame by an :class:`IVisionDetector`."""

    label: EntityKind
    confidence: float
    bbox: BoundingBox

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Detection.confidence must be within 0.0-1.0, got {self.confidence}")


class IDenoiser(ABC):
    """Cleans a noisy/occluded camera frame before detection.

    Implemented in Phase 4 by a convolutional denoising autoencoder
    (Unit IV — Representation Learning).
    """

    @abstractmethod
    def denoise(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Return a denoised copy of ``frame``. Must not mutate the input array."""
        raise NotImplementedError


class IVisionDetector(ABC):
    """Finds victims, fire, and obstacles in a camera frame.

    Implemented in Phase 3 by a fine-tuned YOLOv8n model (Unit II —
    Computer Vision).
    """

    @abstractmethod
    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Return every detection found in ``frame``."""
        raise NotImplementedError
