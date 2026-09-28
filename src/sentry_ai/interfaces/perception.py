"""Ports for the perception pipeline: denoising (Unit IV) and detection (Unit II).

Concrete adapters — ``ConvDenoisingAutoencoder`` (Phase 4) and
``YoloDetector`` (Phase 3) — live in ``sentry_ai.perception`` and implement
these ABCs. This file defines the contract only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sentry_ai.domain.entities import Position
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


@dataclass(frozen=True)
class WorldDetection:
    """One object, in map space, possibly seen by several cameras.

    Produced by :class:`~sentry_ai.perception.merger.DetectionMerger` from
    image-space :class:`Detection` s. It lives here, beside ``Detection``,
    rather than with the merger, because decision fusion consumes it
    through its port (ADR 0003) and ports may not import adapters.

    Attributes:
        label: What it is.
        tiles: Every world tile it covers. A victim or a piece of debris
            occupies one; a fire occupies its whole footprint.
        confidence: How sure the detector was. For a merged sighting this is
            the most confident single view of it.
        camera_ids: Every camera that contributed, so downstream code can
            tell a corroborated detection from a lone one.
    """

    label: EntityKind
    tiles: frozenset[Position]
    confidence: float
    camera_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.tiles:
            raise ValueError("WorldDetection must cover at least one tile")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"WorldDetection.confidence must be within 0.0-1.0, got {self.confidence}"
            )

    @property
    def position(self) -> Position:
        """A single representative tile — the one nearest the footprint's centre.

        Single-tile detections return that tile. For a fire this is where the
        blaze is *reported*, which is what a mission log or a HUD marker
        wants; code that must mark every burning cell uses :attr:`tiles`.
        """
        if len(self.tiles) == 1:
            return next(iter(self.tiles))
        mean_x = sum(tile.x for tile in self.tiles) / len(self.tiles)
        mean_y = sum(tile.y for tile in self.tiles) / len(self.tiles)
        return min(
            sorted(self.tiles, key=lambda tile: (tile.y, tile.x)),
            key=lambda tile: (tile.x - mean_x) ** 2 + (tile.y - mean_y) ** 2,
        )

    @property
    def corroborated(self) -> bool:
        """Whether more than one camera saw this."""
        return len(self.camera_ids) > 1


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

    def detect_many(self, frames: Sequence[NDArray[np.uint8]]) -> list[list[Detection]]:
        """Detect in several frames at once; one list per frame, in order.

        Added in Phase 8, and additive: the default simply calls
        :meth:`detect` per frame, so every existing detector keeps working.
        An adapter that can batch overrides it — the four CCTV frames of one
        tick are exactly such a batch, and on the YOLO adapter batching cuts
        their cost from ~80 ms to ~30 ms on a CPU.
        """
        return [self.detect(frame) for frame in frames]
