"""A captured camera frame: pixels plus the ground truth inside them.

A frame is deliberately more than an image. Every later phase needs the
labels as much as the pixels — Phase 3 to fine-tune YOLOv8n, Phase 4 to
score a denoiser, and every phase after that to check whether a detection
landed on the right tile. Producing the two together, from the same
rasterization pass, is the only way they cannot disagree.

Annotations are :class:`~sentry_ai.interfaces.perception.Detection` values
with confidence ``1.0`` — the identical type the trained detector emits, so
prediction and ground truth are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sentry_ai.common.exceptions import DomainValidationError
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind
from sentry_ai.interfaces.perception import BoundingBox, Detection
from sentry_ai.sensors.camera import CameraView

#: The classes the Phase 3 detector is trained on, in the order that fixes
#: their integer ids in a YOLO label file. The specification names exactly
#: these three. Appending is safe; reordering silently invalidates every
#: dataset ever exported, so don't.
YOLO_CLASSES: tuple[EntityKind, ...] = (
    EntityKind.VICTIM,
    EntityKind.FIRE,
    EntityKind.OBSTACLE,
)


def yolo_class_id(label: EntityKind) -> int:
    """The integer class id used for ``label`` in exported label files.

    Raises:
        DomainValidationError: If ``label`` is not a detector class.
    """
    try:
        return YOLO_CLASSES.index(label)
    except ValueError as exc:
        raise DomainValidationError(f"{label} is not one of the detector classes") from exc


@dataclass(frozen=True)
class CameraFrame:
    """One capture from one camera, with the ground truth it contains.

    Attributes:
        view: Where this camera was pointed, and the projection needed to
            turn a detection back into a world tile.
        pixels: ``uint8`` RGB array of shape ``(height, width, 3)``.
        annotations: Ground-truth detections visible in this frame.
    """

    view: CameraView
    pixels: NDArray[np.uint8]
    annotations: tuple[Detection, ...] = ()

    def __post_init__(self) -> None:
        if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
            raise DomainValidationError(
                f"CameraFrame.pixels must be (height, width, 3), got {self.pixels.shape}"
            )
        expected = (self.view.frame_height, self.view.frame_width, 3)
        if self.pixels.shape != expected:
            raise DomainValidationError(
                f"CameraFrame.pixels shape {self.pixels.shape} does not match camera "
                f"'{self.view.camera_id}' geometry {expected}"
            )

    @property
    def camera_id(self) -> str:
        """Which camera produced this frame."""
        return self.view.camera_id

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.pixels.shape[1])

    def world_position_of(self, detection: Detection) -> Position:
        """The world tile a detection sits on, from its box centre.

        The command center calls this on every detection before merging it
        into the occupancy grid — it is the bridge from image space back to
        map space.
        """
        box = detection.bbox
        centre_x = min((box.x_min + box.x_max) // 2, self.view.frame_width - 1)
        centre_y = min((box.y_min + box.y_max) // 2, self.view.frame_height - 1)
        return self.view.to_world(centre_x, centre_y)

    def with_pixels(self, pixels: NDArray[np.uint8]) -> CameraFrame:
        """A copy of this frame carrying different pixels but the same truth.

        Used by :mod:`sentry_ai.sensors.degradation`: smoke and noise change
        what the camera *sees*, never where the victims actually are.
        """
        return CameraFrame(view=self.view, pixels=pixels, annotations=self.annotations)

    def to_yolo_lines(self) -> list[str]:
        """This frame's labels in YOLO format: ``class cx cy w h``, normalized.

        Detections whose label is not one of :data:`YOLO_CLASSES` are
        skipped rather than raising — a frame may legitimately annotate
        something the detector is not being trained to find.
        """
        lines: list[str] = []
        for detection in self.annotations:
            if detection.label not in YOLO_CLASSES:
                continue
            lines.append(self._yolo_line(detection))
        return lines

    def _yolo_line(self, detection: Detection) -> str:
        """Format one detection as a normalized YOLO label row."""
        box = detection.bbox
        width = (box.x_max - box.x_min) / self.width
        height = (box.y_max - box.y_min) / self.height
        centre_x = (box.x_min + box.x_max) / 2 / self.width
        centre_y = (box.y_min + box.y_max) / 2 / self.height
        return (
            f"{yolo_class_id(detection.label)} "
            f"{centre_x:.6f} {centre_y:.6f} {width:.6f} {height:.6f}"
        )


def clipped_box(
    x_min: int, y_min: int, x_max: int, y_max: int, view: CameraView
) -> BoundingBox | None:
    """A box clipped to ``view``'s frame, or ``None`` if nothing remains.

    Entities on the edge of a camera's footprint are partly out of shot.
    Clipping keeps every exported label inside its image — a box that runs
    past the frame edge is rejected outright by most detector toolchains.
    """
    left = max(0, x_min)
    top = max(0, y_min)
    right = min(view.frame_width, x_max)
    bottom = min(view.frame_height, y_max)
    if right <= left or bottom <= top:
        return None
    return BoundingBox(x_min=left, y_min=top, x_max=right, y_max=bottom)
