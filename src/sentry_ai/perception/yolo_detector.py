"""YOLOv8n adapter implementing :class:`IVisionDetector` (Unit II).

Wraps an Ultralytics model behind the project's own port so nothing
downstream imports ``ultralytics``. That matters for more than tidiness:
the mission loop, the occupancy grid, and every test that fakes a detector
stay free of a heavyweight dependency that pulls in Torch.

The adapter's whole job is translation. Ultralytics speaks in tensors,
``xyxy`` floats, and integer class ids; the rest of this project speaks in
:class:`~sentry_ai.interfaces.perception.Detection`,
:class:`~sentry_ai.domain.enums.EntityKind`, and integer pixel boxes. The
mapping between the two lives here and nowhere else.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger
from sentry_ai.interfaces.perception import BoundingBox, Detection, IVisionDetector
from sentry_ai.sensors.frame import YOLO_CLASSES

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    from ultralytics import YOLO

logger = get_logger(__name__)

#: Weights file shipped by Ultralytics for transfer learning. Nano because
#: the target machine is a laptop, per the project's hardware constraint.
PRETRAINED_WEIGHTS = "yolov8n.pt"


class YoloDetector(IVisionDetector):
    """Finds victims, fire, and debris in a camera frame."""

    def __init__(
        self,
        weights_path: Path,
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 256,
        device: str = "cpu",
    ) -> None:
        """Load a trained model.

        Args:
            weights_path: A ``.pt`` file produced by ``train_yolo.py``.
            confidence: Minimum score for a detection to be returned. The
                command center would rather investigate a false victim than
                miss a real one, so this is deliberately permissive.
            iou: IoU threshold for non-maximum suppression.
            image_size: Inference resolution. Must match training, or every
                box lands in the wrong place.
            device: ``"cpu"``, ``"cuda"``, or a device index.

        Raises:
            AssetNotFoundError: If the weights file does not exist. Checked
                eagerly because the alternative is a confusing failure deep
                inside a mission.
        """
        if not weights_path.is_file():
            raise AssetNotFoundError(f"YOLO weights not found: {weights_path}")

        from ultralytics import YOLO  # noqa: PLC0415 - keeps Torch off the import path

        self._model: YOLO = YOLO(str(weights_path))
        self._confidence = confidence
        self._iou = iou
        self._image_size = image_size
        self._device = device
        logger.info("Loaded YOLO detector from %s on %s", weights_path, device)

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Return every detection found in ``frame``.

        Args:
            frame: An ``(h, w, 3)`` RGB array, exactly as
                :class:`~sentry_ai.sensors.frame.CameraFrame` provides one.

        Returns:
            Detections in frame-pixel coordinates, unsorted.
        """
        results = list(
            self._model.predict(
                source=frame[:, :, ::-1],  # Ultralytics expects BGR
                conf=self._confidence,
                iou=self._iou,
                imgsz=self._image_size,
                device=self._device,
                verbose=False,
            )
        )
        if not results:
            return []
        return self._to_detections(results[0], frame.shape[1], frame.shape[0])

    @staticmethod
    def _to_detections(result: Any, width: int, height: int) -> list[Detection]:
        """Convert one Ultralytics result into the project's own type.

        Boxes are clipped to the frame; anything falling entirely outside
        it is dropped rather than squashed against an edge.
        """
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        for xyxy, confidence, class_id in zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
        ):
            label_index = int(class_id)
            if not 0 <= label_index < len(YOLO_CLASSES):
                logger.warning("Detector produced unknown class id %d — skipped", label_index)
                continue
            box = _clamped_box(xyxy, width, height)
            if box is None:
                continue
            detections.append(
                Detection(
                    label=YOLO_CLASSES[label_index],
                    confidence=min(1.0, max(0.0, float(confidence))),
                    bbox=box,
                )
            )
        return detections


def _clamped_box(xyxy: list[float], width: int, height: int) -> BoundingBox | None:
    """Clip a float box to the frame and round it to integer pixels.

    Returns ``None`` when nothing of the box lies inside the frame — a
    detection entirely off-image is garbage, and
    :class:`~sentry_ai.interfaces.perception.BoundingBox` would reject the
    zero-area result anyway.

    The integer box *covers* the float one — the minimum is floored and the
    maximum ceiled — so rounding never shaves a pixel off a detection. A box
    that rounds to sub-pixel is widened to one pixel rather than dropped:
    victims are the smallest and rarest class here, so discarding a marginal
    one is the more expensive mistake.
    """
    left = max(0.0, min(xyxy[0], float(width)))
    top = max(0.0, min(xyxy[1], float(height)))
    right = max(0.0, min(xyxy[2], float(width)))
    bottom = max(0.0, min(xyxy[3], float(height)))
    if right <= left or bottom <= top:
        return None

    x_min, y_min = math.floor(left), math.floor(top)
    return BoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=max(x_min + 1, min(math.ceil(right), width)),
        y_max=max(y_min + 1, min(math.ceil(bottom), height)),
    )
