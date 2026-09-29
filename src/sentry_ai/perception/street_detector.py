"""Real-world object detection on the real street photos (Phase 9 display).

The mission's own detector was trained on the simulated city and never sees
a photograph. This one is the stock YOLOv8n trained on COCO, run on the
Mapillary photo of the street the vehicle is on, so the front-camera panel
shows real vehicles and people detected in a real Chennai or Chicago street —
what an onboard camera in that street would report. It informs the display
only; the driving models do not read it.

Each photo is detected once and cached, so a photo seen again costs nothing.
Ultralytics is imported only when the model loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger

logger = get_logger(__name__)

#: Display relabels per ``region`` of a map config. COCO has no autorickshaw
#: class, and on Indian streets reports autos as trucks.
REGION_RELABELS: dict[str, dict[str, str]] = {"india": {"Truck": "Auto/Truck"}}

#: COCO classes worth showing on a street, and what the display calls them.
STREET_CLASSES: dict[str, str] = {
    "car": "Car",
    "motorcycle": "Bike",
    "bicycle": "Cycle",
    "bus": "Bus",
    "truck": "Truck",
    "person": "Pedestrian",
    "traffic light": "Signal",
}


@dataclass(frozen=True)
class StreetDetection:
    """One object found in a photo: what, how sure, and where (pixels, x0 y0 x1 y1)."""

    label: str
    confidence: float
    box: tuple[float, float, float, float]


class StreetPhotoDetector:
    """Detects street objects in photos with a pretrained COCO YOLOv8, caching per photo."""

    def __init__(
        self,
        model: Any,
        confidence: float = 0.35,
        relabel: dict[str, str] | None = None,
    ) -> None:
        """Wrap a loaded ``ultralytics.YOLO``; keep detections at or above ``confidence``.

        ``relabel`` renames display labels for a region, e.g. ``{"Truck":
        "Auto/Truck"}`` in India, where COCO — which has no autorickshaw
        class — reports autorickshaws as trucks.
        """
        self._model = model
        self._confidence = confidence
        self._relabel = relabel or {}
        self._cache: dict[str, tuple[StreetDetection, ...]] = {}

    @classmethod
    def from_weights(
        cls, weights: Path, device: str = "cpu", relabel: dict[str, str] | None = None
    ) -> StreetPhotoDetector:
        """Load COCO weights (``models/pretrained/yolov8n.pt``).

        Raises:
            AssetNotFoundError: If the file does not exist.
        """
        if not weights.is_file():
            raise AssetNotFoundError(f"street detector weights not found: {weights}")
        from ultralytics import YOLO  # noqa: PLC0415 - heavy import, only when used

        model = YOLO(str(weights))
        model.to(device)
        logger.info("Loaded street-photo detector %s on %s", weights, device)
        return cls(model, relabel=relabel)

    def detect(self, photo: Path) -> tuple[StreetDetection, ...]:
        """Street objects in ``photo``, from the cache after the first call."""
        key = str(photo)
        if key not in self._cache:
            self._cache[key] = self._run(photo)
        return self._cache[key]

    def _run(self, photo: Path) -> tuple[StreetDetection, ...]:
        result = self._model.predict(str(photo), conf=self._confidence, verbose=False)[0]
        names = result.names
        found = []
        for box in result.boxes:
            name = names[int(box.cls)]
            if name in STREET_CLASSES:
                x0, y0, x1, y1 = (float(v) for v in box.xyxy[0].tolist())
                label = STREET_CLASSES[name]
                label = self._relabel.get(label, label)
                found.append(StreetDetection(label, float(box.conf), (x0, y0, x1, y1)))
        return tuple(found)
