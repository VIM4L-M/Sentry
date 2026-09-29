#!/usr/bin/env python3
"""Live object detection and driving decisions from the laptop's camera (review demo).

For "show it something": point the webcam at a phone showing a street, a
toy car, a printed photo of traffic — or pass an image file. YOLOv8 (COCO)
boxes the road users it finds, and a panel says what SENTRY would do about
them and why: "Traffic ahead (4 vehicles): I wait and keep my gap", "Person in
front: I stop and give way", "Road clear: I drive on to the victim".

Only road-scene classes are shown by default (person, car, auto/truck, bus,
two-wheeler, cycle, traffic light, stop sign, cow, dog, horse, sheep), at a
confidence of 0.5 or more. COCO knows 80 everyday classes, and a camera
pointed around a room otherwise reports hands as teddy bears and brushes as
refrigerators. ``--all-classes`` shows everything. COCO has no autorickshaw
class; a bus or truck box that is mostly autorickshaw yellow is shown as
"Auto-rickshaw", otherwise autos show as "Auto/Truck" (sometimes "Car").

The model is YOLOv8m at 1280 px by default (about 45 ms a frame on an RTX
4060): small, far traffic lights are only found at that size, and the signal
colour decides between going with the traffic and waiting. ``--model n|s``
and ``--imgsz 640`` are faster. ``--sentry`` uses the mission's own detector, trained on the
simulated city's top-down frames, which is expected to do badly on real
photos — it knows the world it was trained on.

Keys: ``q`` or Escape quits, ``s`` saves a snapshot to ``runs/live/``.

Usage:
    python scripts/live_camera.py                       # webcam 0
    python scripts/live_camera.py --camera 1            # another camera
    python scripts/live_camera.py --image street.jpg    # one image, window + saved copy
    python scripts/live_camera.py --model m             # most accurate, slower
    python scripts/live_camera.py --all-classes         # every COCO class
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentry_ai.decision.scene_reasoner import (  # noqa: E402
    ACTION_COLOUR,
    DecisionHold,
    SeenObject,
    decide,
)

#: The stock COCO models by size, and the mission's own detector.
COCO_WEIGHTS = {s: f"models/pretrained/yolov8{s}.pt" for s in ("n", "s", "m")}
SENTRY_WEIGHTS = "models/yolo/labelfix/weights/best.pt"

#: Road-scene COCO classes and their on-screen names.
ROAD_CLASSES: dict[str, str] = {
    "person": "Person",
    "car": "Car",
    "truck": "Auto/Truck",
    "bus": "Bus",
    "motorcycle": "Two-wheeler",
    "bicycle": "Cycle",
    "traffic light": "Signal",
    "stop sign": "Stop sign",
    "auto": "Auto-rickshaw",
    "cow": "Cow",
    "dog": "Dog",
    "horse": "Horse",
    "sheep": "Sheep",
}

#: Traffic lights are small and far away, so they keep a lower confidence bar.
LIGHT_CONFIDENCE = 0.3

#: Share of yellow pixels that turns a COCO "bus"/"truck" into an autorickshaw.
AUTO_YELLOW_SHARE = 0.09

#: Where snapshots and annotated images are written.
OUTPUT_DIR = "runs/live"
#: Width of the decision panel, pixels.
PANEL_W = 400


def main() -> int:
    """Open the camera (or image) and show detections until the user quits."""
    args = _parse_args()
    import cv2  # noqa: PLC0415 - comes with ultralytics
    import torch  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    weights = PROJECT_ROOT / (SENTRY_WEIGHTS if args.sentry else COCO_WEIGHTS[args.model])
    if not weights.is_file() and not args.sentry:
        weights = PROJECT_ROOT / COCO_WEIGHTS["n"]
    if not weights.is_file():
        raise SystemExit(f"weights not found: {weights}")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = YOLO(str(weights))
    output = PROJECT_ROOT / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    source = "mission detector" if args.sentry else f"YOLOv8{weights.stem[-1]} COCO"
    title = f"SENTRY AI - live detection ({source})"
    road_only = not args.all_classes and not args.sentry
    view = _View(cv2, model, device, args.confidence, road_only, args.imgsz)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"cannot read image: {args.image}")
        annotated = view.annotate(frame, time.perf_counter())
        path = output / f"detected_{Path(args.image).stem}.jpg"
        cv2.imwrite(str(path), annotated)
        print(f"saved {path.relative_to(PROJECT_ROOT)}")
        _show(cv2, title, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"no camera at index {args.camera}; try --camera 1 or --image")
    try:
        _loop(cv2, capture, view, title, output, device)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


class _View:
    """Detects, decides, and draws one frame."""

    def __init__(  # noqa: PLR0913
        self,
        cv2,
        model,
        device: str,
        confidence: float,
        road_only: bool,
        imgsz: int = 1280,  # noqa: ANN001
    ) -> None:
        self._cv2 = cv2
        self._imgsz = imgsz
        self._model = model
        self._device = device
        self._confidence = confidence
        self._road_only = road_only
        self._hold = DecisionHold()

    def annotate(self, frame, now: float):  # noqa: ANN001, ANN201
        cv2 = self._cv2
        floor = min(self._confidence, LIGHT_CONFIDENCE)
        result = self._model.predict(
            frame, conf=floor, imgsz=self._imgsz, device=self._device, verbose=False
        )[0]
        height, width = frame.shape[:2]
        seen: list[SeenObject] = []
        drawn = frame.copy()
        for box in result.boxes:
            name = result.names[int(box.cls)]
            if self._road_only and name not in ROAD_CLASSES:
                continue
            if name != "traffic light" and float(box.conf) < self._confidence:
                continue
            x0, y0, x1, y1 = (int(v) for v in box.xyxy[0].tolist())
            if name in ("car", "truck") and y1 > 0.93 * height and x1 - x0 > 0.6 * width:
                continue  # the dashcam car's own bonnet, not a road user
            if name in ("bus", "truck") and _is_auto_yellow(frame[y0:y1, x0:x1]):
                name = "auto"  # COCO has no autorickshaw class; Chennai autos are yellow
            colour = _light_colour(frame[y0:y1, x0:x1]) if name == "traffic light" else None
            share = (x0 / width, y0 / height, x1 / width, y1 / height)
            seen.append(SeenObject(name, float(box.conf), share, colour))
            label = ROAD_CLASSES.get(name, name)
            if colour:
                label = f"{label} ({colour})"
            _box(cv2, drawn, (x0, y0, x1, y1), f"{label} {float(box.conf):.0%}", _box_bgr(name))
        decision = self._hold.update(decide(seen), now)
        return _with_panel(cv2, drawn, decision, len(seen))


def _loop(cv2, capture, view: _View, title: str, output: Path, device: str) -> None:  # noqa: ANN001
    last = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            print("camera stopped sending frames")
            return
        now = time.perf_counter()
        annotated = view.annotate(frame, now)
        fps = 1.0 / max(1e-6, now - last)
        last = now
        cv2.putText(
            annotated, f"{fps:4.1f} fps  on {device}", (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 255), 2,
        )  # fmt: skip
        _show(cv2, title, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return
        if key == ord("s"):
            path = output / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(path), annotated)
            print(f"saved {path.relative_to(PROJECT_ROOT)}")


def _show(cv2, title: str, image) -> None:  # noqa: ANN001
    """Show ``image``, shrunk if it is larger than a laptop screen."""
    height, width = image.shape[:2]
    scale = min(1.0, 1400 / width, 760 / height)
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)))
    cv2.imshow(title, image)


def _box(cv2, image, xyxy, text: str, bgr) -> None:  # noqa: ANN001
    x0, y0, x1, y1 = xyxy
    cv2.rectangle(image, (x0, y0), (x1, y1), bgr, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    top = max(0, y0 - th - 8)
    cv2.rectangle(image, (x0, top), (x0 + tw + 8, top + th + 8), bgr, -1)
    cv2.putText(image, text, (x0 + 4, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


def _box_bgr(name: str) -> tuple[int, int, int]:
    if name == "person":
        return (80, 80, 255)
    if name in ("truck", "bus", "auto"):
        return (40, 190, 255)
    if name in ("cow", "dog", "horse", "sheep"):
        return (200, 120, 255)
    if name in ("traffic light", "stop sign"):
        return (255, 220, 80)
    return (140, 214, 40)


def _with_panel(cv2, frame, decision, count: int):  # noqa: ANN001, ANN202
    """``frame`` with the decision panel added on its right."""
    import numpy as np  # noqa: PLC0415

    height = frame.shape[0]
    panel = np.full((height, PANEL_W, 3), (28, 22, 18), dtype=np.uint8)
    r, g, b = ACTION_COLOUR[decision.action]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, "SENTRY DECISION", (16, 34), font, 0.6, (180, 180, 180), 1)
    cv2.rectangle(panel, (16, 48), (PANEL_W - 16, 108), (b, g, r), -1)
    cv2.putText(panel, decision.text, (28, 92), font, 1.2, (0, 0, 0), 3)
    y = 142
    for line in _wrap(decision.headline, 30):
        cv2.putText(panel, line, (16, y), font, 0.62, (b, g, r), 2)
        y += 26
    y += 10
    if decision.reasons:
        cv2.putText(panel, "Also:", (16, y), font, 0.55, (170, 170, 170), 1)
        y += 26
    for reason in decision.reasons:
        for i, line in enumerate(_wrap(reason, 34)):
            cv2.putText(panel, ("- " if i == 0 else "  ") + line, (16, y), font, 0.52,
                        (230, 230, 230), 1)  # fmt: skip
            y += 22
        y += 6
    footer = [
        f"{count} road user(s) in view",
        "Rule: never hit a road user,",
        "then reach the victim fastest.",
    ]
    for i, line in enumerate(footer):
        cv2.putText(panel, line, (16, height - 70 + i * 22), font, 0.5, (150, 150, 150), 1)
    return np.hstack([frame, panel])


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def _is_auto_yellow(crop) -> bool:  # noqa: ANN001
    """Whether a bus/truck box is mostly the yellow of an Indian autorickshaw."""
    import cv2  # noqa: PLC0415

    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yellow = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 38) & (hsv[..., 1] > 60) & (hsv[..., 2] > 60)
    return float(yellow.mean()) > AUTO_YELLOW_SHARE


def _light_colour(crop) -> str | None:  # noqa: ANN001
    """``red``, ``amber`` or ``green`` for the brightest lamp in a traffic-light crop."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lit = (hsv[..., 1] > 90) & (hsv[..., 2] > 150)
    if lit.sum() < max(4, 0.01 * lit.size):
        return None
    hue = hsv[..., 0][lit]
    counts = {
        "red": int(np.sum((hue < 10) | (hue > 165))),
        "amber": int(np.sum((hue >= 10) & (hue < 35))),
        "green": int(np.sum((hue >= 45) & (hue < 95))),
    }
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] > 0 else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live YOLOv8 detection and driving decisions.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--image", help="Detect on this image file instead of the camera.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum confidence.")
    parser.add_argument("--model", choices=("n", "s", "m"), default="m", help="YOLOv8 size.")
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="Inference size; 1280 finds far signals."
    )
    parser.add_argument("--all-classes", action="store_true", help="Show all 80 COCO classes.")
    parser.add_argument("--sentry", action="store_true", help="Use the mission's own detector.")
    parser.add_argument("--cpu", action="store_true", help="Run on the CPU even with a GPU.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
