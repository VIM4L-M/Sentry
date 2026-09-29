#!/usr/bin/env python3
"""Live object detection from the laptop's camera, or on any image (review demo).

For "show it something": point the webcam at a person, a phone, a toy car, a
printed photo of a street — or pass an image file — and YOLOv8 draws what it
finds, with its confidence, in real time. The model is the stock COCO
YOLOv8n (``models/pretrained/yolov8n.pt``), the same one that reads the real
street photos in the command center: 80 everyday classes, including person,
car, motorcycle, bus, truck, bicycle and traffic light.

The mission's own detector (``--sentry``) is the one trained on the simulated
city's top-down camera frames. It is shown for completeness; on real photos
it is expected to do badly, which is itself a point worth making — it knows
the world it was trained on.

Keys: ``q`` or Escape quits, ``s`` saves a snapshot to ``runs/live/``.

Usage:
    python scripts/live_camera.py                       # webcam 0
    python scripts/live_camera.py --camera 1            # another camera
    python scripts/live_camera.py --image street.jpg    # one image, window + saved copy
    python scripts/live_camera.py --sentry              # the mission's own detector
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The stock COCO model, and the mission's own detector.
COCO_WEIGHTS = "models/pretrained/yolov8n.pt"
SENTRY_WEIGHTS = "models/yolo/labelfix/weights/best.pt"

#: Where snapshots and annotated images are written.
OUTPUT_DIR = "runs/live"


def main() -> int:
    """Open the camera (or image) and show detections until the user quits."""
    args = _parse_args()
    import cv2  # noqa: PLC0415 - comes with ultralytics
    import torch  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    weights = PROJECT_ROOT / (SENTRY_WEIGHTS if args.sentry else COCO_WEIGHTS)
    if not weights.is_file():
        raise SystemExit(f"weights not found: {weights}")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = YOLO(str(weights))
    output = PROJECT_ROOT / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    source = "mission detector" if args.sentry else "YOLOv8 COCO"
    title = f"SENTRY AI - live detection ({source})"

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"cannot read image: {args.image}")
        annotated = _annotate(model, frame, device, args.confidence)
        path = output / f"detected_{Path(args.image).stem}.jpg"
        cv2.imwrite(str(path), annotated)
        print(f"saved {path.relative_to(PROJECT_ROOT)}")
        cv2.imshow(title, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"no camera at index {args.camera}; try --camera 1 or --image")
    try:
        _loop(cv2, capture, model, device, args.confidence, title, output)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def _loop(cv2, capture, model, device, confidence, title, output) -> None:  # noqa: ANN001
    last = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            print("camera stopped sending frames")
            return
        annotated = _annotate(model, frame, device, confidence)
        now = time.perf_counter()
        fps = 1.0 / max(1e-6, now - last)
        last = now
        cv2.putText(
            annotated, f"{fps:4.1f} fps  on {device}", (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 255), 2,
        )  # fmt: skip
        cv2.imshow(title, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return
        if key == ord("s"):
            path = output / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(path), annotated)
            print(f"saved {path.relative_to(PROJECT_ROOT)}")


def _annotate(model, frame, device: str, confidence: float):  # noqa: ANN001, ANN202
    """``frame`` with every detection boxed and labelled with its confidence."""
    result = model.predict(frame, conf=confidence, device=device, verbose=False)[0]
    return result.plot()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live YOLOv8 detection from a camera or image.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--image", help="Detect on this image file instead of the camera.")
    parser.add_argument("--confidence", type=float, default=0.35, help="Minimum confidence.")
    parser.add_argument("--sentry", action="store_true", help="Use the mission's own detector.")
    parser.add_argument("--cpu", action="store_true", help="Run on the CPU even with a GPU.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
