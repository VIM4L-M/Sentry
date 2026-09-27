#!/usr/bin/env python3
"""Scores the denoiser, and whether it helps the detector (milestone M4).

Two questions, answered on the held-out validation missions:

1. **Does it clean frames?** PSNR of the corrupted input and of the
   denoiser's output against the clean original, at each severity.
2. **Does that help detection?** Given detector weights, mAP50 and victim
   recall on the corrupted frames and on the denoised ones, at each
   severity. This is the M4 claim — "measurably improves detection under
   injected noise" — and it is the one that matters. A denoiser can raise
   PSNR by smoothing, and smoothing is exactly what erases an 8 px victim.

Severity is a multiple of ``sensors.degradation``: ``1.0`` is what the
detector was trained on, ``2.0`` is twice the smoke, blur and noise. Every
frame keeps the same smoke pattern across severities, so rows differ by
strength alone.

The detector that scores the denoised frames can differ from the one that
scores the corrupted frames. That is the fair comparison: each pipeline gets
the detector trained on its own input. A detector trained on smoky frames is
at a disadvantage on denoised ones, and ``--denoised-detector`` removes it —
see ``scripts/build_denoised_dataset.py``.

Usage:
    python scripts/evaluate_denoiser.py
    python scripts/evaluate_denoiser.py --detector models/yolo/labelfix/weights/best.pt
    python scripts/evaluate_denoiser.py --detector models/yolo/labelfix/weights/best.pt \
        --denoised-detector models/yolo/denoised/weights/best.pt
    python scripts/evaluate_denoiser.py --severities 0 1 1.5 2 2.5 --device cuda
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import DegradationConfig, YoloTrainingConfig
from sentry_ai.perception.autoencoder import ConvDenoisingAutoencoder
from sentry_ai.training.dataset import DatasetLayout, read_png
from sentry_ai.training.denoising import (
    SeverityResult,
    corrupt_frames,
    evaluate_reconstruction,
    write_detection_split,
)
from sentry_ai.training.yolo import EvaluationReport, YoloEvaluator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)

#: One validation frame: file stem, clean RGB pixels, YOLO label text.
Sample = tuple[str, NDArray[np.uint8], str]


def main() -> int:
    """Score reconstruction, then — given detector weights — detection."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    setup_logging(loader.resolve("configs/logging.yaml"))

    autoencoder_config = loader.load_autoencoder_config(args.config)
    degradation = loader.load_sensor_config(args.sensors).degradation
    weights = (
        loader.resolve(args.weights)
        if args.weights
        else autoencoder_config.runs_dir / args.run / "best.pt"
    )
    denoiser = ConvDenoisingAutoencoder.from_checkpoint(weights, device=args.device)
    samples = _validation_samples(DatasetLayout(autoencoder_config.dataset_dir), args.limit)

    print(f"Denoiser {weights} on {len(samples)} validation frame(s)\n")
    reconstruction = evaluate_reconstruction(
        denoiser, [pixels for _, pixels, _ in samples], degradation, args.severities, args.seed
    )
    _print_reconstruction(reconstruction)

    if args.detector is None:
        print("\nPass --detector <weights> to measure the effect on detection (M4).")
        return 0

    yolo_config = replace(loader.load_yolo_config(args.yolo_config), device=args.device)
    detector = loader.resolve(args.detector)
    denoised_detector = (
        loader.resolve(args.denoised_detector) if args.denoised_detector else detector
    )
    _compare_detection(
        denoiser,
        samples,
        degradation,
        yolo_config,
        (detector, denoised_detector),
        autoencoder_config.runs_dir / args.run / "detection_eval",
        args,
    )
    return 0


def _validation_samples(layout: DatasetLayout, limit: int | None) -> list[Sample]:
    """The clean validation frames and their labels.

    Clean rather than the stored degraded ones, because every severity is
    produced fresh from the clean original.
    """
    paths = sorted(layout.clean("val").glob("*.png"))
    if not paths:
        raise AssetNotFoundError(
            f"No clean validation frames in {layout.clean('val')}. "
            f"Run scripts/build_dataset.py first."
        )
    if limit is not None:
        # Evenly spaced rather than the first N, which would all be one
        # mission and one camera.
        paths = [paths[int(i)] for i in np.linspace(0, len(paths) - 1, min(limit, len(paths)))]
    samples: list[Sample] = []
    for path in paths:
        label_path = layout.labels("val") / f"{path.stem}.txt"
        labels = label_path.read_text(encoding="utf-8") if label_path.is_file() else ""
        samples.append((path.stem, read_png(path), labels))
    return samples


def _compare_detection(
    denoiser: ConvDenoisingAutoencoder,
    samples: list[Sample],
    degradation: DegradationConfig,
    yolo_config: YoloTrainingConfig,
    detectors: tuple[Path, Path],
    work_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Score corrupted frames with the first detector, denoised with the second."""
    stems = [stem for stem, _, _ in samples]
    labels = [text for _, _, text in samples]
    clean = [pixels for _, pixels, _ in samples]

    rows: list[tuple[float, EvaluationReport, EvaluationReport]] = []
    for severity in args.severities:
        corrupted = corrupt_frames(clean, degradation, severity, args.seed)
        denoised = [denoiser.denoise(frame) for frame in corrupted]
        reports = []
        arms = (("corrupted", corrupted, detectors[0]), ("denoised", denoised, detectors[1]))
        for arm, frames, weights in arms:
            root = work_dir / f"s{severity:g}" / arm
            write_detection_split(root, zip(stems, frames, labels, strict=True))
            evaluator = YoloEvaluator(replace(yolo_config, dataset_dir=root))
            reports.append(evaluator.evaluate(weights, run_name=f"eval_s{severity:g}_{arm}"))
        rows.append((severity, reports[0], reports[1]))

    _print_detection(rows, detectors)


def _print_reconstruction(results: list[SeverityResult]) -> None:
    print("Reconstruction (PSNR vs clean, higher is better)")
    print(f"  {'severity':>8}  {'input':>9}  {'denoised':>9}  {'gain':>8}")
    for result in results:
        print(
            f"  {result.severity:>8g}  {_db(result.psnr_degraded):>9}  "
            f"{_db(result.psnr_denoised):>9}  {_gain(result.gain):>8}"
        )


def _print_detection(
    rows: list[tuple[float, EvaluationReport, EvaluationReport]], detectors: tuple[Path, Path]
) -> None:
    same = detectors[0] == detectors[1]
    print(
        "\nDetection (same detector, corrupted input vs denoised input)"
        if same
        else f"\nDetection (corrupted -> {detectors[0]}, denoised -> {detectors[1]})"
    )
    print(
        f"  {'severity':>8}  {'mAP50':>7} -> {'mAP50':<7}  "
        f"{'victim recall':>13} -> {'':<6}"
    )
    for severity, corrupted, denoised in rows:
        print(
            f"  {severity:>8g}  {corrupted.map50:>7.4f} -> {denoised.map50:<7.4f}  "
            f"{_victim_recall(corrupted):>13} -> {_victim_recall(denoised):<6}"
        )
    print(
        "\nM4 holds where the right-hand numbers beat the left at a severity above "
        "1.0 — the detector was trained at 1.0, so that is where it needs help."
    )


def _victim_recall(report: EvaluationReport) -> str:
    victim = report.metrics_for("victim")
    return "n/a" if victim is None else f"{victim.recall:.4f}"


def _gain(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:+.2f}"


def _db(value: float) -> str:
    return "clean" if math.isinf(value) else f"{value:.2f} dB"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the denoising autoencoder.")
    parser.add_argument(
        "--config",
        default="configs/training/autoencoder.yaml",
        help="Autoencoder config (default: configs/training/autoencoder.yaml).",
    )
    parser.add_argument(
        "--sensors", default="configs/sensors.yaml", help="Sensors config (degradation)."
    )
    parser.add_argument("--weights", help="Denoiser checkpoint (default: the run's best.pt).")
    parser.add_argument("--run", default="sentry", help="Training run name to score.")
    parser.add_argument(
        "--severities",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0],
        help="Corruption multiples to test (default: 1.0 1.5 2.0).",
    )
    parser.add_argument(
        "--detector", help="YOLO weights. When given, also compares detection with/without."
    )
    parser.add_argument(
        "--denoised-detector",
        help="YOLO weights for the denoised arm (default: the same as --detector).",
    )
    parser.add_argument(
        "--yolo-config",
        default="configs/training/yolo.yaml",
        help="YOLO config, for image size (default: configs/training/yolo.yaml).",
    )
    parser.add_argument("--device", default="cpu", help="Inference device: cpu, cuda, index.")
    parser.add_argument("--limit", type=int, help="Score only this many frames, evenly spaced.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the corruption.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
