"""Fine-tunes YOLOv8n on the synthetic dataset (Unit II).

Thin on purpose. Ultralytics already owns the training loop, the
augmentation pipeline, and the metric reporting; re-implementing any of
that would be worse code doing the same job. What this module adds is the
project's own contract around it: a validated config in, a
:class:`TrainingOutcome` out, and one place that decides which device to
use.

Transfer learning, not training from scratch: the COCO-pretrained nano
checkpoint already knows edges, blobs, and small-object structure, and the
synthetic dataset here is far too small to learn those from nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.schema import YoloTrainingConfig

logger = get_logger(__name__)

#: Metrics pulled out of Ultralytics' results for reporting. mAP50-95 is the
#: headline number; the per-class breakdown matters more here because the
#: classes are badly unbalanced and victims are both the rarest and the ones
#: that matter.
_REPORTED_METRICS: dict[str, str] = {
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
}


@dataclass(frozen=True)
class TrainingOutcome:
    """What a finished training run produced.

    Attributes:
        weights_path: The best checkpoint, ready for ``YoloDetector``.
        metrics: Validation metrics, keyed by short name.
        run_dir: Directory holding curves, confusion matrix, and samples.
    """

    weights_path: Path
    metrics: dict[str, float]
    run_dir: Path

    def as_display_rows(self) -> list[tuple[str, str]]:
        """Label/value pairs for direct printing."""
        rows = [(name, f"{value:.4f}") for name, value in self.metrics.items()]
        rows.append(("weights", str(self.weights_path)))
        return rows


class YoloTrainer:
    """Runs one fine-tuning job and reports where the weights landed."""

    def __init__(self, config: YoloTrainingConfig) -> None:
        """Create a trainer.

        Args:
            config: Validated hyperparameters.
        """
        self._config = config

    def train(self, run_name: str = "sentry") -> TrainingOutcome:
        """Fine-tune the detector and return where the weights landed.

        Args:
            run_name: Subdirectory of ``runs_dir`` for this run's outputs.

        Raises:
            AssetNotFoundError: If the dataset or the pretrained checkpoint
                is missing. Both are checked before anything expensive
                starts.
        """
        data_yaml = self._config.dataset_dir / "data.yaml"
        if not data_yaml.is_file():
            raise AssetNotFoundError(
                f"Dataset descriptor not found: {data_yaml}. "
                f"Run scripts/build_dataset.py first."
            )
        if not self._config.pretrained_weights.is_file():
            raise AssetNotFoundError(
                f"Pretrained checkpoint not found: {self._config.pretrained_weights}. "
                f"Run scripts/fetch_pretrained.py first."
            )

        from ultralytics import YOLO  # noqa: PLC0415 - keeps Torch off the import path

        device = self.resolve_device()
        logger.info(
            "Fine-tuning %s for %d epochs at %dpx on %s",
            self._config.pretrained_weights,
            self._config.epochs,
            self._config.image_size,
            device,
        )
        model = YOLO(str(self._config.pretrained_weights))
        results = model.train(**self._train_arguments(data_yaml, device, run_name))
        return self._outcome(model, results, run_name)

    def resolve_device(self) -> str:
        """The device to train on, resolving ``"auto"`` against what exists.

        Public because the composition root reports it before a long run
        starts — discovering an hour in that it fell back to CPU is a poor
        way to find out.
        """
        if self._config.device != "auto":
            return self._config.device

        import torch  # noqa: PLC0415 - only needed to answer this question

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _train_arguments(self, data_yaml: Path, device: str, run_name: str) -> dict[str, Any]:
        """Translate this project's config into Ultralytics' keyword names."""
        config = self._config
        return {
            "data": str(data_yaml),
            "epochs": config.epochs,
            "imgsz": config.image_size,
            "batch": config.batch_size,
            "patience": config.patience,
            "device": device,
            "seed": config.seed,
            "project": str(config.runs_dir),
            "name": run_name,
            "exist_ok": True,
            "verbose": True,
            "fliplr": config.horizontal_flip,
            "flipud": config.vertical_flip,
            "mosaic": config.mosaic,
            "scale": config.scale,
            "translate": config.translate,
            "hsv_v": config.hsv_value,
            # Hue and saturation jitter are switched off: class identity here
            # is largely *carried* by colour (fire is orange, victims are
            # pink), so shifting hue would actively destroy the label.
            "hsv_h": 0.0,
            "hsv_s": 0.0,
        }

    def _outcome(self, model: Any, results: Any, run_name: str) -> TrainingOutcome:
        """Collect weights path and validation metrics from a finished run."""
        run_dir = self._config.runs_dir / run_name
        weights = getattr(getattr(model, "trainer", None), "best", None)
        weights_path = Path(weights) if weights else run_dir / "weights" / "best.pt"

        raw = dict(getattr(results, "results_dict", {}) or {})
        metrics = {
            short: float(raw[key]) for key, short in _REPORTED_METRICS.items() if key in raw
        }
        return TrainingOutcome(weights_path=weights_path, metrics=metrics, run_dir=run_dir)
