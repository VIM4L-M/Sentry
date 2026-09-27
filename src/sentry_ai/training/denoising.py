"""Trains and scores the denoising autoencoder (Unit IV).

**Training pairs are made on the fly, not read from disk.** The dataset
already holds a degraded copy of every frame, but only at one corruption
level with one fixed smoke pattern per frame. A network trained on that
learns to undo *that* smoke. Instead, every training sample is a random
crop of a clean frame, flipped or rotated, then corrupted by a
:class:`~sentry_ai.sensors.degradation.FrameDegrader` at a random severity.
Every epoch sees new smoke, and the severity range reaches past what the
dataset was captured at, so the denoiser keeps working when a fire makes
things worse.

**Validation uses the stored pairs.** ``images/val`` holds exactly the
degraded frames the detector is scored on, so validating against those
measures the denoiser on the pipeline's real input, not on its own training
distribution. Validation missions are held out whole, as for the detector,
so no validation frame's scene was ever trained on.

Every number this module reports is PSNR, and always alongside the
**baseline**: the PSNR of the degraded frame itself. "31 dB" means nothing
on its own; "31 dB, up from 18" is the result.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.schema import AutoencoderTrainingConfig, DegradationConfig
from sentry_ai.interfaces.perception import IDenoiser
from sentry_ai.perception.autoencoder import (
    AutoencoderArchitecture,
    DenoisingAutoencoderNet,
    to_tensor,
)
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.frame import YOLO_CLASSES
from sentry_ai.training.checkpoint import save_checkpoint
from sentry_ai.training.dataset import DatasetLayout, read_png, write_png, write_text
from sentry_ai.training.metrics import CsvMetricLogger
from sentry_ai.training.seed import resolve_device, seed_everything

logger = get_logger(__name__)

#: A training sample: (corrupted, clean), both ``(3, h, w)`` floats in 0-1.
TensorPair = tuple[torch.Tensor, torch.Tensor]

#: Called after every epoch with that epoch's metric row.
EpochCallback = Callable[[dict[str, float]], None]


class CorruptedCropDataset(Dataset[TensorPair]):
    """Random crops of clean frames, corrupted fresh every epoch.

    Deterministic despite the randomness: sample ``i`` in epoch ``e`` is
    drawn from a generator seeded with ``(seed, e, i)``, so a run replays
    exactly, and the result does not depend on how many DataLoader workers
    there are or which one fetched the sample.
    """

    def __init__(
        self,
        clean_paths: Sequence[Path],
        degradation: DegradationConfig,
        crop_size: int,
        severity_range: tuple[float, float],
        seed: int,
    ) -> None:
        """Create the dataset.

        Args:
            clean_paths: Uncorrupted frames to crop from.
            degradation: The shipped corruption, scaled per sample by a
                severity drawn from ``severity_range``.
            crop_size: Side of each square crop, in pixels.
            severity_range: ``(min, max)`` corruption multiplier.
            seed: Base seed for every crop and corruption.

        Raises:
            ValueError: If ``clean_paths`` is empty.
        """
        if not clean_paths:
            raise ValueError("CorruptedCropDataset needs at least one clean frame")
        self._paths = list(clean_paths)
        self._degradation = degradation
        self._crop_size = crop_size
        self._severity_range = severity_range
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch whose crops and corruption to produce."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> TensorPair:
        rng = np.random.default_rng((self._seed, self._epoch, index))
        clean = _augment(_random_crop(read_png(self._paths[index]), self._crop_size, rng), rng)
        severity = float(rng.uniform(*self._severity_range))
        corrupted = FrameDegrader(self._degradation.scaled(severity), rng).degrade_pixels(clean)
        return to_tensor(corrupted), to_tensor(clean)


@dataclass(frozen=True)
class DenoiserTrainingOutcome:
    """What a finished training run produced.

    Attributes:
        weights_path: The best checkpoint, ready for ``ConvDenoisingAutoencoder``.
        run_dir: Holds ``best.pt``, ``last.pt``, their metadata, and ``metrics.csv``.
        epochs_run: Epochs actually trained (fewer than configured if it
            stopped early).
        best_epoch: The epoch ``weights_path`` was saved at.
        best_val_loss: Validation loss at ``best_epoch``.
        val_psnr: Validation PSNR of the denoised frames at ``best_epoch``, in dB.
        baseline_psnr: Validation PSNR of the degraded frames themselves.
        parameters: Trainable parameter count.
    """

    weights_path: Path
    run_dir: Path
    epochs_run: int
    best_epoch: int
    best_val_loss: float
    val_psnr: float
    baseline_psnr: float
    parameters: int

    def as_display_rows(self) -> list[tuple[str, str]]:
        """Label/value pairs for direct printing."""
        return [
            ("epochs", f"{self.epochs_run} (best {self.best_epoch})"),
            ("val loss", f"{self.best_val_loss:.5f}"),
            ("PSNR input", f"{self.baseline_psnr:.2f} dB"),
            ("PSNR output", f"{self.val_psnr:.2f} dB"),
            ("gain", f"{self.val_psnr - self.baseline_psnr:+.2f} dB"),
            ("parameters", f"{self.parameters:,}"),
            ("weights", str(self.weights_path)),
        ]


class AutoencoderTrainer:
    """Runs one training job and reports where the best weights landed."""

    def __init__(self, config: AutoencoderTrainingConfig, degradation: DegradationConfig) -> None:
        """Create a trainer.

        Args:
            config: Architecture and hyperparameters.
            degradation: The shipped corruption from ``sensors.yaml`` — the
                severity-1.0 point training samples are scaled around.
        """
        self._config = config
        self._degradation = degradation
        self._layout = DatasetLayout(root=config.dataset_dir)

    def resolve_device(self) -> str:
        """The device to train on. Public so a script can warn before a CPU run."""
        return resolve_device(self._config.device)

    def train(
        self, run_name: str = "sentry", on_epoch: EpochCallback | None = None
    ) -> DenoiserTrainingOutcome:
        """Train, validate every epoch, and keep the best checkpoint.

        Args:
            run_name: Subdirectory of ``runs_dir`` for this run's outputs.
            on_epoch: Optional progress hook, called with each epoch's metrics.

        Raises:
            AssetNotFoundError: If the dataset is missing or incomplete.
                Checked before anything expensive starts.
        """
        config = self._config
        train_paths = sorted(self._layout.clean("train").glob("*.png"))
        if not train_paths:
            raise AssetNotFoundError(
                f"No clean training frames in {self._layout.clean('train')}. "
                f"Run scripts/build_dataset.py first."
            )
        val_pairs = load_pairs(self._layout, "val")

        seed_everything(config.seed)
        device = torch.device(self.resolve_device())
        run_dir = config.runs_dir / run_name

        network = DenoisingAutoencoderNet(_architecture(config)).to(device)
        optimizer = torch.optim.Adam(
            network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        loss_fn = _loss(config.loss)
        crops = CorruptedCropDataset(
            train_paths,
            self._degradation,
            config.crop_size,
            (config.severity_min, config.severity_max),
            config.seed,
        )
        loader = DataLoader(
            crops,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            generator=torch.Generator().manual_seed(config.seed),
            pin_memory=device.type == "cuda",
        )
        baseline = mean_psnr((degraded, clean) for degraded, clean in val_pairs)
        logger.info(
            "Training denoiser: %d parameters, %d train frames, %d val pairs, "
            "input PSNR %.2f dB, on %s",
            network.parameter_count(),
            len(train_paths),
            len(val_pairs),
            baseline,
            device,
        )

        metrics = CsvMetricLogger(run_dir / "metrics.csv")
        best = _Best()
        epoch = 0
        for epoch in range(1, config.epochs + 1):
            crops.set_epoch(epoch)
            train_loss = _train_epoch(network, loader, optimizer, loss_fn, device)
            val_loss, val_psnr = _validate(network, val_pairs, loss_fn, device)
            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "baseline_psnr": baseline,
            }
            metrics.log(row)
            if on_epoch is not None:
                on_epoch(row)

            scores = {"val_loss": val_loss, "val_psnr": val_psnr, "baseline_psnr": baseline}
            payload = network.to_checkpoint({"epoch": epoch, **scores})
            save_checkpoint(run_dir / "last.pt", payload, config, scores)
            if val_loss < best.loss:
                best = _Best(epoch=epoch, loss=val_loss, psnr=val_psnr)
                save_checkpoint(run_dir / "best.pt", payload, config, scores)
            elif config.patience and epoch - best.epoch >= config.patience:
                logger.info("No improvement for %d epochs — stopping", config.patience)
                break

        return DenoiserTrainingOutcome(
            weights_path=run_dir / "best.pt",
            run_dir=run_dir,
            epochs_run=epoch,
            best_epoch=best.epoch,
            best_val_loss=best.loss,
            val_psnr=best.psnr,
            baseline_psnr=baseline,
            parameters=network.parameter_count(),
        )


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SeverityResult:
    """How the denoiser did at one corruption level.

    Attributes:
        severity: Multiple of the shipped ``sensors.degradation``.
        frames: Frames scored.
        psnr_degraded: Mean PSNR of the corrupted input, in dB — the baseline.
        psnr_denoised: Mean PSNR of the denoiser's output, in dB.
    """

    severity: float
    frames: int
    psnr_degraded: float
    psnr_denoised: float

    @property
    def gain(self) -> float:
        """How many dB the denoiser added. Negative means it made things worse.

        ``nan`` at severity 0, where the input is the clean frame itself and
        there is nothing to gain — ``inf`` minus anything is not a number
        worth printing.
        """
        if math.isinf(self.psnr_degraded):
            return math.nan
        return self.psnr_denoised - self.psnr_degraded


def evaluate_reconstruction(
    denoiser: IDenoiser,
    clean_frames: Sequence[NDArray[np.uint8]],
    degradation: DegradationConfig,
    severities: Iterable[float],
    seed: int = 0,
) -> list[SeverityResult]:
    """Corrupt each clean frame at each severity and score the denoiser on it.

    Works through the :class:`IDenoiser` port, so it scores whatever the
    pipeline would actually hold. Frame ``i`` is corrupted with a generator
    seeded ``(seed, i)`` at every severity: the smoke pattern stays put and
    only its strength changes, so rows differ by severity and nothing else.

    Raises:
        ValueError: If ``clean_frames`` is empty.
    """
    if not clean_frames:
        raise ValueError("evaluate_reconstruction needs at least one frame")
    results: list[SeverityResult] = []
    for severity in severities:
        degraded = corrupt_frames(clean_frames, degradation, severity, seed)
        denoised = [denoiser.denoise(frame) for frame in degraded]
        results.append(
            SeverityResult(
                severity=severity,
                frames=len(clean_frames),
                psnr_degraded=mean_psnr(zip(degraded, clean_frames, strict=True)),
                psnr_denoised=mean_psnr(zip(denoised, clean_frames, strict=True)),
            )
        )
    return results


def corrupt_frames(
    clean_frames: Sequence[NDArray[np.uint8]],
    degradation: DegradationConfig,
    severity: float,
    seed: int,
) -> list[NDArray[np.uint8]]:
    """Corrupt every frame at ``severity``; frame ``i`` always gets seed ``(seed, i)``."""
    scaled = degradation.scaled(severity)
    return [
        FrameDegrader(scaled, np.random.default_rng((seed, index))).degrade_pixels(frame)
        for index, frame in enumerate(clean_frames)
    ]


def write_detection_split(
    root: Path, samples: Iterable[tuple[str, NDArray[np.uint8], str]]
) -> Path:
    """Write frames and their YOLO labels as a validation-only dataset.

    Used to score the detector on denoised frames: ``YoloEvaluator`` reads a
    dataset from disk, so the frames it is to be scored on have to be put
    there. Anything this function previously wrote under ``root`` is
    removed first, so a smaller rerun cannot inherit stale frames.

    Args:
        root: Directory to write into.
        samples: ``(stem, rgb_pixels, label_text)`` per frame.

    Returns:
        The ``data.yaml`` to hand to the evaluator.
    """
    layout = DatasetLayout(root=root)
    for directory in layout.owned_directories():
        if directory.exists():
            shutil.rmtree(directory)
    for stem, pixels, labels in samples:
        write_png(layout.images("val") / f"{stem}.png", pixels)
        write_text(layout.labels("val") / f"{stem}.txt", labels.strip())
    document = {
        "path": str(root.resolve()),
        # Ultralytics insists on a train entry even for validation; pointing
        # it at the validation images means nothing is ever trained on here.
        "train": "images/val",
        "val": "images/val",
        "names": {index: kind.value for index, kind in enumerate(YOLO_CLASSES)},
    }
    write_text(layout.data_yaml, yaml.safe_dump(document, sort_keys=False).strip())
    return layout.data_yaml


@dataclass(frozen=True)
class DenoisedDatasetStats:
    """What :func:`write_denoised_dataset` wrote, per split."""

    frames: dict[str, int]

    def as_display_rows(self) -> list[tuple[str, str]]:
        """Label/value pairs for direct printing."""
        return [(split, str(count)) for split, count in self.frames.items()]


def write_denoised_dataset(
    source: DatasetLayout,
    target: DatasetLayout,
    denoiser: IDenoiser,
    splits: Sequence[str] = ("train", "val"),
) -> DenoisedDatasetStats:
    """Copy a detection dataset with every degraded frame run through ``denoiser``.

    This is how the detector is retrained on what it will actually be shown
    once the denoiser is in the pipeline. A detector trained on smoky frames
    has learned what a victim looks like *through smoke*; handing it a
    denoised frame is a distribution shift, and measured, it costs more than
    the denoiser gains. Training on denoised frames removes the shift.

    Labels are copied unchanged — denoising moves pixels, never objects.
    Clean frames are copied too, so the result is a complete dataset that
    ``train_autoencoder.py`` could read as well. The split is inherited,
    whole missions and all, so validation stays held out.

    Raises:
        AssetNotFoundError: If a split has no degraded frames.
    """
    for directory in target.owned_directories():
        if directory.exists():
            shutil.rmtree(directory)
    counts: dict[str, int] = {}
    for split in splits:
        degraded_paths = sorted(source.images(split).glob("*.png"))
        if not degraded_paths:
            raise AssetNotFoundError(f"No degraded frames in {source.images(split)}")
        for path in degraded_paths:
            write_png(target.images(split) / path.name, denoiser.denoise(read_png(path)))
            _copy_if_present(source.labels(split) / f"{path.stem}.txt", target.labels(split))
            _copy_if_present(source.clean(split) / path.name, target.clean(split))
        counts[split] = len(degraded_paths)
    document = {
        "path": str(target.root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: kind.value for index, kind in enumerate(YOLO_CLASSES)},
    }
    write_text(target.data_yaml, yaml.safe_dump(document, sort_keys=False).strip())
    return DenoisedDatasetStats(frames=counts)


def load_pairs(
    layout: DatasetLayout, split: str
) -> list[tuple[NDArray[np.uint8], NDArray[np.uint8]]]:
    """Every ``(degraded, clean)`` frame pair in ``split``, matched by file name.

    Raises:
        AssetNotFoundError: If the split has no degraded frames, or one has
            no clean counterpart — a half-written dataset should fail here,
            not score as if it were complete.
    """
    degraded_paths = sorted(layout.images(split).glob("*.png"))
    if not degraded_paths:
        raise AssetNotFoundError(
            f"No degraded frames in {layout.images(split)}. Run scripts/build_dataset.py first."
        )
    pairs: list[tuple[NDArray[np.uint8], NDArray[np.uint8]]] = []
    for degraded_path in degraded_paths:
        clean_path = layout.clean(split) / degraded_path.name
        if not clean_path.is_file():
            raise AssetNotFoundError(f"No clean frame paired with {degraded_path}")
        pairs.append((read_png(degraded_path), read_png(clean_path)))
    return pairs


def psnr(image: NDArray[np.uint8], reference: NDArray[np.uint8]) -> float:
    """Peak signal-to-noise ratio of ``image`` against ``reference``, in dB.

    ``inf`` for identical images. Higher is better; each +3 dB roughly
    halves the squared error.

    Raises:
        ValueError: If the shapes differ.
    """
    if image.shape != reference.shape:
        raise ValueError(f"psnr needs equal shapes, got {image.shape} and {reference.shape}")
    difference = image.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(difference**2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(255.0**2 / mse)


def mean_psnr(pairs: Iterable[tuple[NDArray[np.uint8], NDArray[np.uint8]]]) -> float:
    """Average PSNR over ``(image, reference)`` pairs.

    Averaged in dB, per frame, which is the convention. Identical pairs are
    skipped rather than letting one ``inf`` swamp the mean; if *every* pair
    is identical the answer really is ``inf``.
    """
    scores = [psnr(image, reference) for image, reference in pairs]
    finite = [score for score in scores if math.isfinite(score)]
    if not scores:
        raise ValueError("mean_psnr needs at least one pair")
    return sum(finite) / len(finite) if finite else math.inf


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Best:
    """The best validation result so far."""

    epoch: int = 0
    loss: float = math.inf
    psnr: float = 0.0


def _copy_if_present(source: Path, directory: Path) -> None:
    """Copy ``source`` into ``directory`` if it exists; a background frame has no label."""
    if source.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, directory / source.name)


def _architecture(config: AutoencoderTrainingConfig) -> AutoencoderArchitecture:
    return AutoencoderArchitecture(
        base_channels=config.base_channels,
        depth=config.depth,
        skip_connections=config.skip_connections,
    )


def _loss(name: str) -> nn.Module:
    """The reconstruction loss the config names. Validated by the config already."""
    return nn.L1Loss() if name == "l1" else nn.MSELoss()


def _train_epoch(
    network: DenoisingAutoencoderNet,
    loader: DataLoader[TensorPair],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """One pass over the training crops; returns the mean loss per sample."""
    network.train()
    total, count = 0.0, 0
    for corrupted, clean in loader:
        corrupted, clean = corrupted.to(device), clean.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(network(corrupted), clean)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * corrupted.shape[0]
        count += corrupted.shape[0]
    return total / count


def _validate(
    network: DenoisingAutoencoderNet,
    pairs: Sequence[tuple[NDArray[np.uint8], NDArray[np.uint8]]],
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Mean loss and mean PSNR over whole validation frames, one at a time.

    One at a time because the frames are not all one size — the onboard
    view is square, the CCTV views are not.
    """
    network.eval()
    losses: list[float] = []
    scores: list[float] = []
    with torch.inference_mode():
        for degraded, clean in pairs:
            target = to_tensor(clean).unsqueeze(0).to(device)
            output = network(to_tensor(degraded).unsqueeze(0).to(device))
            losses.append(float(loss_fn(output, target).item()))
            mse = float(torch.mean((output - target) ** 2).item())
            scores.append(math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse))
    finite = [score for score in scores if math.isfinite(score)]
    return sum(losses) / len(losses), (sum(finite) / len(finite) if finite else math.inf)


def _random_crop(
    image: NDArray[np.uint8], size: int, rng: np.random.Generator
) -> NDArray[np.uint8]:
    """A ``size`` x ``size`` window from a random position in ``image``.

    Raises:
        ValueError: If the image is smaller than the crop.
    """
    height, width = image.shape[0], image.shape[1]
    if height < size or width < size:
        raise ValueError(f"cannot take a {size}px crop from a {width}x{height} frame")
    top = int(rng.integers(0, height - size + 1))
    left = int(rng.integers(0, width - size + 1))
    return image[top : top + size, left : left + size]


def _augment(image: NDArray[np.uint8], rng: np.random.Generator) -> NDArray[np.uint8]:
    """Random quarter-turn and mirror — all eight symmetries of a square.

    Legal because the city is seen from directly overhead: a rotated or
    mirrored crop is another plausible piece of city, the same argument
    that lets the detector train with vertical flips.
    """
    rotated = np.rot90(image, k=int(rng.integers(0, 4)))
    if rng.random() < 0.5:
        rotated = rotated[:, ::-1]
    return np.ascontiguousarray(rotated)
