"""Unit tests for sentry_ai.training.denoising.

The trainer is exercised end to end on a dataset of a few tiny frames with a
network a few hundred parameters wide: enough to prove it reads the right
files, writes the right files, and produces a checkpoint the live adapter
can load — which is the contract. Whether a *real* run denoises well is a
training result, recorded in docs/architecture/phase4-denoising.md.
"""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("torch", reason="the trainer needs torch (requirements-ml.txt)")
pytest.importorskip("cv2", reason="the trainer reads frames with opencv (requirements-ml.txt)")

from sentry_ai.common.exceptions import AssetNotFoundError  # noqa: E402
from sentry_ai.config.schema import AutoencoderTrainingConfig, DegradationConfig  # noqa: E402
from sentry_ai.interfaces.perception import IDenoiser  # noqa: E402
from sentry_ai.perception.autoencoder import ConvDenoisingAutoencoder  # noqa: E402
from sentry_ai.training.checkpoint import load_metadata  # noqa: E402
from sentry_ai.training.dataset import DatasetLayout, read_png, write_png  # noqa: E402
from sentry_ai.training.denoising import (  # noqa: E402
    AutoencoderTrainer,
    CorruptedCropDataset,
    SeverityResult,
    corrupt_frames,
    evaluate_reconstruction,
    load_pairs,
    mean_psnr,
    psnr,
    write_denoised_dataset,
    write_detection_split,
)

_DEGRADATION = DegradationConfig(smoke_density=0.4, blur_radius=1, noise_std=10.0)


def _image(height: int = 24, width: int = 32, seed: int = 0) -> NDArray[np.uint8]:
    """Blocky rather than uniform noise, so crops and flips are distinguishable."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (height // 8 + 1, width // 8 + 1, 3), dtype=np.uint8)
    tiled = np.kron(blocks, np.ones((8, 8, 1), dtype=np.uint8))
    return np.ascontiguousarray(tiled[:height, :width])


def _dataset(root: Path, train: int = 3, val: int = 2) -> DatasetLayout:
    """A miniature build_dataset.py output: clean frames, degraded val frames."""
    layout = DatasetLayout(root=root)
    for index in range(train):
        write_png(layout.clean("train") / f"t{index}.png", _image(seed=index))
    for index in range(val):
        clean = _image(seed=100 + index)
        write_png(layout.clean("val") / f"v{index}.png", clean)
        corrupted = corrupt_frames([clean], _DEGRADATION, 1.0, index)[0]
        write_png(layout.images("val") / f"v{index}.png", corrupted)
    return layout


def _config(root: Path, **overrides: object) -> AutoencoderTrainingConfig:
    base = AutoencoderTrainingConfig(
        dataset_dir=root,
        runs_dir=root / "runs",
        base_channels=2,
        depth=2,
        crop_size=16,
        epochs=2,
        batch_size=2,
        num_workers=0,
        device="cpu",
        seed=3,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


class _Identity(IDenoiser):
    def denoise(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return frame.copy()


class _Oracle(IDenoiser):
    """Returns the clean frame it was told about — a perfect denoiser."""

    def __init__(self, clean: list[NDArray[np.uint8]], corrupted: list[NDArray[np.uint8]]) -> None:
        pairs = zip(corrupted, clean, strict=True)
        self._answers = {frame.tobytes(): answer for frame, answer in pairs}

    def denoise(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return self._answers[frame.tobytes()]


class TestPsnr:
    def test_identical_images_score_infinity(self) -> None:
        image = _image()
        assert psnr(image, image) == math.inf

    def test_a_known_error_gives_a_known_score(self) -> None:
        """A uniform error of 1 level is MSE 1: 20 * log10(255) = 48.13 dB."""
        reference = np.full((4, 4, 3), 100, dtype=np.uint8)
        assert psnr(reference + 1, reference) == pytest.approx(48.1308, abs=1e-3)

    def test_more_error_scores_lower(self) -> None:
        reference = np.full((4, 4, 3), 100, dtype=np.uint8)
        assert psnr(reference + 10, reference) < psnr(reference + 2, reference)

    def test_no_uint8_wraparound(self) -> None:
        """200 - 250 must be -50, not 206."""
        low = np.full((2, 2, 3), 200, dtype=np.uint8)
        high = np.full((2, 2, 3), 250, dtype=np.uint8)
        assert psnr(low, high) == pytest.approx(10 * math.log10(255**2 / 50**2))

    def test_mismatched_shapes_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            psnr(_image(8, 8), _image(8, 16))

    def test_the_mean_ignores_identical_pairs(self) -> None:
        reference = np.full((4, 4, 3), 100, dtype=np.uint8)
        pairs = [(reference, reference), (reference + 1, reference)]
        assert mean_psnr(pairs) == pytest.approx(psnr(reference + 1, reference))

    def test_the_mean_of_nothing_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            mean_psnr([])


class TestCorruptedCrops:
    def _crops(self, tmp_path: Path, **overrides: object) -> CorruptedCropDataset:
        paths = []
        for index in range(2):
            path = tmp_path / f"f{index}.png"
            write_png(path, _image(seed=index))
            paths.append(path)
        options: dict[str, object] = {
            "clean_paths": paths,
            "degradation": _DEGRADATION,
            "crop_size": 16,
            "severity_range": (0.5, 2.0),
            "seed": 1,
        }
        options.update(overrides)
        return CorruptedCropDataset(**options)  # type: ignore[arg-type]

    def test_samples_are_aligned_crops_in_range(self, tmp_path: Path) -> None:
        corrupted, clean = self._crops(tmp_path)[0]
        assert corrupted.shape == clean.shape == (3, 16, 16)
        assert float(clean.min()) >= 0.0 and float(clean.max()) <= 1.0

    def test_the_input_is_actually_corrupted(self, tmp_path: Path) -> None:
        corrupted, clean = self._crops(tmp_path)[0]
        assert not bool((corrupted == clean).all())

    def test_the_clean_half_is_a_real_piece_of_the_frame(self, tmp_path: Path) -> None:
        """No corruption leaks into the target — only crop, flip, rotate."""
        _, clean = self._crops(tmp_path, severity_range=(0.0, 0.0))[0]
        pixels = (clean.permute(1, 2, 0) * 255).round().int().reshape(-1, 3).tolist()
        values = {tuple(pixel) for pixel in pixels}
        frame = {tuple(pixel) for pixel in _image(seed=0).reshape(-1, 3).tolist()}
        assert values <= frame

    def test_the_same_epoch_replays_exactly(self, tmp_path: Path) -> None:
        crops = self._crops(tmp_path)
        crops.set_epoch(4)
        first = crops[1]
        crops.set_epoch(4)
        second = crops[1]
        assert bool((first[0] == second[0]).all()) and bool((first[1] == second[1]).all())

    def test_every_epoch_sees_new_crops(self, tmp_path: Path) -> None:
        crops = self._crops(tmp_path)
        crops.set_epoch(1)
        first = crops[0][0]
        crops.set_epoch(2)
        assert not bool((crops[0][0] == first).all())

    def test_a_crop_larger_than_the_frame_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="crop"):
            self._crops(tmp_path, crop_size=64)[0]

    def test_an_empty_dataset_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            self._crops(tmp_path, clean_paths=[])


class TestPairs:
    def test_pairs_are_matched_by_name(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        pairs = load_pairs(layout, "val")
        assert len(pairs) == 2
        assert np.array_equal(pairs[0][1], read_png(layout.clean("val") / "v0.png"))

    def test_a_missing_clean_frame_fails(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        (layout.clean("val") / "v1.png").unlink()
        with pytest.raises(AssetNotFoundError, match="v1.png"):
            load_pairs(layout, "val")

    def test_an_empty_split_fails(self, tmp_path: Path) -> None:
        with pytest.raises(AssetNotFoundError):
            load_pairs(DatasetLayout(root=tmp_path), "val")


class TestReconstructionEvaluation:
    def test_doing_nothing_gains_nothing(self) -> None:
        clean = [_image(seed=1), _image(seed=2)]
        (result,) = evaluate_reconstruction(_Identity(), clean, _DEGRADATION, [1.0])
        assert result.gain == pytest.approx(0.0)
        assert result.frames == 2

    def test_a_perfect_denoiser_scores_as_clean(self) -> None:
        clean = [_image(seed=1), _image(seed=2)]
        corrupted = corrupt_frames(clean, _DEGRADATION, 1.5, seed=0)
        (result,) = evaluate_reconstruction(_Oracle(clean, corrupted), clean, _DEGRADATION, [1.5])
        assert result.psnr_denoised == math.inf

    def test_heavier_corruption_scores_lower(self) -> None:
        clean = [_image(seed=1)]
        light, heavy = evaluate_reconstruction(_Identity(), clean, _DEGRADATION, [0.5, 2.0])
        assert heavy.psnr_degraded < light.psnr_degraded

    def test_corruption_is_reproducible(self) -> None:
        clean = [_image(seed=1)]
        first = corrupt_frames(clean, _DEGRADATION, 1.0, seed=9)
        assert np.array_equal(first[0], corrupt_frames(clean, _DEGRADATION, 1.0, seed=9)[0])

    def test_no_frames_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            evaluate_reconstruction(_Identity(), [], _DEGRADATION, [1.0])


class TestSeverityResult:
    def test_the_gain_is_output_minus_input(self) -> None:
        result = SeverityResult(severity=1.0, frames=1, psnr_degraded=20.0, psnr_denoised=26.5)
        assert result.gain == pytest.approx(6.5)

    def test_there_is_no_gain_to_report_on_a_clean_input(self) -> None:
        result = SeverityResult(severity=0.0, frames=1, psnr_degraded=math.inf, psnr_denoised=30.0)
        assert math.isnan(result.gain)


class _Inverter(IDenoiser):
    def denoise(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return 255 - frame


class TestDenoisedDataset:
    def _source(self, root: Path) -> DatasetLayout:
        layout = _dataset(root)
        for split in ("train", "val"):
            for clean_path in layout.clean(split).glob("*.png"):
                image_path = layout.images(split) / clean_path.name
                if not image_path.exists():
                    write_png(image_path, read_png(clean_path))
                label = layout.labels(split) / f"{clean_path.stem}.txt"
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text(f"0 0.5 0.5 0.1 0.1  # {clean_path.stem}\n")
        return layout

    def test_every_frame_goes_through_the_denoiser(self, tmp_path: Path) -> None:
        source = self._source(tmp_path / "src")
        target = DatasetLayout(tmp_path / "out")
        write_denoised_dataset(source, target, _Inverter())

        original = read_png(source.images("val") / "v0.png")
        assert np.array_equal(read_png(target.images("val") / "v0.png"), 255 - original)

    def test_labels_and_clean_frames_are_copied_unchanged(self, tmp_path: Path) -> None:
        source = self._source(tmp_path / "src")
        target = DatasetLayout(tmp_path / "out")
        write_denoised_dataset(source, target, _Inverter())

        label = (target.labels("train") / "t1.txt").read_text()
        assert label == (source.labels("train") / "t1.txt").read_text()
        assert np.array_equal(
            read_png(target.clean("val") / "v1.png"), read_png(source.clean("val") / "v1.png")
        )

    def test_the_split_is_inherited_whole(self, tmp_path: Path) -> None:
        source = self._source(tmp_path / "src")
        stats = write_denoised_dataset(source, DatasetLayout(tmp_path / "out"), _Inverter())
        assert stats.frames == {"train": 3, "val": 2}
        assert "val: images/val" in (tmp_path / "out" / "data.yaml").read_text()

    def test_a_split_with_no_frames_fails(self, tmp_path: Path) -> None:
        with pytest.raises(AssetNotFoundError):
            write_denoised_dataset(
                DatasetLayout(tmp_path / "empty"), DatasetLayout(tmp_path / "out"), _Inverter()
            )


class TestDetectionSplit:
    def test_frames_labels_and_descriptor_are_written(self, tmp_path: Path) -> None:
        data_yaml = write_detection_split(tmp_path, [("a", _image(), "0 0.5 0.5 0.1 0.1\n")])
        layout = DatasetLayout(root=tmp_path)

        assert data_yaml == layout.data_yaml
        assert np.array_equal(read_png(layout.images("val") / "a.png"), _image())
        assert (layout.labels("val") / "a.txt").read_text().strip() == "0 0.5 0.5 0.1 0.1"
        assert "val: images/val" in data_yaml.read_text()

    def test_a_rerun_leaves_no_stale_frames(self, tmp_path: Path) -> None:
        write_detection_split(tmp_path, [("a", _image(), ""), ("b", _image(), "")])
        write_detection_split(tmp_path, [("a", _image(), "")])
        assert [path.name for path in DatasetLayout(tmp_path).images("val").iterdir()] == ["a.png"]


class TestTrainer:
    def test_a_run_writes_loadable_weights_and_a_log(self, tmp_path: Path) -> None:
        _dataset(tmp_path)
        config = _config(tmp_path)
        rows: list[dict[str, float]] = []

        outcome = AutoencoderTrainer(config, _DEGRADATION).train("tiny", on_epoch=rows.append)

        assert outcome.epochs_run == 2
        assert [row["epoch"] for row in rows] == [1.0, 2.0]
        assert outcome.weights_path == tmp_path / "runs" / "tiny" / "best.pt"
        assert (outcome.run_dir / "last.pt").is_file()
        with (outcome.run_dir / "metrics.csv").open(newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 2

        cleaned = ConvDenoisingAutoencoder.from_checkpoint(outcome.weights_path).denoise(_image())
        assert cleaned.shape == _image().shape

    def test_the_baseline_is_the_degraded_frames_own_score(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        outcome = AutoencoderTrainer(_config(tmp_path, epochs=1), _DEGRADATION).train("tiny")
        assert outcome.baseline_psnr == pytest.approx(mean_psnr(load_pairs(layout, "val")))

    def test_metadata_records_the_best_epoch(self, tmp_path: Path) -> None:
        _dataset(tmp_path)
        outcome = AutoencoderTrainer(_config(tmp_path), _DEGRADATION).train("tiny")
        metadata = load_metadata(outcome.weights_path)
        assert metadata["metrics"]["val_loss"] == pytest.approx(outcome.best_val_loss)

    def test_the_same_seed_trains_the_same_network(self, tmp_path: Path) -> None:
        _dataset(tmp_path)
        config = _config(tmp_path, epochs=1)
        first = AutoencoderTrainer(config, _DEGRADATION).train("a")
        second = AutoencoderTrainer(config, _DEGRADATION).train("b")
        assert first.best_val_loss == second.best_val_loss

    def test_patience_stops_a_run_that_stopped_improving(self, tmp_path: Path) -> None:
        _dataset(tmp_path)
        config = _config(tmp_path, epochs=30, patience=1, learning_rate=1e-9)
        outcome = AutoencoderTrainer(config, _DEGRADATION).train("stalled")
        assert outcome.epochs_run < 30

    def test_a_missing_dataset_fails_before_training(self, tmp_path: Path) -> None:
        with pytest.raises(AssetNotFoundError, match="build_dataset"):
            AutoencoderTrainer(_config(tmp_path), _DEGRADATION).train()
