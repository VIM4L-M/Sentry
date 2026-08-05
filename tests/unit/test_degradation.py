"""Unit tests for sentry_ai.sensors.degradation.FrameDegrader.

The degrader's contract is narrow but load-bearing: it must visibly corrupt
the pixels, leave the labels completely alone, and stay reproducible when
handed a seeded generator. Everything Unit IV trains on depends on those
three properties holding together.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentry_ai.config.schema import DegradationConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind
from sentry_ai.interfaces.perception import BoundingBox, Detection
from sentry_ai.sensors.camera import CameraView
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.frame import CameraFrame

_VIEW = CameraView(
    camera_id="cam",
    origin=Position(0, 0),
    width_tiles=8,
    height_tiles=8,
    tile_size_px=8,
)

_ANNOTATION = Detection(
    label=EntityKind.VICTIM,
    confidence=1.0,
    bbox=BoundingBox(x_min=10, y_min=10, x_max=20, y_max=20),
)


def _frame(fill: int = 120) -> CameraFrame:
    pixels = np.full((_VIEW.frame_height, _VIEW.frame_width, 3), fill, dtype=np.uint8)
    return CameraFrame(view=_VIEW, pixels=pixels, annotations=(_ANNOTATION,))


def _gradient_frame() -> CameraFrame:
    """A frame with hard vertical edges, so blur has something to soften."""
    pixels = np.zeros((_VIEW.frame_height, _VIEW.frame_width, 3), dtype=np.uint8)
    pixels[:, ::2] = 255
    return CameraFrame(view=_VIEW, pixels=pixels)


def _degrader(rng_seed: int = 0, **overrides: object) -> FrameDegrader:
    return FrameDegrader(
        DegradationConfig(**overrides),  # type: ignore[arg-type]
        np.random.default_rng(rng_seed),
    )


class TestOutputContract:
    def test_the_result_is_still_a_valid_frame(self) -> None:
        degraded = _degrader().degrade(_frame())
        assert degraded.pixels.shape == _frame().pixels.shape
        assert degraded.pixels.dtype == np.uint8

    def test_pixels_stay_inside_the_8_bit_range(self) -> None:
        degraded = _degrader(noise_std=80.0, smoke_density=1.0).degrade(_frame(fill=250))
        assert degraded.pixels.min() >= 0
        assert degraded.pixels.max() <= 255

    def test_the_ground_truth_survives_untouched(self) -> None:
        """Smoke hides the victim; it does not move them."""
        degraded = _degrader(smoke_density=1.0, noise_std=40.0).degrade(_frame())
        assert degraded.annotations == (_ANNOTATION,)
        assert degraded.view is _VIEW

    def test_the_clean_frame_is_not_mutated(self) -> None:
        frame = _frame()
        original = frame.pixels.copy()
        _degrader(noise_std=50.0).degrade(frame)
        assert np.array_equal(frame.pixels, original)


class TestCorruptionActuallyHappens:
    def test_a_degraded_frame_differs_from_the_clean_one(self) -> None:
        frame = _frame()
        assert not np.array_equal(_degrader().degrade(frame).pixels, frame.pixels)

    def test_smoke_pulls_the_image_toward_its_grey(self) -> None:
        frame = _frame(fill=20)
        degraded = _degrader(
            smoke_density=1.0, smoke_grey=200, blur_radius=0, noise_std=0.0
        ).degrade(frame)
        assert degraded.pixels.mean() > frame.pixels.mean() + 50

    def test_blur_softens_hard_edges(self) -> None:
        frame = _gradient_frame()
        sharp_spread = float(frame.pixels.std())
        degraded = _degrader(
            blur_radius=2, smoke_density=0.0, noise_std=0.0
        ).degrade(frame)
        assert float(degraded.pixels.std()) < sharp_spread

    def test_noise_adds_variation_to_a_flat_image(self) -> None:
        degraded = _degrader(
            noise_std=20.0, smoke_density=0.0, blur_radius=0
        ).degrade(_frame())
        assert float(degraded.pixels.std()) > 5.0


class TestDisablingEachStage:
    def test_everything_off_is_a_passthrough(self) -> None:
        frame = _frame()
        degraded = _degrader(
            smoke_density=0.0, blur_radius=0, noise_std=0.0
        ).degrade(frame)
        assert np.array_equal(degraded.pixels, frame.pixels)

    def test_zero_smoke_leaves_brightness_alone(self) -> None:
        frame = _frame(fill=30)
        degraded = _degrader(
            smoke_density=0.0, blur_radius=0, noise_std=0.0
        ).degrade(frame)
        assert degraded.pixels.mean() == pytest.approx(30.0)


class TestReproducibility:
    def test_the_same_seed_produces_the_same_corruption(self) -> None:
        first = _degrader(rng_seed=42).degrade(_frame())
        second = _degrader(rng_seed=42).degrade(_frame())
        assert np.array_equal(first.pixels, second.pixels)

    def test_a_different_seed_produces_different_corruption(self) -> None:
        first = _degrader(rng_seed=1).degrade(_frame())
        second = _degrader(rng_seed=2).degrade(_frame())
        assert not np.array_equal(first.pixels, second.pixels)

    def test_successive_captures_differ_so_a_dataset_is_not_one_sample(self) -> None:
        """A degrader reused across a run must keep sampling new smoke."""
        degrader = _degrader(rng_seed=7)
        first = degrader.degrade(_frame())
        second = degrader.degrade(_frame())
        assert not np.array_equal(first.pixels, second.pixels)


class TestTrainingPairs:
    def test_a_pair_is_corrupted_then_clean(self) -> None:
        frame = _frame()
        corrupted, clean = _degrader().degrade_pair(frame)
        assert clean is frame
        assert not np.array_equal(corrupted.pixels, clean.pixels)

    def test_both_halves_of_a_pair_are_pixel_aligned(self) -> None:
        """Perfect alignment is the whole reason for corrupting synthetically."""
        corrupted, clean = _degrader().degrade_pair(_frame())
        assert corrupted.pixels.shape == clean.pixels.shape
        assert corrupted.annotations == clean.annotations
