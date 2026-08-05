"""Unit tests for sentry_ai.sensors.frame: labels, projection, YOLO export."""

from __future__ import annotations

import numpy as np
import pytest

from sentry_ai.common.exceptions import DomainValidationError
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind
from sentry_ai.interfaces.perception import BoundingBox, Detection
from sentry_ai.sensors.camera import CameraView
from sentry_ai.sensors.frame import (
    YOLO_CLASSES,
    CameraFrame,
    clipped_box,
    yolo_class_id,
)

_VIEW = CameraView(
    camera_id="cam",
    origin=Position(10, 4),
    width_tiles=4,
    height_tiles=2,
    tile_size_px=10,
)


def _pixels(view: CameraView = _VIEW) -> np.ndarray:
    return np.zeros((view.frame_height, view.frame_width, 3), dtype=np.uint8)


def _detection(label: EntityKind, box: tuple[int, int, int, int]) -> Detection:
    return Detection(
        label=label,
        confidence=1.0,
        bbox=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
    )


class TestFrameValidation:
    def test_a_well_formed_frame_is_accepted(self) -> None:
        frame = CameraFrame(view=_VIEW, pixels=_pixels())
        assert (frame.width, frame.height) == (40, 20)
        assert frame.camera_id == "cam"

    def test_a_non_rgb_array_is_rejected(self) -> None:
        with pytest.raises(DomainValidationError, match="height, width, 3"):
            CameraFrame(view=_VIEW, pixels=np.zeros((20, 40), dtype=np.uint8))

    def test_pixels_that_disagree_with_the_camera_are_rejected(self) -> None:
        """Silently accepting a mismatched frame would misplace every label."""
        with pytest.raises(DomainValidationError, match="does not match camera"):
            CameraFrame(view=_VIEW, pixels=np.zeros((21, 40, 3), dtype=np.uint8))


class TestLocatingDetectionsOnTheMap:
    def test_a_detection_maps_back_to_the_tile_it_sits_on(self) -> None:
        frame = CameraFrame(view=_VIEW, pixels=_pixels())
        # A marker inside the tile at world (12, 5) -> frame pixels 20..30, 10..20.
        detection = _detection(EntityKind.VICTIM, (22, 12, 28, 18))
        assert frame.world_position_of(detection) == Position(12, 5)

    def test_a_box_touching_the_far_edge_still_resolves(self) -> None:
        """Box centres can land exactly on the frame bound; clamping avoids a raise."""
        frame = CameraFrame(view=_VIEW, pixels=_pixels())
        detection = _detection(EntityKind.FIRE, (0, 0, 40, 20))
        assert frame.world_position_of(detection) == Position(12, 5)


class TestYoloExport:
    def test_class_ids_follow_the_declared_order(self) -> None:
        assert YOLO_CLASSES == (EntityKind.VICTIM, EntityKind.FIRE, EntityKind.OBSTACLE)
        assert [yolo_class_id(kind) for kind in YOLO_CLASSES] == [0, 1, 2]

    def test_a_non_detector_class_has_no_id(self) -> None:
        with pytest.raises(DomainValidationError, match="not one of the detector classes"):
            yolo_class_id(EntityKind.VEHICLE)

    def test_a_label_row_is_normalized_centre_and_size(self) -> None:
        frame = CameraFrame(
            view=_VIEW,
            pixels=_pixels(),
            annotations=(_detection(EntityKind.VICTIM, (10, 5, 30, 15)),),
        )
        assert frame.to_yolo_lines() == ["0 0.500000 0.500000 0.500000 0.500000"]

    def test_every_normalized_value_stays_within_the_frame(self) -> None:
        frame = CameraFrame(
            view=_VIEW,
            pixels=_pixels(),
            annotations=(
                _detection(EntityKind.VICTIM, (0, 0, 4, 4)),
                _detection(EntityKind.OBSTACLE, (36, 16, 40, 20)),
            ),
        )
        for line in frame.to_yolo_lines():
            values = [float(part) for part in line.split()[1:]]
            assert all(0.0 <= value <= 1.0 for value in values)

    def test_non_detector_classes_are_skipped_not_raised(self) -> None:
        frame = CameraFrame(
            view=_VIEW,
            pixels=_pixels(),
            annotations=(
                _detection(EntityKind.VEHICLE, (0, 0, 4, 4)),
                _detection(EntityKind.FIRE, (10, 5, 30, 15)),
            ),
        )
        assert frame.to_yolo_lines() == ["1 0.500000 0.500000 0.500000 0.500000"]


class TestReplacingPixels:
    def test_swapping_pixels_keeps_the_ground_truth(self) -> None:
        """Smoke changes what the camera sees, never where the victim is."""
        annotations = (_detection(EntityKind.VICTIM, (10, 5, 30, 15)),)
        frame = CameraFrame(view=_VIEW, pixels=_pixels(), annotations=annotations)
        degraded = frame.with_pixels(np.full_like(frame.pixels, 200))

        assert degraded.annotations == annotations
        assert degraded.view is frame.view
        assert not np.array_equal(degraded.pixels, frame.pixels)


class TestClipping:
    def test_a_fully_visible_box_is_unchanged(self) -> None:
        box = clipped_box(4, 4, 12, 12, view=_VIEW)
        assert box == BoundingBox(x_min=4, y_min=4, x_max=12, y_max=12)

    def test_a_box_overhanging_an_edge_is_trimmed(self) -> None:
        box = clipped_box(-6, -6, 12, 12, view=_VIEW)
        assert box == BoundingBox(x_min=0, y_min=0, x_max=12, y_max=12)

    def test_a_box_entirely_off_frame_is_dropped(self) -> None:
        assert clipped_box(50, 50, 60, 60, view=_VIEW) is None

    def test_a_box_reduced_to_nothing_is_dropped(self) -> None:
        assert clipped_box(-10, 0, 0, 10, view=_VIEW) is None
