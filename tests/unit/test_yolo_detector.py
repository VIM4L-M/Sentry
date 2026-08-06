"""Unit tests for sentry_ai.perception.yolo_detector.

The adapter's job is translation — Ultralytics' floats and class indices in,
this project's :class:`Detection` out — and translation is exactly what can
be tested without a trained model or a GPU. The conversion is driven with a
stand-in result object rather than a real inference run, so these stay fast
and do not depend on weights existing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.domain.enums import EntityKind
from sentry_ai.perception.yolo_detector import YoloDetector, _clamped_box
from sentry_ai.sensors.frame import YOLO_CLASSES


@dataclass
class _FakeTensor:
    """Stands in for a Torch tensor, which only ``.tolist()`` is used on."""

    values: list

    def tolist(self) -> list:
        return self.values


@dataclass
class _FakeBoxes:
    xyxy: _FakeTensor
    conf: _FakeTensor
    cls: _FakeTensor


@dataclass
class _FakeResult:
    boxes: _FakeBoxes | None


def _result(rows: list[tuple[list[float], float, int]]) -> _FakeResult:
    return _FakeResult(
        boxes=_FakeBoxes(
            xyxy=_FakeTensor([row[0] for row in rows]),
            conf=_FakeTensor([row[1] for row in rows]),
            cls=_FakeTensor([float(row[2]) for row in rows]),
        )
    )


class TestLoading:
    def test_missing_weights_fail_immediately(self, tmp_path: Path) -> None:
        """Better here than deep inside a mission."""
        with pytest.raises(AssetNotFoundError, match="weights not found"):
            YoloDetector(weights_path=tmp_path / "nope.pt")


class TestTranslatingResults:
    def test_each_box_becomes_a_detection(self) -> None:
        result = _result([([10.0, 20.0, 30.0, 40.0], 0.9, 0)])
        detections = YoloDetector._to_detections(result, width=256, height=160)

        assert len(detections) == 1
        assert detections[0].label is EntityKind.VICTIM
        assert detections[0].confidence == pytest.approx(0.9)
        assert detections[0].bbox.x_min == 10

    def test_class_indices_map_to_the_declared_order(self) -> None:
        rows = [([0.0, 0.0, 8.0, 8.0], 0.5, index) for index in range(len(YOLO_CLASSES))]
        detections = YoloDetector._to_detections(_result(rows), width=64, height=64)
        assert [d.label for d in detections] == list(YOLO_CLASSES)

    def test_an_unknown_class_id_is_skipped_not_crashed(self) -> None:
        """A retrained model with more classes must not take a mission down."""
        rows = [([0.0, 0.0, 8.0, 8.0], 0.5, 0), ([0.0, 0.0, 8.0, 8.0], 0.5, 99)]
        detections = YoloDetector._to_detections(_result(rows), width=64, height=64)
        assert len(detections) == 1

    def test_a_result_without_boxes_yields_nothing(self) -> None:
        assert YoloDetector._to_detections(_FakeResult(boxes=None), 64, 64) == []

    def test_no_boxes_yields_no_detections(self) -> None:
        assert YoloDetector._to_detections(_result([]), 64, 64) == []

    def test_confidence_is_clamped_into_range(self) -> None:
        """Detection rejects anything outside 0-1; float noise must not trip it."""
        rows = [([0.0, 0.0, 8.0, 8.0], 1.0000001, 0)]
        detections = YoloDetector._to_detections(_result(rows), 64, 64)
        assert 0.0 <= detections[0].confidence <= 1.0

    def test_a_zero_area_box_is_dropped(self) -> None:
        rows = [([10.0, 10.0, 10.0, 10.0], 0.9, 0)]
        assert YoloDetector._to_detections(_result(rows), 64, 64) == []

    def test_a_box_entirely_off_frame_is_dropped(self) -> None:
        rows = [([500.0, 500.0, 600.0, 600.0], 0.9, 0)]
        assert YoloDetector._to_detections(_result(rows), 256, 160) == []

    def test_a_tiny_but_real_box_survives(self) -> None:
        """Victims are the smallest class; discarding marginal ones is costly."""
        rows = [([10.0, 10.0, 10.4, 10.4], 0.9, 0)]
        detections = YoloDetector._to_detections(_result(rows), 256, 160)
        assert len(detections) == 1
        assert detections[0].bbox.x_max > detections[0].bbox.x_min


class TestClampingBoxes:
    def test_the_integer_box_covers_the_float_one(self) -> None:
        """Rounding must never shave a pixel off a detection."""
        box = _clamped_box([10.4, 20.6, 30.4, 40.6], width=256, height=160)
        assert box is not None
        assert (box.x_min, box.y_min, box.x_max, box.y_max) == (10, 20, 31, 41)

    def test_a_box_overhanging_an_edge_is_trimmed(self) -> None:
        box = _clamped_box([-5.0, -5.0, 300.0, 200.0], width=256, height=160)
        assert box is not None
        assert (box.x_min, box.y_min, box.x_max, box.y_max) == (0, 0, 256, 160)

    def test_a_box_never_exceeds_the_frame(self) -> None:
        """A label running past the image is rejected by most toolchains."""
        box = _clamped_box([250.0, 155.0, 999.0, 999.0], width=256, height=160)
        assert box is not None
        assert box.x_max <= 256
        assert box.y_max <= 160

    def test_a_box_entirely_off_frame_is_dropped(self) -> None:
        assert _clamped_box([500.0, 500.0, 600.0, 600.0], width=256, height=160) is None

    @pytest.mark.parametrize(
        "xyxy",
        [
            [0.0, 0.0, 0.4, 0.4],
            [255.6, 159.6, 255.7, 159.7],
        ],
    )
    def test_sub_pixel_boxes_never_produce_an_invalid_bounding_box(
        self, xyxy: list[float]
    ) -> None:
        box = _clamped_box(xyxy, width=256, height=160)
        if box is not None:
            assert box.x_max > box.x_min
            assert box.y_max > box.y_min
