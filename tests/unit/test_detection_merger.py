"""Unit tests for sentry_ai.perception.merger.DetectionMerger (Phase 3.2).

The whole reason merging is its own step is that it can be tested without a
model, without pixels, and without a map: hand-built detections in,
world-space objects out. Every test here does exactly that.

The case that matters most is the asymmetry. Two cameras seeing one victim
must yield one victim; two victims on neighbouring tiles must stay two
people. Getting that backwards either invents a casualty or erases one.
"""

from __future__ import annotations

import pytest

from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind
from sentry_ai.interfaces.perception import BoundingBox, Detection
from sentry_ai.perception.merger import (
    CameraObservation,
    DetectionMerger,
    WorldDetection,
)
from sentry_ai.sensors.camera import CameraView

TILE = 16


def _view(camera_id: str, origin_x: int = 0, origin_y: int = 0) -> CameraView:
    return CameraView(
        camera_id=camera_id,
        origin=Position(origin_x, origin_y),
        width_tiles=16,
        height_tiles=10,
        tile_size_px=TILE,
    )


def _marker(view: CameraView, tile: Position, kind: EntityKind, conf: float = 0.9) -> Detection:
    """A small detection centred in ``tile``, as the rasterizer would paint it."""
    x_min, y_min, _, _ = view.tile_rect(tile)
    return Detection(
        label=kind,
        confidence=conf,
        bbox=BoundingBox(x_min + 4, y_min + 4, x_min + 12, y_min + 12),
    )


def _span(view: CameraView, first: Position, last: Position, conf: float = 0.9) -> Detection:
    """A fire-sized detection spanning ``first``..``last`` inclusive."""
    x_min, y_min, _, _ = view.tile_rect(first)
    _, _, x_max, y_max = view.tile_rect(last)
    return Detection(
        label=EntityKind.FIRE,
        confidence=conf,
        bbox=BoundingBox(x_min, y_min, x_max, y_max),
    )


@pytest.fixture
def merger() -> DetectionMerger:
    return DetectionMerger()


class TestProjection:
    def test_a_detection_becomes_a_world_tile(self, merger: DetectionMerger) -> None:
        view = _view("cctv_nw")
        seen = (_marker(view, Position(3, 2), EntityKind.VICTIM),)
        found = merger.merge([CameraObservation(view, seen)])
        assert len(found) == 1
        assert found[0].tiles == {Position(3, 2)}
        assert found[0].label is EntityKind.VICTIM

    def test_a_cameras_origin_offsets_the_result(self, merger: DetectionMerger) -> None:
        """The east cameras start at x=14; a box at their left edge is not x=0."""
        view = _view("cctv_ne", origin_x=14)
        seen = (_marker(view, Position(14, 0), EntityKind.VICTIM),)
        found = merger.merge([CameraObservation(view, seen)])
        assert found[0].tiles == {Position(14, 0)}

    def test_the_reporting_camera_is_recorded(self, merger: DetectionMerger) -> None:
        view = _view("cctv_sw", origin_y=10)
        seen = (_marker(view, Position(1, 11), EntityKind.VICTIM),)
        found = merger.merge([CameraObservation(view, seen)])
        assert found[0].camera_ids == {"cctv_sw"}
        assert not found[0].corroborated

    def test_no_observations_yields_nothing(self, merger: DetectionMerger) -> None:
        assert merger.merge([]) == []

    def test_a_camera_reporting_nothing_yields_nothing(self, merger: DetectionMerger) -> None:
        assert merger.merge([CameraObservation(_view("cctv_nw"))]) == []


class TestOverlappingCameras:
    """The two camera columns overlap by two tiles, so this is the real case."""

    def test_one_victim_seen_twice_is_one_victim(self, merger: DetectionMerger) -> None:
        west, east = _view("cctv_nw"), _view("cctv_ne", origin_x=14)
        shared = Position(15, 3)
        found = merger.merge(
            [
                CameraObservation(west, (_marker(west, shared, EntityKind.VICTIM, 0.8),)),
                CameraObservation(east, (_marker(east, shared, EntityKind.VICTIM, 0.95),)),
            ]
        )
        assert len(found) == 1
        assert found[0].tiles == {shared}

    def test_both_cameras_are_credited(self, merger: DetectionMerger) -> None:
        west, east = _view("cctv_nw"), _view("cctv_ne", origin_x=14)
        shared = Position(15, 3)
        found = merger.merge(
            [
                CameraObservation(west, (_marker(west, shared, EntityKind.VICTIM),)),
                CameraObservation(east, (_marker(east, shared, EntityKind.VICTIM),)),
            ]
        )
        assert found[0].camera_ids == {"cctv_nw", "cctv_ne"}
        assert found[0].corroborated

    def test_the_most_confident_view_wins(self, merger: DetectionMerger) -> None:
        """A camera with a clear line of sight is not dragged down by one in smoke."""
        west, east = _view("cctv_nw"), _view("cctv_ne", origin_x=14)
        shared = Position(15, 3)
        found = merger.merge(
            [
                CameraObservation(west, (_marker(west, shared, EntityKind.VICTIM, 0.31),)),
                CameraObservation(east, (_marker(east, shared, EntityKind.VICTIM, 0.97),)),
            ]
        )
        assert found[0].confidence == pytest.approx(0.97)

    def test_merging_does_not_depend_on_camera_order(self, merger: DetectionMerger) -> None:
        west, east = _view("cctv_nw"), _view("cctv_ne", origin_x=14)
        shared = Position(15, 3)
        west_obs = CameraObservation(west, (_marker(west, shared, EntityKind.VICTIM, 0.4),))
        east_obs = CameraObservation(east, (_marker(east, shared, EntityKind.VICTIM, 0.9),))
        assert merger.merge([west_obs, east_obs]) == merger.merge([east_obs, west_obs])


class TestVictimsAreNeverOverMerged:
    """Erasing a person is the worst error this system can make."""

    def test_two_victims_on_adjacent_tiles_stay_two(self, merger: DetectionMerger) -> None:
        view = _view("cctv_nw")
        found = merger.merge(
            [
                CameraObservation(
                    view,
                    (
                        _marker(view, Position(4, 4), EntityKind.VICTIM),
                        _marker(view, Position(5, 4), EntityKind.VICTIM),
                    ),
                )
            ]
        )
        assert len(found) == 2
        assert {tile for found_one in found for tile in found_one.tiles} == {
            Position(4, 4),
            Position(5, 4),
        }

    def test_two_victims_seen_by_two_cameras_stay_two(self, merger: DetectionMerger) -> None:
        west, east = _view("cctv_nw"), _view("cctv_ne", origin_x=14)
        left, right = Position(14, 5), Position(15, 5)
        found = merger.merge(
            [
                CameraObservation(
                    west,
                    (
                        _marker(west, left, EntityKind.VICTIM),
                        _marker(west, right, EntityKind.VICTIM),
                    ),
                ),
                CameraObservation(
                    east,
                    (
                        _marker(east, left, EntityKind.VICTIM),
                        _marker(east, right, EntityKind.VICTIM),
                    ),
                ),
            ]
        )
        assert len(found) == 2
        assert all(found_one.corroborated for found_one in found)

    def test_adjacent_debris_stays_separate(self, merger: DetectionMerger) -> None:
        view = _view("cctv_nw")
        found = merger.merge(
            [
                CameraObservation(
                    view,
                    (
                        _marker(view, Position(2, 2), EntityKind.OBSTACLE),
                        _marker(view, Position(3, 2), EntityKind.OBSTACLE),
                    ),
                )
            ]
        )
        assert len(found) == 2


class TestFireSpansCameraSeams:
    """A fire is clipped differently by each camera, so its halves only touch."""

    def test_two_touching_halves_are_one_fire(self, merger: DetectionMerger) -> None:
        west, east = _view("cctv_nw"), _view("cctv_ne", origin_x=14)
        found = merger.merge(
            [
                CameraObservation(west, (_span(west, Position(12, 4), Position(15, 6)),)),
                CameraObservation(east, (_span(east, Position(16, 4), Position(18, 6)),)),
            ]
        )
        assert len(found) == 1
        assert found[0].label is EntityKind.FIRE
        assert Position(12, 4) in found[0].tiles
        assert Position(18, 6) in found[0].tiles

    def test_the_merged_footprint_is_the_union(self, merger: DetectionMerger) -> None:
        """3.3 marks every burning tile, so none may be dropped."""
        west, east = _view("cctv_nw"), _view("cctv_ne", origin_x=14)
        found = merger.merge(
            [
                CameraObservation(west, (_span(west, Position(12, 4), Position(15, 4)),)),
                CameraObservation(east, (_span(east, Position(16, 4), Position(18, 4)),)),
            ]
        )
        assert found[0].tiles == {Position(x, 4) for x in range(12, 19)}

    def test_two_distant_fires_stay_two(self, merger: DetectionMerger) -> None:
        view = _view("cctv_nw")
        found = merger.merge(
            [
                CameraObservation(
                    view,
                    (
                        _span(view, Position(1, 1), Position(2, 2)),
                        _span(view, Position(8, 7), Position(9, 8)),
                    ),
                )
            ]
        )
        assert len(found) == 2

    def test_a_fire_spanning_three_footprints_merges_transitively(
        self, merger: DetectionMerger
    ) -> None:
        """A meets B and B meets C, so all three are one fire."""
        view = _view("cctv_nw")
        found = merger.merge(
            [
                CameraObservation(
                    view,
                    (
                        _span(view, Position(1, 1), Position(2, 1)),
                        _span(view, Position(5, 1), Position(6, 1)),
                        _span(view, Position(3, 1), Position(4, 1)),
                    ),
                )
            ]
        )
        assert len(found) == 1
        assert found[0].tiles == {Position(x, 1) for x in range(1, 7)}


class TestDifferentClassesNeverMerge:
    def test_a_victim_on_a_debris_tile_stays_two_objects(self, merger: DetectionMerger) -> None:
        """Nothing here undoes the rasterizer's one-annotation-per-tile rule;
        the merger simply never conflates classes."""
        view = _view("cctv_nw")
        tile = Position(6, 6)
        found = merger.merge(
            [
                CameraObservation(
                    view,
                    (
                        _marker(view, tile, EntityKind.VICTIM),
                        _marker(view, tile, EntityKind.OBSTACLE),
                    ),
                )
            ]
        )
        assert len(found) == 2
        assert {found_one.label for found_one in found} == {
            EntityKind.VICTIM,
            EntityKind.OBSTACLE,
        }

    def test_fire_does_not_absorb_an_adjacent_victim(self, merger: DetectionMerger) -> None:
        view = _view("cctv_nw")
        found = merger.merge(
            [
                CameraObservation(
                    view,
                    (
                        _span(view, Position(1, 1), Position(2, 1)),
                        _marker(view, Position(3, 1), EntityKind.VICTIM),
                    ),
                )
            ]
        )
        assert len(found) == 2


class TestOutputIsDeterministic:
    def test_results_are_sorted_by_label_then_position(self, merger: DetectionMerger) -> None:
        view = _view("cctv_nw")
        found = merger.merge(
            [
                CameraObservation(
                    view,
                    (
                        _marker(view, Position(5, 5), EntityKind.OBSTACLE),
                        _marker(view, Position(1, 1), EntityKind.VICTIM),
                        _marker(view, Position(2, 0), EntityKind.VICTIM),
                    ),
                )
            ]
        )
        assert [(f.label, f.position) for f in found] == [
            (EntityKind.OBSTACLE, Position(5, 5)),
            (EntityKind.VICTIM, Position(2, 0)),
            (EntityKind.VICTIM, Position(1, 1)),
        ]


class TestWorldDetection:
    def test_a_single_tile_detection_reports_that_tile(self) -> None:
        found = WorldDetection(EntityKind.VICTIM, frozenset({Position(4, 7)}), 0.9)
        assert found.position == Position(4, 7)

    def test_a_footprint_reports_its_centre(self) -> None:
        tiles = frozenset({Position(x, 5) for x in range(4, 9)})
        assert WorldDetection(EntityKind.FIRE, tiles, 0.8).position == Position(6, 5)

    def test_an_empty_footprint_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one tile"):
            WorldDetection(EntityKind.VICTIM, frozenset(), 0.5)

    def test_an_out_of_range_confidence_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            WorldDetection(EntityKind.VICTIM, frozenset({Position(0, 0)}), 1.4)


class TestProjectionIsLabelAware:
    """Fire projects by tile centre; victims never do.

    Both rules exist because a detector's box is a pixel or two off, and the
    two classes pay opposite prices for that error. An over-claimed tile
    around a fire is an impassable cell in the middle of an open street. A
    dropped victim is a person nobody is sent to.
    """

    def test_a_slightly_wide_fire_box_does_not_claim_the_next_tile(self) -> None:
        view = _view("cctv_nw")
        x_min, y_min, _, _ = view.tile_rect(Position(3, 3))
        # One tile's worth of fire, with the box 5px proud on every side.
        sloppy = Detection(
            label=EntityKind.FIRE,
            confidence=0.9,
            bbox=BoundingBox(x_min - 5, y_min - 5, x_min + TILE + 5, y_min + TILE + 5),
        )
        merged = DetectionMerger().merge([CameraObservation(view, (sloppy,))])
        assert len(merged) == 1
        assert merged[0].tiles == {Position(3, 3)}

    def test_the_same_slop_on_a_victim_box_is_tolerated(self) -> None:
        """Victims keep the generous rule — losing one is unrecoverable."""
        view = _view("cctv_nw")
        x_min, y_min, _, _ = view.tile_rect(Position(3, 3))
        tiny = Detection(
            label=EntityKind.VICTIM,
            confidence=0.9,
            bbox=BoundingBox(x_min + 11, y_min + 11, x_min + 15, y_min + 15),
        )
        merged = DetectionMerger().merge([CameraObservation(view, (tiny,))])
        assert merged[0].tiles == {Position(3, 3)}

    def test_a_fire_box_containing_no_tile_centre_still_reports_something(self) -> None:
        """A hazard must never vanish because it was reported half a tile off."""
        view = _view("cctv_nw")
        x_min, y_min, _, _ = view.tile_rect(Position(3, 3))
        offset = Detection(
            label=EntityKind.FIRE,
            confidence=0.9,
            bbox=BoundingBox(x_min + 10, y_min + 10, x_min + 14, y_min + 14),
        )
        merged = DetectionMerger().merge([CameraObservation(view, (offset,))])
        assert len(merged) == 1
        assert merged[0].tiles

    def test_a_multi_tile_fire_still_spans_its_tiles(self) -> None:
        view = _view("cctv_nw")
        merged = DetectionMerger().merge(
            [CameraObservation(view, (_span(view, Position(2, 2), Position(5, 4)),))]
        )
        assert len(merged[0].tiles) == 12
