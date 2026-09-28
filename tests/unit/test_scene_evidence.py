"""Unit tests for egocentric scene evidence (Phase 7).

Covers the value type in sentry_ai.interfaces.decision and the geometry in
sentry_ai.perception.scene_evidence: a detection must land in the region
named relative to the vehicle's heading, not the compass.
"""

from __future__ import annotations

import pytest

from sentry_ai.common.exceptions import DomainValidationError
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind, Heading
from sentry_ai.interfaces.decision import EVIDENCE_KINDS, EVIDENCE_REGIONS, Region, SceneEvidence
from sentry_ai.interfaces.perception import BoundingBox, Detection
from sentry_ai.perception.scene_evidence import egocentric_evidence
from sentry_ai.sensors.camera import CameraView

TILE = 16


def _view() -> CameraView:
    """A 9x9 onboard-style window whose top-left tile is (0, 0)."""
    return CameraView("onboard", Position(0, 0), 9, 9, TILE)


def _on_tile(kind: EntityKind, x: int, y: int, confidence: float = 0.9) -> Detection:
    """A detection box sitting inside world tile (x, y)."""
    left, top = x * TILE + 4, y * TILE + 4
    return Detection(kind, confidence, BoundingBox(left, top, left + 8, top + 8))


class TestSceneEvidence:
    def test_empty_reads_zero_everywhere(self) -> None:
        evidence = SceneEvidence.empty()
        assert evidence.as_list() == [0.0] * (len(EVIDENCE_KINDS) * len(EVIDENCE_REGIONS))

    def test_at_returns_stored_confidence(self) -> None:
        evidence = SceneEvidence({(EntityKind.FIRE, Region.AHEAD): 0.7})
        assert evidence.at(EntityKind.FIRE, Region.AHEAD) == 0.7
        assert evidence.at(EntityKind.FIRE, Region.LEFT) == 0.0

    def test_rejects_confidence_outside_unit_range(self) -> None:
        with pytest.raises(DomainValidationError):
            SceneEvidence({(EntityKind.FIRE, Region.AHEAD): 1.5})

    def test_rejects_kinds_without_a_slot(self) -> None:
        with pytest.raises(DomainValidationError):
            SceneEvidence({(EntityKind.VEHICLE, Region.AHEAD): 0.5})


class TestEgocentricEvidence:
    def test_debris_in_front_of_a_north_facing_vehicle_is_ahead(self) -> None:
        found = egocentric_evidence(
            [_on_tile(EntityKind.OBSTACLE, 4, 3)], _view(), Position(4, 4), Heading.NORTH
        )
        assert found.at(EntityKind.OBSTACLE, Region.AHEAD) == pytest.approx(0.9)
        assert found.at(EntityKind.OBSTACLE, Region.NEARBY) == pytest.approx(0.9)
        assert found.at(EntityKind.OBSTACLE, Region.LEFT) == 0.0

    def test_the_same_tile_is_left_when_facing_east(self) -> None:
        found = egocentric_evidence(
            [_on_tile(EntityKind.FIRE, 4, 3)], _view(), Position(4, 4), Heading.EAST
        )
        assert found.at(EntityKind.FIRE, Region.LEFT) == pytest.approx(0.9)
        assert found.at(EntityKind.FIRE, Region.AHEAD) == 0.0

    def test_right_and_behind_follow_the_heading(self) -> None:
        detections = [_on_tile(EntityKind.VICTIM, 3, 4), _on_tile(EntityKind.FIRE, 4, 3)]
        found = egocentric_evidence(detections, _view(), Position(4, 4), Heading.SOUTH)
        assert found.at(EntityKind.VICTIM, Region.RIGHT) == pytest.approx(0.9)
        assert found.at(EntityKind.FIRE, Region.BEHIND) == pytest.approx(0.9)

    def test_two_tiles_out_is_ahead_far(self) -> None:
        found = egocentric_evidence(
            [_on_tile(EntityKind.OBSTACLE, 6, 4)], _view(), Position(4, 4), Heading.EAST
        )
        assert found.at(EntityKind.OBSTACLE, Region.AHEAD_FAR) == pytest.approx(0.9)
        assert found.at(EntityKind.OBSTACLE, Region.AHEAD) == 0.0

    def test_the_highest_confidence_wins_a_slot(self) -> None:
        detections = [
            _on_tile(EntityKind.FIRE, 4, 3, 0.4),
            _on_tile(EntityKind.FIRE, 4, 3, 0.8),
        ]
        found = egocentric_evidence(detections, _view(), Position(4, 4), Heading.NORTH)
        assert found.at(EntityKind.FIRE, Region.AHEAD) == pytest.approx(0.8)

    def test_classes_without_a_slot_are_ignored(self) -> None:
        found = egocentric_evidence(
            [_on_tile(EntityKind.SMOKE, 4, 3)], _view(), Position(4, 4), Heading.NORTH
        )
        assert found == SceneEvidence.empty()

    def test_a_vehicle_on_the_map_edge_does_not_fail(self) -> None:
        found = egocentric_evidence(
            [_on_tile(EntityKind.FIRE, 0, 1)], _view(), Position(0, 0), Heading.NORTH
        )
        assert found.at(EntityKind.FIRE, Region.BEHIND) == pytest.approx(0.9)


class TestFireFootprint:
    def test_a_fire_box_spilling_half_a_tile_is_not_ahead(self) -> None:
        # Fire at (5, 3), diagonally ahead-right of a north-facing vehicle at (4, 4).
        # Its box spills into (4, 3), the tile ahead, without covering its centre.
        box = BoundingBox(5 * TILE - 6, 3 * TILE + 2, 6 * TILE, 4 * TILE - 2)
        fire = Detection(EntityKind.FIRE, 0.9, box)
        found = egocentric_evidence([fire], _view(), Position(4, 4), Heading.NORTH)
        assert found.at(EntityKind.FIRE, Region.AHEAD) == 0.0

    def test_a_small_fire_box_without_a_tile_centre_still_counts(self) -> None:
        box = BoundingBox(4 * TILE + 1, 3 * TILE + 1, 4 * TILE + 6, 3 * TILE + 6)
        fire = Detection(EntityKind.FIRE, 0.9, box)
        found = egocentric_evidence([fire], _view(), Position(4, 4), Heading.NORTH)
        assert found.at(EntityKind.FIRE, Region.AHEAD) == pytest.approx(0.9)
