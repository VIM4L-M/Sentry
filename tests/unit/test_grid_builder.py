"""Unit tests for sentry_ai.perception.grid_builder (Phase 3.3).

Like the merger before it, the builder is testable with nothing in the
loop: hand-built world detections in, an occupancy grid out. No model, no
pixels, no cameras.

The property under the most pressure is precedence. The builder and
``OccupancyGrid.from_city_map`` must resolve a contested tile identically,
because one is scored against the other — if they disagreed about a victim
lying in burning rubble, the metric would be measuring the disagreement
rather than the detector.
"""

from __future__ import annotations

from typing import Any

import pytest

from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.perception.grid_builder import DETECTION_TO_OCCUPANCY, OccupancyGridBuilder
from sentry_ai.perception.merger import WorldDetection

# 5x4 city:
#   row0: .....
#   row1: .##..
#   row2: .....
#   row3: ..r..     r = rubble (static debris, part of the survey)
_MAP_DATA: dict[str, Any] = {
    "width": 5,
    "height": 4,
    "grid": [".....", ".##..", ".....", "..r.."],
    "safe_zone": {"position": [0, 0], "radius": 1, "capacity": 2},
    "vehicle_start": [0, 2],
    "victims": [{"id": "v1", "position": [4, 2]}],
    "fires": [{"id": "f1", "position": [3, 0], "intensity": 0.8, "radius": 0}],
}


@pytest.fixture
def city_map() -> CityMap:
    return CityMap.from_config(_MAP_DATA)


@pytest.fixture
def builder(city_map: CityMap) -> OccupancyGridBuilder:
    return OccupancyGridBuilder.from_city_map(city_map)


def _found(
    kind: EntityKind, *tiles: tuple[int, int], confidence: float = 0.9
) -> WorldDetection:
    return WorldDetection(
        label=kind,
        tiles=frozenset(Position(x, y) for x, y in tiles),
        confidence=confidence,
        camera_ids=frozenset({"cctv_nw"}),
    )


class TestTheSurveyedTerrain:
    def test_the_static_layout_survives_into_the_belief(
        self, builder: OccupancyGridBuilder
    ) -> None:
        grid = builder.build([])
        assert grid.code_at(Position(1, 1)) is OccupancyCode.BUILDING
        assert grid.code_at(Position(2, 3)) is OccupancyCode.DEBRIS
        assert grid.code_at(Position(0, 0)) is OccupancyCode.HOSPITAL

    def test_no_detections_means_no_fire_and_no_victims(
        self, builder: OccupancyGridBuilder
    ) -> None:
        """Terrain is surveyed; everything dynamic has to be seen."""
        grid = builder.build([])
        assert grid.positions_with(OccupancyCode.FIRE) == []
        assert grid.positions_with(OccupancyCode.VICTIM) == []

    def test_the_terrain_base_is_never_mutated(self, builder: OccupancyGridBuilder) -> None:
        """One builder serves a whole mission — a build must not poison the next."""
        before = builder.terrain.as_array()
        builder.build([_found(EntityKind.FIRE, (2, 2))], vehicle_position=Position(0, 2))
        assert (builder.terrain.as_array() == before).all()

    def test_two_builds_from_the_same_builder_are_independent(
        self, builder: OccupancyGridBuilder
    ) -> None:
        first = builder.build([_found(EntityKind.FIRE, (2, 2))])
        second = builder.build([])
        assert first.code_at(Position(2, 2)) is OccupancyCode.FIRE
        assert second.code_at(Position(2, 2)) is OccupancyCode.ROAD

    def test_dimensions_come_from_the_survey(
        self, builder: OccupancyGridBuilder, city_map: CityMap
    ) -> None:
        assert (builder.width, builder.height) == (city_map.width, city_map.height)


class TestStampingDetections:
    @pytest.mark.parametrize(
        ("kind", "code"),
        [
            (EntityKind.VICTIM, OccupancyCode.VICTIM),
            (EntityKind.FIRE, OccupancyCode.FIRE),
            (EntityKind.OBSTACLE, OccupancyCode.DEBRIS),
        ],
    )
    def test_each_detector_class_writes_its_code(
        self, builder: OccupancyGridBuilder, kind: EntityKind, code: OccupancyCode
    ) -> None:
        grid = builder.build([_found(kind, (3, 2))])
        assert grid.code_at(Position(3, 2)) is code

    def test_the_mapping_covers_every_detector_class(self) -> None:
        """A new detector class must not silently vanish from the grid."""
        from sentry_ai.sensors.frame import YOLO_CLASSES

        assert set(DETECTION_TO_OCCUPANCY) == set(YOLO_CLASSES)

    def test_a_multi_tile_fire_marks_every_tile(self, builder: OccupancyGridBuilder) -> None:
        grid = builder.build([_found(EntityKind.FIRE, (2, 0), (3, 0), (4, 0))])
        burning = set(grid.positions_with(OccupancyCode.FIRE))
        assert burning == {Position(2, 0), Position(3, 0), Position(4, 0)}

    def test_tiles_off_the_map_are_dropped_not_raised(
        self, builder: OccupancyGridBuilder
    ) -> None:
        """One stray box from a fallible detector must not end a mission."""
        grid = builder.build([_found(EntityKind.FIRE, (2, 2), (99, 99))])
        assert grid.positions_with(OccupancyCode.FIRE) == [Position(2, 2)]

    def test_detections_can_block_a_road_the_survey_called_clear(
        self, builder: OccupancyGridBuilder
    ) -> None:
        """A mid-mission collapse is exactly what the detector is for."""
        assert builder.build([]).is_traversable(Position(3, 2))
        grid = builder.build([_found(EntityKind.OBSTACLE, (3, 2))])
        assert not grid.is_traversable(Position(3, 2))


class TestPrecedence:
    def test_a_victim_outranks_the_fire_on_their_tile(
        self, builder: OccupancyGridBuilder
    ) -> None:
        """Otherwise the planner routes around the person it came to collect."""
        grid = builder.build(
            [_found(EntityKind.FIRE, (3, 2)), _found(EntityKind.VICTIM, (3, 2))]
        )
        assert grid.code_at(Position(3, 2)) is OccupancyCode.VICTIM

    def test_a_victim_outranks_debris_on_their_tile(
        self, builder: OccupancyGridBuilder
    ) -> None:
        grid = builder.build(
            [_found(EntityKind.OBSTACLE, (3, 2)), _found(EntityKind.VICTIM, (3, 2))]
        )
        assert grid.code_at(Position(3, 2)) is OccupancyCode.VICTIM

    def test_fire_outranks_debris(self, builder: OccupancyGridBuilder) -> None:
        grid = builder.build(
            [_found(EntityKind.VICTIM, (4, 2)), _found(EntityKind.OBSTACLE, (3, 2)),
             _found(EntityKind.FIRE, (3, 2))]
        )
        assert grid.code_at(Position(3, 2)) is OccupancyCode.FIRE

    def test_precedence_does_not_depend_on_input_order(
        self, builder: OccupancyGridBuilder
    ) -> None:
        fire = _found(EntityKind.FIRE, (3, 2))
        victim = _found(EntityKind.VICTIM, (3, 2))
        assert builder.build([fire, victim]).cells.tolist() == (
            builder.build([victim, fire]).cells.tolist()
        )

    def test_the_hospital_survives_a_detection_on_top_of_it(
        self, builder: OccupancyGridBuilder
    ) -> None:
        """Losing the drop-off point would strand every victim already aboard."""
        grid = builder.build([_found(EntityKind.OBSTACLE, (0, 0))])
        assert grid.code_at(Position(0, 0)) is OccupancyCode.HOSPITAL

    def test_precedence_matches_the_ground_truth_grid(self, city_map: CityMap) -> None:
        """The builder and the answer key must resolve a tile the same way.

        A victim pinned in burning rubble is the contested case: if these two
        disagreed, the Phase 3.3 metric would be measuring the disagreement
        instead of the detector.
        """
        contested = Position(3, 2)
        city_map.victims[0].position = contested
        city_map.fires[0].position = contested
        city_map.fires[0].radius = 0

        truth = OccupancyGrid.from_city_map(city_map)
        belief = OccupancyGridBuilder.from_city_map(city_map).build(
            [_found(EntityKind.FIRE, (3, 2)), _found(EntityKind.VICTIM, (3, 2))],
            vehicle_position=city_map.vehicle.position,
        )
        assert belief.code_at(contested) is truth.code_at(contested)


class TestTheVehicle:
    def test_the_vehicle_is_stamped_when_given(self, builder: OccupancyGridBuilder) -> None:
        grid = builder.build([], vehicle_position=Position(2, 2))
        assert grid.code_at(Position(2, 2)) is OccupancyCode.VEHICLE

    def test_omitting_the_vehicle_leaves_the_grid_alone(
        self, builder: OccupancyGridBuilder
    ) -> None:
        """The builder is also used to score a grid, where no vehicle belongs."""
        assert builder.build([]).positions_with(OccupancyCode.VEHICLE) == []

    def test_fire_outranks_the_vehicle_marker(self, builder: OccupancyGridBuilder) -> None:
        """VehicleController reads this grid to decide whether it is burning."""
        grid = builder.build(
            [_found(EntityKind.FIRE, (2, 2))], vehicle_position=Position(2, 2)
        )
        assert grid.code_at(Position(2, 2)) is OccupancyCode.FIRE

    def test_a_vehicle_off_the_map_is_ignored(self, builder: OccupancyGridBuilder) -> None:
        grid = builder.build([], vehicle_position=Position(99, 99))
        assert grid.positions_with(OccupancyCode.VEHICLE) == []


class TestConfidenceFiltering:
    def test_low_confidence_detections_are_discarded(self, city_map: CityMap) -> None:
        builder = OccupancyGridBuilder.from_city_map(city_map, min_confidence=0.5)
        grid = builder.build([_found(EntityKind.FIRE, (2, 2), confidence=0.4)])
        assert grid.positions_with(OccupancyCode.FIRE) == []

    def test_detections_on_the_threshold_are_kept(self, city_map: CityMap) -> None:
        builder = OccupancyGridBuilder.from_city_map(city_map, min_confidence=0.5)
        grid = builder.build([_found(EntityKind.FIRE, (2, 2), confidence=0.5)])
        assert grid.positions_with(OccupancyCode.FIRE) == [Position(2, 2)]

    def test_the_default_trusts_the_detector(self, builder: OccupancyGridBuilder) -> None:
        """Choosing a threshold is the composition root's policy, not the builder's."""
        assert builder.min_confidence == 0.0
        grid = builder.build([_found(EntityKind.VICTIM, (4, 2), confidence=0.01)])
        assert grid.code_at(Position(4, 2)) is OccupancyCode.VICTIM

    @pytest.mark.parametrize("threshold", [-0.1, 1.5])
    def test_an_impossible_threshold_is_rejected(
        self, builder: OccupancyGridBuilder, threshold: float
    ) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            OccupancyGridBuilder(terrain=builder.terrain, min_confidence=threshold)
