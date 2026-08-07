"""The whole Phase 3 chain, end to end, against the real city (Phase 3.3).

::

    SensorRig -> (ground truth as a perfect detector) -> DetectionMerger
              -> OccupancyGridBuilder -> GridComparison vs from_city_map

Feeding the rasterizer's own annotations in place of a model's predictions
is the point, exactly as in the 3.2 pipeline test. It makes the detector
perfect by construction, so **every disagreement these tests find is a
defect in the projection, the merge, or the precedence rules** — never in
the weights. Scoring the real detector is a separate exercise with its own
script (``scripts/evaluate_grid.py``).

One structural disagreement survives a perfect detector, and
:class:`TestFireIsOverMarkedNotUnderMarked` is where it is pinned down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import SensorConfig
from sentry_ai.domain.entities import Obstacle, Position
from sentry_ai.domain.enums import TerrainType, VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.navigation.astar import AStarPlanner
from sentry_ai.perception.grid_builder import OccupancyGridBuilder
from sentry_ai.perception.grid_metrics import GridComparison
from sentry_ai.perception.merger import CameraObservation, DetectionMerger
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig

_SENSOR_CONFIG = "configs/sensors.yaml"


@pytest.fixture
def loader(project_root: Path) -> ConfigLoader:
    return ConfigLoader(project_root=project_root)


@pytest.fixture
def sensor_config(loader: ConfigLoader) -> SensorConfig:
    return loader.load_sensor_config(_SENSOR_CONFIG)


@pytest.fixture
def city_map(loader: ConfigLoader) -> CityMap:
    return CityMap.from_config(loader.load_yaml("configs/maps/city_default.yaml"))


@pytest.fixture
def rig(loader: ConfigLoader, sensor_config: SensorConfig) -> SensorRig:
    return SensorRig.from_config(sensor_config, SensorPalette.from_config(loader, _SENSOR_CONFIG))


def _belief(rig: SensorRig, city_map: CityMap, builder: OccupancyGridBuilder) -> OccupancyGrid:
    """Run the full chain with a perfect detector and return the belief grid."""
    merged = DetectionMerger().merge(
        [CameraObservation(frame.view, frame.annotations) for frame in rig.capture_cctv(city_map)]
    )
    return builder.build(merged, vehicle_position=city_map.vehicle.position)


@pytest.fixture
def builder(city_map: CityMap) -> OccupancyGridBuilder:
    return OccupancyGridBuilder.from_city_map(city_map)


@pytest.fixture
def comparison(
    rig: SensorRig, city_map: CityMap, builder: OccupancyGridBuilder
) -> GridComparison:
    return GridComparison.between(
        OccupancyGrid.from_city_map(city_map), _belief(rig, city_map, builder)
    )


class TestAPerfectDetectorRecoversThePeople:
    def test_no_victim_is_missed(self, comparison: GridComparison) -> None:
        """The one failure this system may never make."""
        assert comparison.missed_victims == frozenset()
        assert comparison.finds_every_victim

    def test_no_victim_is_invented(self, comparison: GridComparison) -> None:
        assert comparison.phantom_victims == frozenset()

    def test_victims_land_on_exactly_the_right_tiles(
        self, comparison: GridComparison, city_map: CityMap
    ) -> None:
        score = comparison.scores[OccupancyCode.VICTIM]
        trapped = sum(1 for v in city_map.victims if v.status is VictimStatus.TRAPPED)
        assert score.true_positives == trapped
        assert (score.precision, score.recall) == (1.0, 1.0)

    def test_debris_is_recovered_exactly(self, comparison: GridComparison) -> None:
        """Debris is annotated per tile, so the box-to-tile projection is exact."""
        assert comparison.scores[OccupancyCode.DEBRIS].iou == 1.0

    def test_the_hospital_and_the_vehicle_survive_the_round_trip(
        self, comparison: GridComparison
    ) -> None:
        assert comparison.scores[OccupancyCode.HOSPITAL].iou == 1.0
        assert comparison.scores[OccupancyCode.VEHICLE].iou == 1.0


class TestFireIsOverMarkedNotUnderMarked:
    """The one disagreement a perfect detector cannot remove.

    Ground truth stamps a fire as a Euclidean *disc*
    (``OccupancyGrid.mark_radius``). A camera reports an axis-aligned
    *box*, and a disc's bounding box includes its corners — so the belief
    marks a ring of tiles that are not actually burning.

    This is information loss inherent to bounding boxes, not a bug, and it
    errs in the safe direction: the belief never under-states a fire. The
    builder deliberately does not try to re-inscribe a disc, because
    guessing a radius wrong would convert a harmless detour into a vehicle
    driving into a fire.
    """

    def test_every_burning_tile_is_believed_to_be_burning(
        self, comparison: GridComparison
    ) -> None:
        assert comparison.scores[OccupancyCode.FIRE].recall == 1.0

    def test_the_belief_claims_more_fire_than_there_is(
        self, comparison: GridComparison
    ) -> None:
        score = comparison.scores[OccupancyCode.FIRE]
        assert score.false_positives > 0
        assert score.false_negatives == 0

    def test_the_error_is_never_a_hazard_the_vehicle_would_drive_into(
        self, comparison: GridComparison
    ) -> None:
        assert comparison.missed_hazards == frozenset()
        assert comparison.is_drivable

    def test_the_phantom_tiles_all_border_a_real_fire(
        self, comparison: GridComparison, city_map: CityMap
    ) -> None:
        """Bounding-box corners, not scattered noise."""
        truth = OccupancyGrid.from_city_map(city_map)
        burning = set(truth.positions_with(OccupancyCode.FIRE))
        for phantom in comparison.phantom_obstacles:
            assert any(phantom.distance_to(tile) <= 2.0 for tile in burning)


class TestTheMissionStillWorksOnTheBelief:
    """The metric that actually matters: can the vehicle still do its job?

    A grid can score badly on cells and drive fine, or score well and strand
    someone. These ask A\\* directly, using the unchanged Phase 2 planner.
    """

    def test_every_victim_is_still_reachable(
        self, rig: SensorRig, city_map: CityMap, builder: OccupancyGridBuilder
    ) -> None:
        belief = _belief(rig, city_map, builder)
        planner = AStarPlanner()
        for victim in city_map.victims:
            route = planner.plan(belief, city_map.vehicle.position, victim.position)
            assert not route.is_empty, f"{victim.victim_id} unreachable on the belief grid"

    def test_the_hospital_is_still_reachable(
        self, rig: SensorRig, city_map: CityMap, builder: OccupancyGridBuilder
    ) -> None:
        belief = _belief(rig, city_map, builder)
        route = AStarPlanner().plan(
            belief, city_map.vehicle.position, city_map.safe_zone.position
        )
        assert not route.is_empty

    def test_routes_cost_no_less_than_they_do_on_ground_truth(
        self, rig: SensorRig, city_map: CityMap, builder: OccupancyGridBuilder
    ) -> None:
        """A conservative belief can only ever make a route longer, never shorter.

        A cheaper route on the belief grid would mean it had *erased* a
        hazard the planner should have paid to avoid.
        """
        belief = _belief(rig, city_map, builder)
        truth = OccupancyGrid.from_city_map(city_map)
        planner = AStarPlanner()
        for victim in city_map.victims:
            on_truth = planner.plan(truth, city_map.vehicle.position, victim.position)
            on_belief = planner.plan(belief, city_map.vehicle.position, victim.position)
            assert on_belief.cost >= on_truth.cost - 1e-9


class TestTheTerrainDetectionSplit:
    def test_no_building_is_ever_believed_drivable(
        self, comparison: GridComparison, city_map: CityMap
    ) -> None:
        """Buildings are copied from the survey, never detected, so none can go missing.

        Their *code* can still change: a fire's bounding box may stamp
        ``FIRE`` over a wall it overlaps, which is why building recall is
        below 1.0. That costs nothing — both codes are impassable, and the
        planner only ever asks whether it may pass.
        """
        truth = OccupancyGrid.from_city_map(city_map)
        walls = set(truth.positions_with(OccupancyCode.BUILDING))
        assert walls
        assert walls.isdisjoint(comparison.missed_hazards)

    def test_every_lost_building_tile_was_overwritten_by_fire(
        self, comparison: GridComparison, rig: SensorRig, city_map: CityMap,
        builder: OccupancyGridBuilder,
    ) -> None:
        """Pins the cause, so a genuine terrain regression cannot hide here."""
        truth = OccupancyGrid.from_city_map(city_map)
        belief = _belief(rig, city_map, builder)
        lost = [
            tile
            for tile in truth.positions_with(OccupancyCode.BUILDING)
            if belief.code_at(tile) is not OccupancyCode.BUILDING
        ]
        assert lost, "no overlap left to explain — revisit this test"
        assert all(belief.code_at(tile) is OccupancyCode.FIRE for tile in lost)

    def test_a_collapse_after_the_survey_is_still_found(
        self, rig: SensorRig, city_map: CityMap, builder: OccupancyGridBuilder
    ) -> None:
        """The case that justifies the whole split.

        The builder's terrain was surveyed before this collapse, so the
        static half of the grid is now stale. The debris still has to appear
        — via the cameras — or the vehicle drives into a blocked street.
        """
        target = next(
            tile
            for tile in (Position(x, y) for y in range(city_map.height)
                         for x in range(city_map.width))
            if city_map.tile_at(tile) is TerrainType.ROAD
            and builder.terrain.is_traversable(tile)
        )
        assert _belief(rig, city_map, builder).is_traversable(target)

        city_map.terrain[target] = TerrainType.COLLAPSED_BUILDING
        city_map.obstacles.append(
            Obstacle(
                obstacle_id="collapse_test",
                position=target,
                kind=TerrainType.COLLAPSED_BUILDING,
            )
        )

        assert builder.terrain.is_traversable(target), "the survey is deliberately stale"
        assert not _belief(rig, city_map, builder).is_traversable(target)

    def test_a_rescued_victim_leaves_the_belief(
        self, rig: SensorRig, city_map: CityMap, builder: OccupancyGridBuilder
    ) -> None:
        assert _belief(rig, city_map, builder).positions_with(OccupancyCode.VICTIM)
        for victim in city_map.victims:
            victim.status = VictimStatus.RESCUED
        assert _belief(rig, city_map, builder).positions_with(OccupancyCode.VICTIM) == []


class TestScoresAreStable:
    def test_the_same_world_scores_identically_twice(
        self, rig: SensorRig, city_map: CityMap, builder: OccupancyGridBuilder
    ) -> None:
        truth = OccupancyGrid.from_city_map(city_map)
        first = GridComparison.between(truth, _belief(rig, city_map, builder))
        second = GridComparison.between(truth, _belief(rig, city_map, builder))
        assert first.cell_agreement == second.cell_agreement
        assert first.phantom_obstacles == second.phantom_obstacles

    def test_traversability_agreement_clears_the_bar_a_mission_needs(
        self, comparison: GridComparison
    ) -> None:
        """Loose on purpose — this guards against collapse, not a tuned number.

        The exact figure is recorded in docs/architecture/phase3-detection.md;
        pinning it here would make every camera-layout change a test failure.
        """
        assert comparison.traversability_agreement > 0.95
