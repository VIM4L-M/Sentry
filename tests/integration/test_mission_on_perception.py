"""A whole mission driven by what the cameras see (Phase 3.4 / 3.5).

The Phase 3 deliverable here is not new navigation code — it is the
*absence* of it. ``AStarPlanner``, ``MissionController``, ``VehicleController``
and ``WaypointFollower`` are unchanged Phase 2 code, and swapping the grid's
producer underneath them is a constructor argument:

    build_mission(..., grid_source=DetectedGridSource(...))

If that swap required touching routing or the mission state machine, the
seam ADR 0002 describes would have been broken. These tests run real
missions both ways and compare.

Perception here is :class:`GroundTruthObserver` — a perfect detector — for
the same reason the 3.2 and 3.3 pipeline tests use it: it isolates the
*plumbing* from the weights. A mission that fails with a perfect detector
has a wiring bug. Scoring the real detector is
``scripts/evaluate_grid.py``'s job, and the numbers live in
docs/architecture/phase3-detection.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import AppConfig, SensorConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.interfaces.world import IOccupancyGridSource
from sentry_ai.perception.grid_builder import OccupancyGridBuilder
from sentry_ai.perception.grid_source import (
    DetectedGridSource,
    GroundTruthObserver,
    IFrameObserver,
)
from sentry_ai.perception.merger import CameraObservation
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.frame import CameraFrame
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.simulation.factory import Mission, build_mission
from sentry_ai.simulation.grid_source import GroundTruthGridSource
from sentry_ai.simulation.mission import MissionPhase

_SENSOR_CONFIG = "configs/sensors.yaml"
_MAX_TICKS = 600


@pytest.fixture
def loader(project_root: Path) -> ConfigLoader:
    return ConfigLoader(project_root=project_root)


@pytest.fixture
def app_config(loader: ConfigLoader) -> AppConfig:
    return loader.load_app_config("configs/app.yaml")


@pytest.fixture
def sensor_config(loader: ConfigLoader) -> SensorConfig:
    return loader.load_sensor_config(_SENSOR_CONFIG)


@pytest.fixture
def rig(loader: ConfigLoader, sensor_config: SensorConfig) -> SensorRig:
    return SensorRig.from_config(sensor_config, SensorPalette.from_config(loader, _SENSOR_CONFIG))


def _fresh_city(loader: ConfigLoader, app_config: AppConfig) -> CityMap:
    """A brand-new city — a mission mutates the one it runs in."""
    return CityMap.from_config(loader.load_yaml(app_config.map_config_path))


def _mission(
    loader: ConfigLoader,
    app_config: AppConfig,
    city_map: CityMap,
    grid_source: IOccupancyGridSource,
    seed: int,
) -> Mission:
    return build_mission(
        city_map=city_map,
        simulation_config=loader.load_simulation_config(app_config.simulation_config_path),
        vehicle_config=loader.load_vehicle_config(app_config.vehicle_config_path),
        hazard_seed=seed,
        grid_source=grid_source,
    )


def _perception(
    rig: SensorRig, sensor_config: SensorConfig, city_map: CityMap, degrade: bool = False
) -> DetectedGridSource:
    """The camera pipeline, surveying ``city_map``'s layout at mission start."""
    return DetectedGridSource(
        rig=rig,
        observer=GroundTruthObserver(),
        builder=OccupancyGridBuilder.from_city_map(city_map),
        degrader=(
            FrameDegrader(sensor_config.degradation, np.random.default_rng(0))
            if degrade
            else None
        ),
    )


class TestAMissionRunsOnPerception:
    def test_the_mission_completes(
        self, loader: ConfigLoader, app_config: AppConfig, rig: SensorRig,
        sensor_config: SensorConfig,
    ) -> None:
        """The headline Phase 3 claim: the vehicle does its job on a believed map."""
        city = _fresh_city(loader, app_config)
        mission = _mission(
            loader, app_config, city, _perception(rig, sensor_config, city), seed=7
        )
        stats = mission.engine.run(max_ticks=_MAX_TICKS)

        assert mission.controller.phase is MissionPhase.COMPLETED
        assert stats.victims_rescued > 0

    def test_it_rescues_as_many_as_ground_truth_does(
        self, loader: ConfigLoader, app_config: AppConfig, rig: SensorRig,
        sensor_config: SensorConfig,
    ) -> None:
        """Perception must not cost the mission anyone.

        Same map, same hazard seed, same planner — the only difference is
        where the grid came from.
        """
        truth_city = _fresh_city(loader, app_config)
        truth = _mission(loader, app_config, truth_city, GroundTruthGridSource(), seed=7)
        expected = truth.engine.run(max_ticks=_MAX_TICKS)

        seen_city = _fresh_city(loader, app_config)
        believed = _mission(
            loader, app_config, seen_city, _perception(rig, sensor_config, seen_city), seed=7
        )
        actual = believed.engine.run(max_ticks=_MAX_TICKS)

        assert actual.victims_rescued == expected.victims_rescued
        assert actual.victims_lost <= expected.victims_lost

    def test_it_survives_degraded_frames(
        self, loader: ConfigLoader, app_config: AppConfig, rig: SensorRig,
        sensor_config: SensorConfig,
    ) -> None:
        """Smoke and sensor noise are between the camera and the belief."""
        city = _fresh_city(loader, app_config)
        mission = _mission(
            loader, app_config, city,
            _perception(rig, sensor_config, city, degrade=True), seed=7,
        )
        stats = mission.engine.run(max_ticks=_MAX_TICKS)

        assert mission.controller.phase is MissionPhase.COMPLETED
        assert stats.victims_rescued > 0

    @pytest.mark.parametrize("seed", [1, 3, 11])
    def test_it_holds_across_different_disasters(
        self, loader: ConfigLoader, app_config: AppConfig, rig: SensorRig,
        sensor_config: SensorConfig, seed: int,
    ) -> None:
        """One seed proves nothing — hazards are where the belief gets stale."""
        city = _fresh_city(loader, app_config)
        mission = _mission(
            loader, app_config, city, _perception(rig, sensor_config, city), seed=seed
        )
        mission.engine.run(max_ticks=_MAX_TICKS)
        assert mission.controller.phase is MissionPhase.COMPLETED


class TestTheSeamHolds:
    def test_the_controller_never_learns_which_source_it_has(
        self, loader: ConfigLoader, app_config: AppConfig, rig: SensorRig,
        sensor_config: SensorConfig,
    ) -> None:
        """Both missions are the same class, wired the same way."""
        city = _fresh_city(loader, app_config)
        believed = _mission(
            loader, app_config, city, _perception(rig, sensor_config, city), seed=7
        )
        truth_city = _fresh_city(loader, app_config)
        truth = _mission(loader, app_config, truth_city, GroundTruthGridSource(), seed=7)

        assert type(believed.controller) is type(truth.controller)
        assert type(believed.controller.planner) is type(truth.controller.planner)

    def test_the_grid_is_rebuilt_through_the_source_every_tick(
        self, loader: ConfigLoader, app_config: AppConfig, rig: SensorRig,
        sensor_config: SensorConfig,
    ) -> None:
        """A cached grid would let a collapse go unnoticed until the next replan."""
        city = _fresh_city(loader, app_config)
        source = _CountingSource(_perception(rig, sensor_config, city))
        mission = _mission(loader, app_config, city, source, seed=7)

        before = source.calls
        for _ in range(5):
            mission.engine.tick()
        assert source.calls >= before + 5

    def test_the_default_is_still_ground_truth(
        self, loader: ConfigLoader, app_config: AppConfig
    ) -> None:
        """Phase 2 callers that pass no source must be unaffected."""
        city = _fresh_city(loader, app_config)
        mission = build_mission(
            city_map=city,
            simulation_config=loader.load_simulation_config(app_config.simulation_config_path),
            vehicle_config=loader.load_vehicle_config(app_config.vehicle_config_path),
        )
        assert isinstance(mission.controller.grid_source, GroundTruthGridSource)


class TestThePipelineIsLoadBearing:
    """Guards against the swap quietly doing nothing.

    A perfect detector now reproduces ground truth cell for cell, which is
    the result 3.3 was aiming at — and it means "the believed grid differs
    from truth" can no longer serve as proof that perception ran at all.
    These prove it a better way: break the cameras, and the grid must lose
    exactly what the cameras were supplying.
    """

    def test_perception_reproduces_ground_truth_exactly(
        self, loader: ConfigLoader, app_config: AppConfig, rig: SensorRig,
        sensor_config: SensorConfig,
    ) -> None:
        city = _fresh_city(loader, app_config)
        believed = _perception(rig, sensor_config, city).grid_for(city, city.vehicle.position)
        truth = GroundTruthGridSource().grid_for(city, city.vehicle.position)
        assert np.array_equal(believed.cells, truth.cells)

    def test_a_blind_observer_loses_everything_dynamic(
        self, loader: ConfigLoader, app_config: AppConfig, rig: SensorRig,
        sensor_config: SensorConfig,
    ) -> None:
        """If the grid ignored the observer, this would look identical."""
        city = _fresh_city(loader, app_config)
        blind = DetectedGridSource(
            rig=rig,
            observer=_BlindObserver(),
            builder=OccupancyGridBuilder.from_city_map(city),
        )
        grid = blind.grid_for(city, city.vehicle.position)

        assert grid.positions_with(OccupancyCode.VICTIM) == []
        assert grid.positions_with(OccupancyCode.FIRE) == []

    def test_the_static_survey_survives_a_blind_observer(
        self, loader: ConfigLoader, app_config: AppConfig, rig: SensorRig,
        sensor_config: SensorConfig,
    ) -> None:
        """Terrain comes from the map, so blinding the cameras must not erase it."""
        city = _fresh_city(loader, app_config)
        blind = DetectedGridSource(
            rig=rig,
            observer=_BlindObserver(),
            builder=OccupancyGridBuilder.from_city_map(city),
        )
        grid = blind.grid_for(city, city.vehicle.position)
        assert grid.positions_with(OccupancyCode.BUILDING)
        assert grid.positions_with(OccupancyCode.HOSPITAL)

    def test_every_victim_is_on_the_believed_map(
        self, loader: ConfigLoader, app_config: AppConfig, rig: SensorRig,
        sensor_config: SensorConfig,
    ) -> None:
        city = _fresh_city(loader, app_config)
        believed = _perception(rig, sensor_config, city).grid_for(city, city.vehicle.position)
        found = set(believed.positions_with(OccupancyCode.VICTIM))
        assert {victim.position for victim in city.victims} == found


class _BlindObserver(IFrameObserver):
    """A camera network that reports nothing, ever."""

    def observe(self, frames: Sequence[CameraFrame]) -> list[CameraObservation]:
        return [CameraObservation(frame.view, ()) for frame in frames]


class _CountingSource(IOccupancyGridSource):
    """Counts how often the mission asked for a fresh grid."""

    def __init__(self, inner: IOccupancyGridSource) -> None:
        self._inner = inner
        self.calls = 0

    def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
        self.calls += 1
        return self._inner.grid_for(city_map, vehicle_position)
