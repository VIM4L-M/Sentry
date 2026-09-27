"""The denoiser's place in the camera pipeline (Phase 4).

``DetectedGridSource`` gains one optional stage::

    SensorRig -> FrameDegrader -> [IDenoiser] -> IFrameObserver -> ...

These tests pin down *where* it sits and that inserting it changes nothing
else. They use recording fakes rather than a trained network, so they need
no Torch: what is being tested is the plumbing, and a fake that remembers
what it was handed is the most direct witness of the plumbing there is.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import SensorConfig
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.perception import IDenoiser
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

_SENSOR_CONFIG = "configs/sensors.yaml"


class _RecordingDenoiser(IDenoiser):
    """Remembers every frame it was handed and returns a marked copy."""

    MARK = 7

    def __init__(self) -> None:
        self.seen: list[NDArray[np.uint8]] = []

    def denoise(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        self.seen.append(frame.copy())
        marked = frame.copy()
        marked[0, 0] = self.MARK
        return marked


class _PassThroughDenoiser(IDenoiser):
    def denoise(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return frame.copy()


class _RecordingObserver(IFrameObserver):
    """The ground-truth observer, but remembering the pixels it was shown."""

    def __init__(self) -> None:
        self.seen: list[NDArray[np.uint8]] = []
        self._inner = GroundTruthObserver()

    def observe(self, frames: Sequence[CameraFrame]) -> list[CameraObservation]:
        self.seen.extend(frame.pixels.copy() for frame in frames)
        return self._inner.observe(frames)


@pytest.fixture
def loader(project_root: Path) -> ConfigLoader:
    return ConfigLoader(project_root=project_root)


@pytest.fixture
def sensor_config(loader: ConfigLoader) -> SensorConfig:
    return loader.load_sensor_config(_SENSOR_CONFIG)


@pytest.fixture
def rig(loader: ConfigLoader, sensor_config: SensorConfig) -> SensorRig:
    return SensorRig.from_config(sensor_config, SensorPalette.from_config(loader, _SENSOR_CONFIG))


@pytest.fixture
def city(loader: ConfigLoader) -> CityMap:
    app_config = loader.load_app_config("configs/app.yaml")
    return CityMap.from_config(loader.load_yaml(app_config.map_config_path))


def _source(
    rig: SensorRig,
    sensor_config: SensorConfig,
    city: CityMap,
    observer: IFrameObserver,
    denoiser: IDenoiser | None,
) -> DetectedGridSource:
    return DetectedGridSource(
        rig=rig,
        observer=observer,
        builder=OccupancyGridBuilder.from_city_map(city),
        degrader=FrameDegrader(sensor_config.degradation, np.random.default_rng(0)),
        denoiser=denoiser,
    )


class TestTheDenoiserSitsBetweenDegraderAndDetector:
    def test_it_cleans_every_cctv_frame_once(
        self, rig: SensorRig, sensor_config: SensorConfig, city: CityMap
    ) -> None:
        denoiser = _RecordingDenoiser()
        _source(rig, sensor_config, city, GroundTruthObserver(), denoiser).grid_for(
            city, city.vehicle.position
        )
        assert len(denoiser.seen) == len(rig.cctv_views)

    def test_it_receives_degraded_frames_not_clean_ones(
        self, rig: SensorRig, sensor_config: SensorConfig, city: CityMap
    ) -> None:
        denoiser = _RecordingDenoiser()
        _source(rig, sensor_config, city, GroundTruthObserver(), denoiser).grid_for(
            city, city.vehicle.position
        )
        clean = rig.capture_cctv(city)
        assert not any(
            np.array_equal(seen, frame.pixels)
            for seen, frame in zip(denoiser.seen, clean, strict=True)
        )

    def test_the_detector_sees_the_denoisers_output(
        self, rig: SensorRig, sensor_config: SensorConfig, city: CityMap
    ) -> None:
        observer = _RecordingObserver()
        _source(rig, sensor_config, city, observer, _RecordingDenoiser()).grid_for(
            city, city.vehicle.position
        )
        assert observer.seen
        assert all(int(frame[0, 0, 0]) == _RecordingDenoiser.MARK for frame in observer.seen)


class TestInsertingItChangesNothingElse:
    def test_a_pass_through_denoiser_leaves_the_grid_identical(
        self, rig: SensorRig, sensor_config: SensorConfig, city: CityMap
    ) -> None:
        without = _source(rig, sensor_config, city, GroundTruthObserver(), None)
        with_one = _source(rig, sensor_config, city, GroundTruthObserver(), _PassThroughDenoiser())

        position = city.vehicle.position
        expected = without.grid_for(city, position).as_array()
        assert np.array_equal(with_one.grid_for(city, position).as_array(), expected)

    def test_annotations_survive_denoising(
        self, rig: SensorRig, sensor_config: SensorConfig, city: CityMap
    ) -> None:
        """The denoiser changes pixels; the ground truth rides along untouched."""
        observed: list[CameraObservation] = []

        class _Capture(IFrameObserver):
            def observe(self, frames: Sequence[CameraFrame]) -> list[CameraObservation]:
                result = GroundTruthObserver().observe(frames)
                observed.extend(result)
                return result

        _source(rig, sensor_config, city, _Capture(), _RecordingDenoiser()).grid_for(
            city, city.vehicle.position
        )
        expected = [frame.annotations for frame in rig.capture_cctv(city)]
        assert [observation.detections for observation in observed] == expected
