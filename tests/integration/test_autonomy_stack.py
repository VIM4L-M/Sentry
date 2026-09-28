"""The Phase 8 stack, wired end to end from stand-in models.

Real weights are too slow and too large for a unit suite, and what is under
test here is the wiring, not the models: that every stage of the stack is
connected, timed, and recorded, that the camera-map throttle and lag are
honoured, and that a mission record survives a round trip. Stand-ins are
used in every model slot — a detector that finds nothing, the waypoint
follower as the DQN, a fixed behaviour prediction, fusion that passes the
local decision through.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from sentry_ai.autonomy.profiling import Profiler, TimedDetector
from sentry_ai.autonomy.record import MissionRecord
from sentry_ai.autonomy.stack import TICK_STAGE, AutonomyStack
from sentry_ai.common.exceptions import AssetNotFoundError, ConfigValidationError
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import AutonomyConfig
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.interfaces.decision import FinalAction, IDecisionFusion
from sentry_ai.interfaces.navigation import LocalDecision, LocalObservation
from sentry_ai.interfaces.perception import Detection, IVisionDetector, WorldDetection
from sentry_ai.interfaces.sequence import (
    BehaviourClass,
    BehaviourSignal,
    IMotionPredictor,
    VehicleState,
)
from sentry_ai.perception.grid_source import ModelObserver
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.simulation.grid_source import GroundTruthGridSource, ThrottledGridSource
from sentry_ai.simulation.waypoint_follower import WaypointFollower

_ROOT = Path(__file__).resolve().parents[2]


class _Nothing(IVisionDetector):
    """Sees nothing; counts how it was asked."""

    def __init__(self) -> None:
        self.single = 0
        self.batched = 0

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        self.single += 1
        return []

    def detect_many(self, frames: Sequence[NDArray[np.uint8]]) -> list[list[Detection]]:
        self.batched += 1
        return [[] for _ in frames]


class _Steady(IMotionPredictor):
    def predict(self, state_history: Sequence[VehicleState]) -> BehaviourSignal:
        return BehaviourSignal(BehaviourClass.ADVANCE, 0.9)


class _PassThrough(IDecisionFusion):
    def fuse(
        self,
        observation: LocalObservation,
        sightings: Sequence[WorldDetection],
        behaviour: BehaviourSignal,
        local_decision: LocalDecision,
    ) -> FinalAction:
        return FinalAction(local_decision.action, 1.0)


@pytest.fixture(scope="module")
def loader() -> ConfigLoader:
    return ConfigLoader(project_root=_ROOT)


def _stack(loader: ConfigLoader, **overrides: object) -> tuple[AutonomyStack, _Nothing]:
    sensors = loader.load_sensor_config("configs/sensors.yaml")
    rig = SensorRig.from_config(sensors, SensorPalette.from_config(loader, "configs/sensors.yaml"))
    detector = _Nothing()
    config = replace(AutonomyConfig(), **overrides)  # type: ignore[arg-type]
    stack = AutonomyStack(
        config=config,
        sensor_config=sensors,
        rig=rig,
        cctv_detector=detector,
        denoiser=None,
        onboard_detector=detector,
        dqn=WaypointFollower(),
        lstm=_Steady(),  # type: ignore[arg-type]
        fusion=_PassThrough(),  # type: ignore[arg-type]
    )
    return stack, detector


def _run(loader: ConfigLoader, stack: AutonomyStack, ticks: int) -> MissionRecord:
    app = loader.load_app_config("configs/app.yaml")
    run = stack.mission(
        CityMap.from_config(loader.load_yaml(app.map_config_path)),
        loader.load_simulation_config(app.simulation_config_path),  # type: ignore[arg-type]
        loader.load_vehicle_config(app.vehicle_config_path),  # type: ignore[arg-type]
        hazard_seed=1,
    )
    return run.run(max_ticks=ticks)


class TestTheStackIsWired:
    def test_every_stage_runs_and_is_timed(self, loader: ConfigLoader) -> None:
        stack, _ = _stack(loader)
        record = _run(loader, stack, ticks=15)
        stages = {timing["stage"] for timing in record.timings}
        assert {
            "cctv detect",
            "cctv map (total)",
            "onboard detect",
            "onboard (total)",
            "dqn",
            "lstm",
            "fusion",
            TICK_STAGE,
        } <= stages

    def test_the_cctv_frames_are_detected_as_one_batch(self, loader: ConfigLoader) -> None:
        stack, detector = _stack(loader)
        _run(loader, stack, ticks=5)
        assert detector.batched > 0

    def test_the_path_is_recorded_tick_by_tick(self, loader: ConfigLoader) -> None:
        stack, _ = _stack(loader)
        record = _run(loader, stack, ticks=10)
        assert len(record.path) == record.ticks + 1
        for (x0, y0), (x1, y1) in zip(record.path, record.path[1:], strict=False):
            assert abs(x1 - x0) + abs(y1 - y0) <= 1

    def test_throttling_perceives_less_often(self, loader: ConfigLoader) -> None:
        every_tick, _ = _stack(loader, perception_every=1)
        every_fourth, _ = _stack(loader, perception_every=4)
        calls = {}
        for name, stack in (("1", every_tick), ("4", every_fourth)):
            record = _run(loader, stack, ticks=20)
            calls[name] = next(t["calls"] for t in record.timings if t["stage"] == "cctv detect")
        assert calls["4"] < calls["1"] / 2


class TestThrottledGridSource:
    def test_it_reuses_the_map_between_refreshes(self, loader: ConfigLoader) -> None:
        app = loader.load_app_config("configs/app.yaml")
        city = CityMap.from_config(loader.load_yaml(app.map_config_path))
        throttled = ThrottledGridSource(GroundTruthGridSource(), every=3)
        grids = [throttled.grid_for(city, city.vehicle.position) for _ in range(7)]
        assert grids[0] is grids[1] is grids[2]
        assert grids[3] is not grids[2]
        assert grids[6] is not grids[5]

    def test_every_one_is_the_inner_source(self, loader: ConfigLoader) -> None:
        app = loader.load_app_config("configs/app.yaml")
        city = CityMap.from_config(loader.load_yaml(app.map_config_path))
        throttled = ThrottledGridSource(GroundTruthGridSource(), every=1)
        first = throttled.grid_for(city, city.vehicle.position)
        assert isinstance(first, OccupancyGrid)
        assert throttled.grid_for(city, city.vehicle.position) is not first

    def test_zero_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            ThrottledGridSource(GroundTruthGridSource(), every=0)


class TestBatchedDetection:
    def test_the_port_default_loops_detect(self) -> None:
        class _Counting(IVisionDetector):
            calls = 0

            def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
                _Counting.calls += 1
                return []

        frames = [np.zeros((4, 4, 3), dtype=np.uint8)] * 3
        assert _Counting().detect_many(frames) == [[], [], []]
        assert _Counting.calls == 3

    def test_the_model_observer_asks_for_one_batch(self, loader: ConfigLoader) -> None:
        sensors = loader.load_sensor_config("configs/sensors.yaml")
        rig = SensorRig.from_config(
            sensors, SensorPalette.from_config(loader, "configs/sensors.yaml")
        )
        app = loader.load_app_config("configs/app.yaml")
        city = CityMap.from_config(loader.load_yaml(app.map_config_path))
        detector = _Nothing()
        observations = ModelObserver(detector).observe(rig.capture_cctv(city))
        assert len(observations) == len(rig.cctv_views)
        assert (detector.batched, detector.single) == (1, 0)

    def test_timing_wraps_batches_too(self) -> None:
        profiler = Profiler()
        TimedDetector(_Nothing(), profiler, "x").detect_many([np.zeros((2, 2, 3), np.uint8)])
        assert profiler.timings()[0].calls == 1


class TestMissionRecord:
    def test_it_round_trips(self, loader: ConfigLoader, tmp_path: Path) -> None:
        stack, _ = _stack(loader)
        record = _run(loader, stack, ticks=8)
        path = record.save(tmp_path)
        assert path.name.endswith("_full-stack_seed1.json")
        assert MissionRecord.load(path) == record

    def test_it_is_plain_json(self, loader: ConfigLoader, tmp_path: Path) -> None:
        stack, _ = _stack(loader)
        document = json.loads(_run(loader, stack, ticks=3).save(tmp_path).read_text())
        assert document["format"] == 1 and "timings" in document and "path" in document

    def test_it_reports_the_mean_tick(self, loader: ConfigLoader) -> None:
        stack, _ = _stack(loader)
        assert _run(loader, stack, ticks=5).mean_tick_ms > 0.0

    def test_a_missing_record_fails_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(AssetNotFoundError):
            MissionRecord.load(tmp_path / "absent.json")


class TestProfiler:
    def test_it_counts_and_spreads_time(self) -> None:
        profiler = Profiler()
        for _ in range(3):
            with profiler.measure("stage"):
                pass
        (timing,) = profiler.timings()
        assert timing.calls == 3
        assert profiler.per_tick_ms("stage", ticks=3) == pytest.approx(timing.mean_ms)
        assert profiler.per_tick_ms("absent", ticks=3) == 0.0


class TestConfig:
    def test_the_shipped_file_loads(self, loader: ConfigLoader) -> None:
        config = loader.load_autonomy_config("configs/autonomy.yaml")
        assert config.models.dqn == _ROOT / "models/dqn/sentry/best.zip"
        assert config.perception_every == 1 and config.belief_lag == 0

    def test_the_denoiser_can_be_switched_off(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text("models:\n  denoiser: null\n", encoding="utf-8")
        assert ConfigLoader(tmp_path).load_autonomy_config("a.yaml").models.denoiser is None

    @pytest.mark.parametrize(
        "overrides",
        [
            {"perception_every": 0},
            {"belief_lag": -1},
            {"image_size": 250},
            {"rescue_threshold": 0.0},
        ],
    )
    def test_invalid_values_are_rejected(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ConfigValidationError):
            AutonomyConfig(**overrides)  # type: ignore[arg-type]
