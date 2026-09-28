"""The full autonomy stack: every trained model, wired and timed (Phase 8).

::

    CCTV x4 -> degrade -> denoise -> YOLO (batched) -> merge -> belief grid
                                                                  |
                                    A* command center <-----------+
                                          | route
    onboard camera -> degrade -> YOLO -> sightings --+
    DQN (local controller) --------------------------+--> MLP fusion -> action
    state history -> LSTM ---------------------------+

The vehicle moves through the real city; the command center plans on the
camera-built map (ADR 0003). Nothing here decides anything — every judgement
lives in the adapters built in Phases 3-7. This module loads them once,
connects them per mission, wraps each expensive stage in a timer, and
writes the mission down.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sentry_ai.autonomy.profiling import (
    Profiler,
    TimedController,
    TimedDenoiser,
    TimedDetector,
    TimedFusion,
    TimedGridSource,
    TimedPredictor,
    timed_call,
)
from sentry_ai.autonomy.record import MissionRecord
from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.config.schema import (
    AppConfig,
    AutonomyConfig,
    SensorConfig,
    SimulationConfig,
    VehicleConfig,
)
from sentry_ai.decision.dqn_controller import DqnLocalController
from sentry_ai.decision.fused_controller import FusedLocalController
from sentry_ai.decision.fusion import MlpFusion
from sentry_ai.domain.entities import Position
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.navigation import ILocalController, LocalDecision, LocalObservation
from sentry_ai.interfaces.perception import IDenoiser, IVisionDetector
from sentry_ai.interfaces.world import IOccupancyGridSource
from sentry_ai.perception.grid_builder import OccupancyGridBuilder
from sentry_ai.perception.grid_source import DetectedGridSource, ModelObserver
from sentry_ai.perception.onboard import OnboardSensing
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.sequence.lstm_predictor import LstmMotionPredictor
from sentry_ai.simulation.factory import Mission, build_mission
from sentry_ai.simulation.grid_source import LaggedGridSource, ThrottledGridSource

logger = get_logger(__name__)

#: The stage every tick's wall-clock time is charged to.
TICK_STAGE = "tick (total)"


class PathRecorder(ILocalController):
    """Passes decisions through unchanged, noting where the vehicle was each tick."""

    def __init__(self, inner: ILocalController, start: Position) -> None:
        self._inner = inner
        self.path: list[tuple[int, int]] = [start.as_tuple()]

    def decide(self, observation: LocalObservation) -> LocalDecision:
        decision = self._inner.decide(observation)
        self.path.append(observation.position.as_tuple())
        return decision


@dataclass
class AutonomousMission:
    """One mission driven by the full stack, with its timer and its path."""

    mission: Mission
    controller: FusedLocalController
    driver: ILocalController
    recorder: PathRecorder
    profiler: Profiler
    seed: int | None
    settings: dict[str, Any]

    def tick(self) -> bool:
        """Run one tick, timed. Returns ``False`` once the mission is over."""
        with self.profiler.measure(TICK_STAGE):
            result = self.mission.engine.tick()
        return result is not None

    def run(self, max_ticks: int) -> MissionRecord:
        """Drive to the end (or ``max_ticks``) and return the record."""
        for _ in range(max_ticks):
            if not self.tick():
                break
        return self.record()

    def record(self, label: str = "full stack") -> MissionRecord:
        """Write down what happened so far."""
        engine = self.mission.engine
        ticks = max(1, engine.stats.ticks)
        timings = [
            {
                "stage": timing.stage,
                "calls": timing.calls,
                "mean_ms": round(timing.mean_ms, 3),
                "per_tick_ms": round(self.profiler.per_tick_ms(timing.stage, ticks), 3),
            }
            for timing in self.profiler.timings()
        ]
        log = self.mission.controller.events
        return MissionRecord(
            label=label,
            seed=self.seed,
            outcome=self.mission.controller.phase.value,
            failure_reason=engine.stats.failure_reason,
            stats=asdict(engine.stats) | {
                "vehicle_health": self.mission.city_map.vehicle.health_percent,
                "vehicle_battery": self.mission.city_map.vehicle.battery_percent,
            },
            path=list(self.recorder.path),
            timings=timings,
            events=[
                {
                    "at": round(event.at_seconds, 2),
                    "kind": event.kind.value,
                    "message": event.message,
                }
                for event in log.recent(log.capacity)
            ],
            settings=self.settings,
        )


class AutonomyStack:
    """Every trained model, loaded once, ready to drive any number of missions."""

    def __init__(
        self,
        config: AutonomyConfig,
        sensor_config: SensorConfig,
        rig: SensorRig,
        cctv_detector: IVisionDetector,
        denoiser: IDenoiser | None,
        onboard_detector: IVisionDetector,
        dqn: ILocalController,
        lstm: LstmMotionPredictor,
        fusion: MlpFusion,
    ) -> None:
        """Hold already-loaded adapters. Most callers want :meth:`load`."""
        self.config = config
        self._sensor_config = sensor_config
        self._rig = rig
        self._cctv_detector = cctv_detector
        self._denoiser = denoiser
        self._onboard_detector = onboard_detector
        self._dqn = dqn
        self._lstm = lstm
        self._fusion = fusion
        self._seeds = itertools.count(config.seed)

    @classmethod
    def load(
        cls, loader: ConfigLoader, app_config: AppConfig, config: AutonomyConfig
    ) -> AutonomyStack:
        """Load every model named in ``config`` onto its device.

        Raises:
            ConfigurationError: If the app config names no sensor config.
            AssetNotFoundError: If any model file is missing — checked by
                each adapter before anything is driven.
        """
        from sentry_ai.common.exceptions import ConfigurationError  # noqa: PLC0415
        from sentry_ai.perception.autoencoder import ConvDenoisingAutoencoder  # noqa: PLC0415
        from sentry_ai.perception.yolo_detector import YoloDetector  # noqa: PLC0415

        if app_config.sensor_config_path is None:
            raise ConfigurationError("the full stack needs 'sensor_config' in the app config")
        device = _device(config.device)
        sensor_config = loader.load_sensor_config(app_config.sensor_config_path)
        rig = SensorRig.from_config(
            sensor_config, SensorPalette.from_config(loader, app_config.sensor_config_path)
        )
        models = config.models
        started = time.perf_counter()

        def detector(path: Any) -> YoloDetector:
            return YoloDetector(
                weights_path=path,
                confidence=config.confidence,
                image_size=config.image_size,
                device=device,
            )

        stack = cls(
            config=config,
            sensor_config=sensor_config,
            rig=rig,
            cctv_detector=detector(models.cctv_detector),
            denoiser=(
                None
                if models.denoiser is None
                else ConvDenoisingAutoencoder.from_checkpoint(models.denoiser, device=device)
            ),
            onboard_detector=detector(models.onboard_detector),
            dqn=DqnLocalController.from_file(models.dqn, device=device),
            lstm=LstmMotionPredictor.from_checkpoint(models.lstm, device=device),
            fusion=MlpFusion.from_checkpoint(models.fusion, device=device),
        )
        logger.info("Full stack loaded on %s in %.1fs", device, time.perf_counter() - started)
        return stack

    def mission(
        self,
        city_map: CityMap,
        simulation_config: SimulationConfig,
        vehicle_config: VehicleConfig,
        hazard_seed: int | None = None,
        wrap: Any = None,
    ) -> AutonomousMission:
        """Wire a fresh, timed, recorded mission over ``city_map``.

        Args:
            city_map: The world. A mission mutates it; use a fresh one each time.
            simulation_config: Rules, planner, hazards.
            vehicle_config: Physics.
            hazard_seed: Overrides the config's hazard seed.
            wrap: Optional ``controller -> controller`` applied last — how the
                live window adds its keyboard override.
        """
        profiler = Profiler()
        degradation = self._sensor_config.degradation
        cctv_seed, onboard_seed = next(self._seeds), next(self._seeds)

        grid_source: IOccupancyGridSource = DetectedGridSource(
            rig=self._rig,
            observer=ModelObserver(TimedDetector(self._cctv_detector, profiler, "cctv detect")),
            builder=OccupancyGridBuilder.from_city_map(city_map),
            degrader=FrameDegrader(degradation, np.random.default_rng(cctv_seed)),
            denoiser=(
                None
                if self._denoiser is None
                else TimedDenoiser(self._denoiser, profiler, "cctv denoise")
            ),
        )
        grid_source = ThrottledGridSource(grid_source, self.config.perception_every)
        if self.config.belief_lag:
            grid_source = LaggedGridSource(grid_source, self.config.belief_lag)
        grid_source = TimedGridSource(grid_source, profiler, "cctv map (total)")

        sensing = OnboardSensing(
            rig=self._rig,
            observer=ModelObserver(
                TimedDetector(self._onboard_detector, profiler, "onboard detect")
            ),
            degrader=FrameDegrader(degradation, np.random.default_rng(onboard_seed)),
        )
        controller = FusedLocalController(
            TimedController(self._dqn, profiler, "dqn"),
            TimedPredictor(self._lstm, profiler, "lstm"),
            TimedFusion(self._fusion, profiler, "fusion"),
            timed_call(sensing.sightings, profiler, "onboard (total)"),
        )
        recorder = PathRecorder(controller, city_map.vehicle.position)
        driver = wrap(recorder) if wrap is not None else recorder
        mission = build_mission(
            city_map=city_map,
            simulation_config=simulation_config,
            vehicle_config=vehicle_config,
            controller=driver,
            hazard_seed=hazard_seed,
            grid_source=grid_source,
        )
        sensing.attach(city_map)
        return AutonomousMission(
            mission=mission,
            controller=controller,
            driver=driver,
            recorder=recorder,
            profiler=profiler,
            seed=hazard_seed,
            settings={
                "models": {
                    name: None if path is None else path.name
                    for name, path in asdict(self.config.models).items()
                },
                "perception_every": self.config.perception_every,
                "belief_lag": self.config.belief_lag,
                "denoiser": self._denoiser is not None,
            },
        )


def _device(requested: str) -> str:
    """Resolve ``"auto"`` to CUDA when Torch sees a GPU, else CPU."""
    if requested != "auto":
        return requested

    import torch  # noqa: PLC0415 - already loaded by every model here

    return "cuda" if torch.cuda.is_available() else "cpu"
