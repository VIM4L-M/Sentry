"""Per-stage timing of the full autonomy stack (Phase 8).

Every expensive stage sits behind a port, so each can be wrapped in a timed
decorator that implements the same port: the engine and the controllers
cannot tell they are being measured. The result is a table of where a
tick's time goes — which is what "profile against the target hardware"
needs, and what decides which knob to turn when a machine is too slow.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sentry_ai.domain.entities import Position
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.interfaces.decision import FinalAction, IDecisionFusion
from sentry_ai.interfaces.navigation import ILocalController, LocalDecision, LocalObservation
from sentry_ai.interfaces.perception import Detection, IDenoiser, IVisionDetector, WorldDetection
from sentry_ai.interfaces.sequence import BehaviourSignal, IMotionPredictor, VehicleState
from sentry_ai.interfaces.world import IOccupancyGridSource


@dataclass(frozen=True)
class StageTiming:
    """One stage's accumulated cost."""

    stage: str
    calls: int
    total_seconds: float

    @property
    def mean_ms(self) -> float:
        """Average milliseconds per call."""
        return 1000.0 * self.total_seconds / self.calls if self.calls else 0.0


@dataclass
class Profiler:
    """Accumulates wall-clock time per named stage."""

    _totals: dict[str, float] = field(default_factory=dict)
    _calls: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Time the enclosed block and charge it to ``stage``."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self._totals[stage] = self._totals.get(stage, 0.0) + time.perf_counter() - started
            self._calls[stage] = self._calls.get(stage, 0) + 1

    def timings(self) -> list[StageTiming]:
        """Every stage measured so far, in first-seen order."""
        return [StageTiming(name, self._calls[name], total) for name, total in self._totals.items()]

    def per_tick_ms(self, stage: str, ticks: int) -> float:
        """A stage's total cost spread over ``ticks`` — what it adds to an average tick."""
        return 1000.0 * self._totals.get(stage, 0.0) / ticks if ticks else 0.0


class TimedDetector(IVisionDetector):
    """An :class:`IVisionDetector` that charges its time to ``stage``."""

    def __init__(self, inner: IVisionDetector, profiler: Profiler, stage: str) -> None:
        self._inner, self._profiler, self._stage = inner, profiler, stage

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        with self._profiler.measure(self._stage):
            return self._inner.detect(frame)

    def detect_many(self, frames: Sequence[NDArray[np.uint8]]) -> list[list[Detection]]:
        with self._profiler.measure(self._stage):
            return self._inner.detect_many(frames)


class TimedDenoiser(IDenoiser):
    """An :class:`IDenoiser` that charges its time to ``stage``."""

    def __init__(self, inner: IDenoiser, profiler: Profiler, stage: str) -> None:
        self._inner, self._profiler, self._stage = inner, profiler, stage

    def denoise(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        with self._profiler.measure(self._stage):
            return self._inner.denoise(frame)


class TimedGridSource(IOccupancyGridSource):
    """An :class:`IOccupancyGridSource` that charges its time to ``stage``."""

    def __init__(self, inner: IOccupancyGridSource, profiler: Profiler, stage: str) -> None:
        self._inner, self._profiler, self._stage = inner, profiler, stage

    def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
        with self._profiler.measure(self._stage):
            return self._inner.grid_for(city_map, vehicle_position)


class TimedController(ILocalController):
    """An :class:`ILocalController` that charges its time to ``stage``."""

    def __init__(self, inner: ILocalController, profiler: Profiler, stage: str) -> None:
        self._inner, self._profiler, self._stage = inner, profiler, stage

    def decide(self, observation: LocalObservation) -> LocalDecision:
        with self._profiler.measure(self._stage):
            return self._inner.decide(observation)


class TimedPredictor(IMotionPredictor):
    """An :class:`IMotionPredictor` that charges its time to ``stage``."""

    def __init__(self, inner: IMotionPredictor, profiler: Profiler, stage: str) -> None:
        self._inner, self._profiler, self._stage = inner, profiler, stage

    def predict(self, state_history: Sequence[VehicleState]) -> BehaviourSignal:
        with self._profiler.measure(self._stage):
            return self._inner.predict(state_history)


class TimedFusion(IDecisionFusion):
    """An :class:`IDecisionFusion` that charges its time to ``stage``."""

    def __init__(self, inner: IDecisionFusion, profiler: Profiler, stage: str) -> None:
        self._inner, self._profiler, self._stage = inner, profiler, stage

    def fuse(
        self,
        observation: LocalObservation,
        sightings: Sequence[WorldDetection],
        behaviour: BehaviourSignal,
        local_decision: LocalDecision,
    ) -> FinalAction:
        with self._profiler.measure(self._stage):
            return self._inner.fuse(observation, sightings, behaviour, local_decision)


def timed_call(
    call: Callable[[], Sequence[WorldDetection]], profiler: Profiler, stage: str
) -> Callable[[], Sequence[WorldDetection]]:
    """Wrap a zero-argument sensing callable so its time is charged to ``stage``."""

    def run() -> Sequence[WorldDetection]:
        with profiler.measure(stage):
            return call()

    return run
