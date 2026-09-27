"""The camera pipeline as a grid producer (Phase 3.4).

Assembles the Phase 3 steps — and, optionally, Phase 4's denoiser — into
one thing the mission loop can hold::

    SensorRig -> FrameDegrader -> [IDenoiser] -> IVisionDetector
              -> DetectionMerger -> OccupancyGridBuilder

and exposes it as an
:class:`~sentry_ai.interfaces.world.IOccupancyGridSource`, so that swapping
perception in for ground truth is an injection rather than an edit.

Nothing here decides anything. Every judgement — what merges, what outranks
what, what a box projects to — was made in 3.2 and 3.3 and is only being
sequenced. That is deliberate: a composition module that also contains logic
is a module you cannot reason about by reading its parts.

**Torch stays off the import path.** The detector arrives as an
``IVisionDetector``, so importing this module costs nothing and the
ground-truth observer below runs the whole chain with no model at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from sentry_ai.common.logging_config import get_logger
from sentry_ai.domain.entities import Position
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.interfaces.perception import IDenoiser, IVisionDetector
from sentry_ai.interfaces.world import IOccupancyGridSource
from sentry_ai.perception.grid_builder import OccupancyGridBuilder
from sentry_ai.perception.merger import CameraObservation, DetectionMerger
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.frame import CameraFrame
from sentry_ai.sensors.rig import SensorRig

logger = get_logger(__name__)


class IFrameObserver(ABC):
    """Turns captured frames into per-camera detections.

    Exists so "run the model" and "use the answer key" are the same shape.
    Substituting a perfect detector is how every downstream defect gets
    attributed: if a mission fails with :class:`GroundTruthObserver`, the
    fault is in the merge, the grid, or the planner — never in the weights.
    """

    @abstractmethod
    def observe(self, frames: Sequence[CameraFrame]) -> list[CameraObservation]:
        """Report what is in each frame, one observation per camera."""
        raise NotImplementedError


class ModelObserver(IFrameObserver):
    """Runs a trained detector over every frame."""

    def __init__(self, detector: IVisionDetector) -> None:
        """Wrap ``detector``, which sees pixels and nothing else."""
        self._detector = detector

    def observe(self, frames: Sequence[CameraFrame]) -> list[CameraObservation]:
        """Detect in each frame, keeping it paired with the view that made it."""
        return [
            CameraObservation(frame.view, tuple(self._detector.detect(frame.pixels)))
            for frame in frames
        ]


class GroundTruthObserver(IFrameObserver):
    """Substitutes each frame's own annotations for a model's predictions.

    A perfect detector, by construction. Not a stub for a missing feature —
    it is the control condition, and the reason a mission that fails on
    perception can be diagnosed at all.
    """

    def observe(self, frames: Sequence[CameraFrame]) -> list[CameraObservation]:
        """Report the ground truth the rasterizer painted into each frame."""
        return [CameraObservation(frame.view, frame.annotations) for frame in frames]


class DetectedGridSource(IOccupancyGridSource):
    """Builds the command center's belief map from what the cameras saw.

    The Phase 3 counterpart to
    :class:`~sentry_ai.simulation.grid_source.GroundTruthGridSource`, and
    interchangeable with it.
    """

    def __init__(
        self,
        rig: SensorRig,
        observer: IFrameObserver,
        builder: OccupancyGridBuilder,
        degrader: FrameDegrader | None = None,
        merger: DetectionMerger | None = None,
        denoiser: IDenoiser | None = None,
    ) -> None:
        """Compose the pipeline.

        Args:
            rig: The camera network. Only the CCTV views are used — the
                onboard camera sees what the vehicle is already standing in,
                so it adds nothing the fixed cameras have not covered.
            observer: How frames become detections.
            builder: Holds the surveyed terrain and applies precedence.
            degrader: Applied before observation, so the detector sees the
                same smoke and noise it was trained on. ``None`` feeds clean
                frames, which is the right choice only for a control run.
            merger: Defaults to a plain :class:`DetectionMerger`; injectable
                because its rules are the sort of thing an experiment varies.
            denoiser: Phase 4. Cleans each frame after the degrader and
                before the observer — the one place it can go, since it
                exists to undo what the degrader did. ``None`` hands the
                observer the corrupted frames directly, the Phase 3
                behaviour.
        """
        self._rig = rig
        self._observer = observer
        self._builder = builder
        self._degrader = degrader
        self._merger = merger if merger is not None else DetectionMerger()
        self._denoiser = denoiser

    def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
        """Look at the city through the cameras and report what is believed."""
        frames = self._rig.capture_cctv(city_map)
        if self._degrader is not None:
            frames = [self._degrader.degrade(frame) for frame in frames]
        if self._denoiser is not None:
            frames = [frame.with_pixels(self._denoiser.denoise(frame.pixels)) for frame in frames]
        merged = self._merger.merge(self._observer.observe(frames))
        logger.debug("built a belief grid from %d merged detection(s)", len(merged))
        return self._builder.build(merged, vehicle_position=vehicle_position)
