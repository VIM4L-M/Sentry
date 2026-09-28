"""Turns the vehicle's own camera into egocentric :class:`SceneEvidence` (Phase 7).

The command center plans on a map assembled from the CCTV network, and that
map can lag the world — a collapse is on the street seconds before it is on
the map. The onboard camera sees what is in front of the vehicle *now*.
This module is how that view reaches the fusion network: every detection's
box is projected onto the tiles it covers (the same exact projection the
CCTV pipeline uses) and each tile is then named relative to the vehicle —
ahead, left, right, behind.

Two parts, kept apart so each is testable without the other:

* :func:`egocentric_evidence` — pure geometry: detections + camera view +
  vehicle pose in, :class:`SceneEvidence` out.
* :class:`OnboardEvidenceSource` — captures the onboard frame, optionally
  degrades and denoises it, runs a detector, and hands the result to
  :func:`egocentric_evidence`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind, Heading
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.decision import EVIDENCE_KINDS, Region, SceneEvidence
from sentry_ai.interfaces.perception import Detection, IDenoiser, IVisionDetector
from sentry_ai.sensors.camera import CameraView
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.rig import SensorRig

#: A tile coordinate that may lie off the map.
Coordinate = tuple[int, int]

#: Tiles within this many steps (grid distance) count as ``Region.NEARBY``.
NEARBY_RADIUS = 2


def egocentric_evidence(
    detections: Iterable[Detection],
    view: CameraView,
    position: Position,
    heading: Heading,
) -> SceneEvidence:
    """Name every detection by where it sits relative to the vehicle.

    A detection covering several tiles counts in every region it touches —
    a fire box spanning the tile ahead and the one beside it is both
    ``AHEAD`` and ``RIGHT``. Each slot keeps the highest confidence seen.
    Classes the fusion network has no slot for (the vehicle, smoke) are
    ignored.
    """
    regions = _region_tiles(position, heading)
    best: dict[tuple[EntityKind, Region], float] = {}
    for detection in detections:
        if detection.label not in EVIDENCE_KINDS:
            continue
        covered = {tile.as_tuple() for tile in _footprint(view, detection)}
        for region, tiles in regions.items():
            if covered & tiles:
                key = (detection.label, region)
                best[key] = max(best.get(key, 0.0), detection.confidence)
    return SceneEvidence(confidence=best)


@dataclass(frozen=True)
class Sighting:
    """One detection placed on the map: what, on which tile, how sure."""

    kind: EntityKind
    position: Position
    confidence: float


def sightings(detections: Iterable[Detection], view: CameraView) -> tuple[Sighting, ...]:
    """Every fusion-relevant detection, one :class:`Sighting` per tile it claims."""
    return tuple(
        Sighting(detection.label, tile, detection.confidence)
        for detection in detections
        if detection.label in EVIDENCE_KINDS
        for tile in _footprint(view, detection)
    )


def _footprint(view: CameraView, detection: Detection) -> frozenset[Position]:
    """The tiles a detection claims — by tile centre for fire, by any overlap otherwise.

    The same rule the CCTV grid uses (``OccupancyGridBuilder``). A fire box
    spills about half a tile past the fire; counting every tile it touches
    reported a fire beside the street as a fire *ahead*, and fusion closed a
    street that was open. A victim or debris marker sits well inside its
    tile, and a centre rule could lose it entirely. Falls back to overlap
    when a small fire box contains no tile centre at all.
    """
    if detection.label is EntityKind.FIRE:
        centred = view.tiles_centred_in_box(detection.bbox)
        if centred:
            return centred
    return view.tiles_of_box(detection.bbox)


def _region_tiles(position: Position, heading: Heading) -> dict[Region, frozenset[Coordinate]]:
    """The tile coordinates each egocentric region refers to, for this pose.

    Plain ``(x, y)`` tuples rather than :class:`Position`: beside the map
    edge a region's tile can be off the map, which ``Position`` refuses to
    represent, and such a tile simply never matches a detection.
    """
    fx, fy = heading.delta
    rx, ry = heading.turn_right().delta

    def offset(forward: int, right: int) -> Coordinate:
        return (position.x + forward * fx + right * rx, position.y + forward * fy + right * ry)

    nearby = frozenset(
        (position.x + dx, position.y + dy)
        for dx in range(-NEARBY_RADIUS, NEARBY_RADIUS + 1)
        for dy in range(-NEARBY_RADIUS, NEARBY_RADIUS + 1)
        if 0 < abs(dx) + abs(dy) <= NEARBY_RADIUS
    )
    return {
        Region.AHEAD: frozenset({offset(1, 0)}),
        Region.AHEAD_FAR: frozenset({offset(2, 0)}),
        Region.LEFT: frozenset({offset(0, -1)}),
        Region.RIGHT: frozenset({offset(0, 1)}),
        Region.BEHIND: frozenset({offset(-1, 0)}),
        Region.NEARBY: nearby,
    }


class OnboardEvidenceSource:
    """The onboard camera, run through the perception stack, as evidence.

    Every stage after capture is optional and injected, so the same class
    serves the live demo (degraded frames, denoiser, YOLO), fast data
    collection (ground-truth labels, no model), and ablations (camera
    switched off).
    """

    def __init__(
        self,
        rig: SensorRig,
        detector: IVisionDetector | None = None,
        degrader: FrameDegrader | None = None,
        denoiser: IDenoiser | None = None,
    ) -> None:
        """Wire the source.

        Args:
            rig: Supplies the onboard frame.
            detector: The model that reads the frame. ``None`` uses the
                frame's own ground-truth labels at full confidence — a
                perfect detector, for fast data collection and for
                isolating fusion from detector error.
            degrader: Smoke, blur and noise applied before detection.
            denoiser: Runs after the degrader, before the detector.
        """
        self._rig = rig
        self._detector = detector
        self._degrader = degrader
        self._denoiser = denoiser
        self.enabled = True
        self.use_denoiser = denoiser is not None
        #: What the last frame's detections were, placed on the map — for the
        #: drive view to highlight. Empty while the camera is switched off.
        self.last_sightings: tuple[Sighting, ...] = ()

    @property
    def has_denoiser(self) -> bool:
        """Whether a denoiser is installed; :attr:`use_denoiser` switches it."""
        return self._denoiser is not None

    def evidence(self, city_map: CityMap) -> SceneEvidence:
        """What the camera reports around the vehicle right now.

        Returns :meth:`SceneEvidence.empty` while :attr:`enabled` is
        ``False`` — the ablation switch for "the camera is blind".
        """
        if not self.enabled:
            self.last_sightings = ()
            return SceneEvidence.empty()
        frame = self._rig.capture_onboard(city_map)
        if self._detector is None:
            detections: Iterable[Detection] = frame.annotations
        else:
            if self._degrader is not None:
                frame = self._degrader.degrade(frame)
            pixels = frame.pixels
            if self._denoiser is not None and self.use_denoiser:
                pixels = self._denoiser.denoise(pixels)
            detections = self._detector.detect(pixels)
        detections = list(detections)
        self.last_sightings = sightings(detections, frame.view)
        vehicle = city_map.vehicle
        return egocentric_evidence(detections, frame.view, vehicle.position, vehicle.heading)
