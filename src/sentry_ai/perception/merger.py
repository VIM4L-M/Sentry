"""Turns per-camera detections into one world-space belief (Phase 3.2).

Four fixed CCTV cameras watch the city and their footprints deliberately
overlap, so the same victim is routinely seen twice. The command center
cannot act on that directly: two boxes in two frames must become one answer
to the question *what is on this tile?* before an occupancy grid can be
built from them.

This module answers exactly that and nothing more. It never touches pixels,
never loads a model, and never builds a grid — it takes detections that
somebody else produced and returns world-space objects. That keeps it
testable with hand-built detections and no Torch in sight, which is the
whole reason Phase 3 is split the way it is.

The merge rule is deliberately asymmetric between people and hazards; see
:class:`DetectionMerger` for why.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sentry_ai.common.logging_config import get_logger
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind
from sentry_ai.interfaces.perception import Detection
from sentry_ai.sensors.camera import CameraView

logger = get_logger(__name__)

#: Hazards occupy an area, so two sightings that merely touch are treated as
#: one thing. People do not: two victims on neighbouring tiles are two
#: people, and merging them would erase one. Fire is the only class whose
#: annotation spans many tiles and gets clipped differently by each camera.
_MERGE_ON_ADJACENCY = frozenset({EntityKind.FIRE})

#: Classes projected by tile *centre* rather than by any pixel overlap.
#: A detector's box is a pixel or two off, and for something spanning many
#: tiles that error claims a whole extra ring — which for fire is a ring of
#: impassable cells. Requiring the centre absorbs up to half a tile of error.
#: Victims are deliberately excluded: their marker fills a quarter of a tile,
#: so the strict rule could drop them entirely, and losing a victim is the
#: one error this system may not make. Same asymmetry as the merge rule.
_PROJECT_BY_CENTRE = frozenset({EntityKind.FIRE})


@dataclass(frozen=True)
class CameraObservation:
    """What one camera reported at one instant.

    Carries the :class:`~sentry_ai.sensors.camera.CameraView` rather than the
    whole :class:`~sentry_ai.sensors.frame.CameraFrame`, because merging needs
    the projection and not the pixels. A test can therefore build one without
    allocating an image.

    Attributes:
        view: Where the camera was pointed, supplying the projection.
        detections: What was found in that frame, in any order.
    """

    view: CameraView
    detections: tuple[Detection, ...] = ()

    @property
    def camera_id(self) -> str:
        """Which camera reported this."""
        return self.view.camera_id


@dataclass(frozen=True)
class WorldDetection:
    """One object, in map space, possibly seen by several cameras.

    Attributes:
        label: What it is.
        tiles: Every world tile it covers. A victim or a piece of debris
            occupies one; a fire occupies its whole footprint.
        confidence: How sure the detector was. For a merged sighting this is
            the most confident single view of it.
        camera_ids: Every camera that contributed, so downstream code can
            tell a corroborated detection from a lone one.
    """

    label: EntityKind
    tiles: frozenset[Position]
    confidence: float
    camera_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.tiles:
            raise ValueError("WorldDetection must cover at least one tile")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"WorldDetection.confidence must be within 0.0-1.0, got {self.confidence}"
            )

    @property
    def position(self) -> Position:
        """A single representative tile — the one nearest the footprint's centre.

        Single-tile detections return that tile. For a fire this is where the
        blaze is *reported*, which is what a mission log or a HUD marker
        wants; code that must mark every burning cell uses :attr:`tiles`.
        """
        if len(self.tiles) == 1:
            return next(iter(self.tiles))
        mean_x = sum(tile.x for tile in self.tiles) / len(self.tiles)
        mean_y = sum(tile.y for tile in self.tiles) / len(self.tiles)
        return min(
            sorted(self.tiles, key=lambda tile: (tile.y, tile.x)),
            key=lambda tile: (tile.x - mean_x) ** 2 + (tile.y - mean_y) ** 2,
        )

    @property
    def corroborated(self) -> bool:
        """Whether more than one camera saw this."""
        return len(self.camera_ids) > 1


class DetectionMerger:
    """Fuses per-camera detections into one world-space belief.

    Two sightings merge when they share a label *and* their world footprints
    meet. What "meet" means depends on what is being merged, and the
    asymmetry is the point:

    * **Victims and debris** merge only when their footprints actually
      overlap. Both are painted inside a single tile, so two cameras seeing
      one victim agree on that tile, while two victims on neighbouring tiles
      stay two victims. Merging on adjacency here would quietly erase a
      person — the worst error this system can make.
    * **Fire** also merges when footprints are edge-adjacent. A fire is
      annotated across its whole visible disc, and a blaze straddling a
      camera seam is clipped into two disjoint halves that touch but do not
      overlap. Reporting those as two fires would be wrong, and the cost of
      over-merging is nil: adjacent burning tiles are one impassable region
      to the planner either way.

    The merger is deliberately not configurable on this point. It is a
    structural consequence of what the classes mean, not a threshold to tune.
    """

    def merge(self, observations: Sequence[CameraObservation]) -> list[WorldDetection]:
        """Fuse every camera's detections into distinct world-space objects.

        Args:
            observations: One entry per camera reporting this instant. Order
                does not affect the result.

        Returns:
            Distinct objects, sorted by label then position, so the output is
            reproducible and a test can assert on it directly.
        """
        projected = [
            candidate
            for observation in observations
            for candidate in self._project(observation)
        ]
        merged = [self._combine(group) for group in self._group(projected)]
        merged.sort(key=lambda found: (found.label.value, found.position.y, found.position.x))

        if len(merged) != len(projected):
            logger.debug(
                "merged %d detection(s) from %d camera(s) into %d object(s)",
                len(projected),
                len(observations),
                len(merged),
            )
        return merged

    @classmethod
    def _project(cls, observation: CameraObservation) -> list[WorldDetection]:
        """Every detection in one frame, converted to world tiles."""
        return [
            WorldDetection(
                label=detection.label,
                tiles=cls._tiles_for(observation, detection),
                confidence=detection.confidence,
                camera_ids=frozenset({observation.camera_id}),
            )
            for detection in observation.detections
        ]

    @staticmethod
    def _tiles_for(observation: CameraObservation, detection: Detection) -> frozenset[Position]:
        """The world footprint of one box, strictly or generously by label.

        See :data:`_PROJECT_BY_CENTRE` for why fire is measured differently
        from the people this system exists to find. A detection that lands
        on no tile at all under the strict rule falls back to the generous
        one — a hazard the detector reported must never disappear because it
        was reported half a tile off.
        """
        if detection.label not in _PROJECT_BY_CENTRE:
            return observation.view.tiles_of_box(detection.bbox)
        centred = observation.view.tiles_centred_in_box(detection.bbox)
        return centred if centred else observation.view.tiles_of_box(detection.bbox)

    def _group(self, candidates: Sequence[WorldDetection]) -> list[list[WorldDetection]]:
        """Partition candidates into groups that describe the same object.

        Grouping is transitive on purpose: if A meets B and B meets C, all
        three are one object even when A and C do not touch. That is what a
        fire spanning three camera footprints looks like.
        """
        groups: list[list[WorldDetection]] = []
        for candidate in candidates:
            matches = [group for group in groups if self._belongs(candidate, group)]
            if not matches:
                groups.append([candidate])
                continue
            # Absorb every group this one bridges, keeping the relation transitive.
            merged_group = [candidate]
            for group in matches:
                merged_group.extend(group)
                groups.remove(group)
            groups.append(merged_group)
        return groups

    def _belongs(self, candidate: WorldDetection, group: Sequence[WorldDetection]) -> bool:
        """Whether ``candidate`` describes the same object as anything in ``group``."""
        return any(
            member.label is candidate.label and self._meets(candidate, member)
            for member in group
        )

    @staticmethod
    def _meets(left: WorldDetection, right: WorldDetection) -> bool:
        """Whether two same-label footprints describe one object.

        Overlap always counts. Edge adjacency counts only for the classes in
        :data:`_MERGE_ON_ADJACENCY` — see :class:`DetectionMerger` for why
        people and hazards are treated differently.
        """
        if not left.tiles.isdisjoint(right.tiles):
            return True
        if left.label not in _MERGE_ON_ADJACENCY:
            return False
        # Plain tuples, not Positions: a tile on the map edge has neighbours
        # at -1, and Position rejects negative coordinates by design.
        neighbourhood = {
            (tile.x + dx, tile.y + dy)
            for tile in left.tiles
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        }
        return any((tile.x, tile.y) in neighbourhood for tile in right.tiles)

    def _combine(self, group: Sequence[WorldDetection]) -> WorldDetection:
        """Collapse one group into a single object."""
        return WorldDetection(
            label=group[0].label,
            tiles=frozenset().union(*(member.tiles for member in group)),
            # The most confident view of a thing, not an average: a camera
            # with a clear line of sight should not be dragged down by one
            # looking through smoke.
            confidence=max(member.confidence for member in group),
            camera_ids=frozenset().union(*(member.camera_ids for member in group)),
        )
