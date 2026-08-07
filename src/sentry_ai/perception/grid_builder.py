"""Turns merged world-space detections into an occupancy grid (Phase 3.3).

This is where perception stops being a list of sightings and becomes the
thing A\\* actually plans over. It is the last step of the Phase 3 chain::

    SensorRig -> IVisionDetector -> DetectionMerger -> OccupancyGridBuilder

and it replaces :meth:`~sentry_ai.domain.occupancy.OccupancyGrid.from_city_map`
as the *producer* of that grid. The ground-truth method stays in the codebase
precisely so this one has something to be scored against — see
:mod:`sentry_ai.perception.grid_metrics`.

**Two sources, one grid.** The static street plan comes from the surveyed
map; only the dynamic codes — ``FIRE``, ``DEBRIS``, ``VICTIM`` — come from
the cameras. That split is a deliberate architectural decision recorded in
PROJECT.md: the simulator generated the layout, so training a detector to
rediscover roads and buildings would cost parameters and add label noise
while teaching the system nothing it does not already know.

The consequence worth being honest about: a building that collapses into a
street mid-mission is *not* refreshed from the map. It has to be detected,
because it is exactly the kind of change a real command center would only
learn about by looking. So the grid this module produces is a belief, and
:mod:`sentry_ai.perception.grid_metrics` measures how wrong it is.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sentry_ai.common.logging_config import get_logger
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind
from sentry_ai.domain.occupancy import CityMapLike, OccupancyCode, OccupancyGrid
from sentry_ai.perception.merger import WorldDetection

logger = get_logger(__name__)

#: Which occupancy code each detector class writes. The three detector
#: classes map onto the three *dynamic* codes exactly; there is no detector
#: class for ``ROAD``, ``BUILDING``, ``HOSPITAL`` or ``VEHICLE`` because
#: none of those is something a camera has to discover.
DETECTION_TO_OCCUPANCY: dict[EntityKind, OccupancyCode] = {
    EntityKind.OBSTACLE: OccupancyCode.DEBRIS,
    EntityKind.FIRE: OccupancyCode.FIRE,
    EntityKind.VICTIM: OccupancyCode.VICTIM,
}

#: The order detections are stamped in, lowest precedence first. This
#: mirrors ``OccupancyGrid.from_city_map`` exactly — terrain, then fire,
#: then victims — so a detector-derived grid and the ground-truth grid
#: resolve a contested tile the same way and remain comparable. A victim
#: pinned in burning rubble reads as ``VICTIM`` in both, which is what lets
#: the planner route *to* them instead of around them.
_STAMP_ORDER: tuple[EntityKind, ...] = (
    EntityKind.OBSTACLE,
    EntityKind.FIRE,
    EntityKind.VICTIM,
)


@dataclass(frozen=True)
class OccupancyGridBuilder:
    """Builds the command center's belief map from what the cameras saw.

    Holds the surveyed terrain and stamps detections onto a copy of it, so
    one builder serves a whole mission and no call can corrupt the next.

    Attributes:
        terrain: The static layout, from
            :meth:`~sentry_ai.domain.occupancy.OccupancyGrid.terrain_only`.
            Never mutated — every :meth:`build` starts from a fresh copy.
        min_confidence: Detections below this are discarded. Defaults to
            ``0.0``, which trusts whatever the detector reported; choosing a
            threshold is the composition root's policy, not this class's.
    """

    terrain: OccupancyGrid
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be within 0.0-1.0, got {self.min_confidence}"
            )

    @classmethod
    def from_city_map(
        cls, city_map: CityMapLike, min_confidence: float = 0.0
    ) -> OccupancyGridBuilder:
        """Survey ``city_map``'s static layout and build against it.

        Reads terrain and nothing else — no fire, no victims, no vehicle.
        Those are what the cameras are for.
        """
        return cls(terrain=OccupancyGrid.terrain_only(city_map), min_confidence=min_confidence)

    @property
    def width(self) -> int:
        """Number of tile columns in the grids this builder produces."""
        return self.terrain.width

    @property
    def height(self) -> int:
        """Number of tile rows in the grids this builder produces."""
        return self.terrain.height

    def build(
        self,
        detections: Sequence[WorldDetection],
        vehicle_position: Position | None = None,
    ) -> OccupancyGrid:
        """Stamp ``detections`` onto the surveyed terrain and return the belief.

        Args:
            detections: Merged, world-space sightings — the output of
                :class:`~sentry_ai.perception.merger.DetectionMerger`. Order
                does not matter; precedence comes from :data:`_STAMP_ORDER`.
            vehicle_position: Where the vehicle is. Supplied by the caller
                rather than detected, because the command center knows where
                its own vehicle is and no detector class reports one.

        Returns:
            A fresh grid. The builder's terrain is left untouched.
        """
        grid = OccupancyGrid(self.terrain.as_array())
        kept = [found for found in detections if found.confidence >= self.min_confidence]

        for kind in _STAMP_ORDER:
            code = DETECTION_TO_OCCUPANCY[kind]
            for found in (one for one in kept if one.label is kind):
                self._stamp(grid, found.tiles, code)

        self._restore_hospital(grid)
        if vehicle_position is not None and grid.in_bounds(vehicle_position):
            grid.mark_vehicle(vehicle_position)

        self._log_discards(detections, kept)
        return grid

    @staticmethod
    def _stamp(grid: OccupancyGrid, tiles: Iterable[Position], code: OccupancyCode) -> None:
        """Write ``code`` across ``tiles``, skipping anything off the map.

        Out-of-bounds tiles are dropped rather than raised on. A belief map
        is built from a fallible detector, and one stray box must not be
        able to end a mission.
        """
        for tile in tiles:
            if grid.in_bounds(tile):
                grid.mark(tile, code)

    def _restore_hospital(self, grid: OccupancyGrid) -> None:
        """Put the hospital back wherever a detection covered it.

        ``from_city_map`` stamps the safe zone *after* victims, so the drop-off
        point outranks anything standing on it. Matching that here keeps the
        two grids comparable, and a hospital that flickers into ``DEBRIS``
        because debris was detected on its apron would strand the mission
        with nowhere to deliver anyone.
        """
        grid.cells[self.terrain.cells == OccupancyCode.HOSPITAL] = OccupancyCode.HOSPITAL

    def _log_discards(
        self, detections: Sequence[WorldDetection], kept: Sequence[WorldDetection]
    ) -> None:
        if len(kept) != len(detections):
            logger.debug(
                "dropped %d detection(s) below confidence %.2f",
                len(detections) - len(kept),
                self.min_confidence,
            )
