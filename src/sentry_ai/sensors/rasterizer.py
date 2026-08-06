"""Paints a slice of the disaster city into an RGB array with labels.

This is the synthetic imagery the whole perception stack is trained and
evaluated on, so two properties matter more than realism:

* **The labels come from the same pass as the pixels.** Every victim, fire,
  and piece of debris is annotated as it is drawn, from the geometry that
  drew it. Ground truth cannot drift out of sync with the image.
* **A static scene renders identically every time.** Each camera owns a
  fixed texture field derived from its id, so the grain in a frame is a
  property of the sensor rather than of the moment. Two captures of an
  unchanged city are byte-identical, which makes "did the world change?"
  a question about the world and not about the renderer.

Texture is not decoration. Flat blocks of uniform color make a detection
problem a detector can solve by reading a single pixel; the per-pixel grain
here is what keeps Phase 3 an honest exercise.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from typing import Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from sentry_ai.common.color import Color
from sentry_ai.config.schema import MarkerScaleConfig
from sentry_ai.domain.entities import FireSource, Position
from sentry_ai.domain.enums import EntityKind, VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import tiles_within
from sentry_ai.interfaces.perception import Detection
from sentry_ai.sensors.camera import CameraView
from sentry_ai.sensors.frame import CameraFrame, clipped_box
from sentry_ai.sensors.palette import SensorPalette


class _Placed(Protocol):
    """Anything with a tile position — the only thing visibility needs."""

    position: Position


_PlacedT = TypeVar("_PlacedT", bound=_Placed)


class FrameRasterizer:
    """Renders a :class:`CameraView` of a city into a labelled frame."""

    def __init__(
        self, palette: SensorPalette, markers: MarkerScaleConfig | None = None
    ) -> None:
        """Create a rasterizer.

        Args:
            palette: Colors and texture strength to paint with.
            markers: How much of a tile each entity's marker fills. Defaults
                to :class:`MarkerScaleConfig`'s own defaults, so existing
                callers that only pass a palette are unaffected.
        """
        self._palette = palette
        self._markers = markers if markers is not None else MarkerScaleConfig()
        self._textures: dict[str, NDArray[np.int16]] = {}

    def render(self, city_map: CityMap, view: CameraView) -> CameraFrame:
        """Capture ``view`` of ``city_map`` as pixels plus ground truth.

        Painting order is background, fire, debris, victims, vehicle — so a
        victim trapped in rubble stays visible, which is exactly the case
        the detector most needs to get right.
        """
        canvas = self._paint_terrain(city_map, view)
        annotations: list[Detection] = [
            *self._paint_fires(canvas, city_map, view),
            *self._paint_obstacles(canvas, city_map, view),
            *self._paint_victims(canvas, city_map, view),
        ]
        self._paint_vehicle(canvas, city_map, view)

        pixels = np.clip(canvas + self._texture_for(view), 0, 255).astype(np.uint8)
        return CameraFrame(view=view, pixels=pixels, annotations=tuple(annotations))

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------

    def _paint_terrain(self, city_map: CityMap, view: CameraView) -> NDArray[np.int16]:
        """Fill every tile with its terrain color, or void if it is off-map."""
        canvas = np.zeros((view.frame_height, view.frame_width, 3), dtype=np.int16)
        for tile in view.world_tiles():
            color = (
                self._palette.terrain[city_map.tile_at(tile)]
                if city_map.in_bounds(tile)
                else self._palette.void
            )
            _fill_tile(canvas, view, tile, color)
        return canvas

    def _texture_for(self, view: CameraView) -> NDArray[np.int16]:
        """This camera's fixed sensor grain, generated once and reused.

        Seeded with CRC32 of the camera id rather than ``hash()``, which
        Python salts per process — a salted seed would make every run
        produce different imagery and quietly destroy reproducibility.
        """
        cached = self._textures.get(view.camera_id)
        if cached is not None and cached.shape[:2] == (view.frame_height, view.frame_width):
            return cached

        jitter = self._palette.texture_jitter
        rng = np.random.default_rng(zlib.crc32(view.camera_id.encode("utf-8")))
        speckle = rng.integers(
            -jitter, jitter + 1, size=(view.frame_height, view.frame_width, 1), dtype=np.int16
        )
        grain: NDArray[np.int16] = np.repeat(speckle, 3, axis=2)
        self._textures[view.camera_id] = grain
        return grain

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def _paint_fires(
        self, canvas: NDArray[np.int16], city_map: CityMap, view: CameraView
    ) -> list[Detection]:
        """Paint every fire's footprint and annotate the ones in shot."""
        detections: list[Detection] = []
        for fire in city_map.fires:
            visible = self._paint_one_fire(canvas, city_map, view, fire)
            if not visible:
                continue
            rects = [view.tile_rect(tile) for tile in visible]
            bounds = (
                min(rect[0] for rect in rects),
                min(rect[1] for rect in rects),
                max(rect[2] for rect in rects),
                max(rect[3] for rect in rects),
            )
            _append_detection(detections, EntityKind.FIRE, bounds, view)
        return detections

    def _paint_one_fire(
        self,
        canvas: NDArray[np.int16],
        city_map: CityMap,
        view: CameraView,
        fire: FireSource,
    ) -> list[Position]:
        """Paint one fire, returning the tiles of it this camera can see.

        Brightness falls off from the seat of the fire and scales with
        intensity, so a fire about to burn out looks different from one at
        its peak — a distinction the detector should be able to make.
        """
        footprint = tiles_within(fire.position, fire.radius, city_map.width, city_map.height)
        visible: list[Position] = []
        for tile in footprint:
            if not view.covers(tile):
                continue
            closeness = 1.0 - fire.position.distance_to(tile) / (fire.radius + 1)
            heat = self._palette.fire_edge.blended_with(
                self._palette.fire_core, closeness * fire.intensity
            )
            _fill_tile(canvas, view, tile, heat)
            visible.append(tile)
        return visible

    def _paint_obstacles(
        self, canvas: NDArray[np.int16], city_map: CityMap, view: CameraView
    ) -> list[Detection]:
        """Paint every discrete obstacle in shot, annotating those not holding a victim.

        Debris under a trapped victim is still *painted* — the rubble is
        physically there, and the victim's smaller marker drawn over it
        leaves a brown ring around a pink core, which is precisely what
        "trapped in rubble" looks like from above.

        It is not *annotated*, because one tile yields one answer and a
        victim outranks the debris pinning them. This is the same precedence
        :class:`~sentry_ai.domain.occupancy.OccupancyGrid` already applies
        when it builds a grid from the true world state: a victim standing
        in debris reads as ``VICTIM`` so the planner can route *to* them.

        Annotating both put the detector in an unwinnable position — a 13 px
        obstacle box containing an 8 px victim box, at a class balance of
        twelve obstacles to every victim — and it learned to answer
        ``OBSTACLE`` every time. That answer is worse than a miss: obstacle
        is an impassable code, so a trapped victim became a wall the planner
        routed around. Measured victim recall was exactly the share of
        victims *not* pinned in rubble.
        """
        occupied = self._trapped_victim_tiles(city_map)
        detections: list[Detection] = []
        for obstacle in _visible(city_map.obstacles, view):
            bounds = _fill_marker(
                canvas, view, obstacle.position, self._palette.debris, self._markers.debris
            )
            if obstacle.position not in occupied:
                _append_detection(detections, EntityKind.OBSTACLE, bounds, view)
        return detections

    @staticmethod
    def _trapped_victim_tiles(city_map: CityMap) -> set[Position]:
        """Tiles holding a victim still awaiting rescue."""
        return {
            victim.position
            for victim in city_map.victims
            if victim.status is VictimStatus.TRAPPED
        }

    def _paint_victims(
        self, canvas: NDArray[np.int16], city_map: CityMap, view: CameraView
    ) -> list[Detection]:
        """Paint and annotate every still-trapped victim in shot.

        Victims aboard the vehicle or already delivered are skipped: they
        are no longer at those coordinates, and labelling them there would
        train the detector to hallucinate.
        """
        trapped = [victim for victim in city_map.victims if victim.status is VictimStatus.TRAPPED]
        detections: list[Detection] = []
        for victim in _visible(trapped, view):
            bounds = _fill_marker(
                canvas, view, victim.position, self._palette.victim, self._markers.victim
            )
            _append_detection(detections, EntityKind.VICTIM, bounds, view)
        return detections

    def _paint_vehicle(
        self, canvas: NDArray[np.int16], city_map: CityMap, view: CameraView
    ) -> None:
        """Paint the rescue vehicle.

        Deliberately not annotated: it is the ego agent, not something the
        detector is asked to find. It is still drawn, because a model that
        never saw it would treat it as an anomaly at inference time.
        """
        if view.covers(city_map.vehicle.position):
            _fill_marker(
                canvas,
                view,
                city_map.vehicle.position,
                self._palette.vehicle,
                self._markers.vehicle,
            )


# ----------------------------------------------------------------------
# Painting primitives
# ----------------------------------------------------------------------


def _visible(entities: Sequence[_PlacedT], view: CameraView) -> list[_PlacedT]:
    """Those entities whose tile this camera can see."""
    return [entity for entity in entities if view.covers(entity.position)]


def _fill_tile(
    canvas: NDArray[np.int16], view: CameraView, tile: Position, color: Color
) -> None:
    """Fill a whole tile with a flat color."""
    x_min, y_min, x_max, y_max = view.tile_rect(tile)
    canvas[y_min:y_max, x_min:x_max] = color.as_tuple()


def _fill_marker(
    canvas: NDArray[np.int16],
    view: CameraView,
    tile: Position,
    color: Color,
    scale: float,
) -> tuple[int, int, int, int]:
    """Fill a centred sub-rectangle of a tile, returning its pixel bounds."""
    x_min, y_min, x_max, y_max = view.tile_rect(tile)
    inset = max(1, round(view.tile_size_px * (1.0 - scale) / 2))
    left, top = x_min + inset, y_min + inset
    right, bottom = max(left + 1, x_max - inset), max(top + 1, y_max - inset)
    canvas[top:bottom, left:right] = color.as_tuple()
    return (left, top, right, bottom)


def _append_detection(
    detections: list[Detection],
    label: EntityKind,
    bounds: tuple[int, int, int, int],
    view: CameraView,
) -> None:
    """Record a ground-truth detection, skipping anything fully off-frame."""
    box = clipped_box(*bounds, view=view)
    if box is not None:
        detections.append(Detection(label=label, confidence=1.0, bbox=box))
