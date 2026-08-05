"""The complete sensor network watching the disaster city.

Agent 1 (the command center) sees the city through fixed CCTV cameras;
Agent 2 (the vehicle) sees it through one camera that travels with it.
:class:`SensorRig` is both, captured together, because the command center's
job is to *merge* those views into one belief about the world.

The rig deliberately does not degrade what it captures. Corruption belongs
to :class:`~sentry_ai.sensors.degradation.FrameDegrader`, kept separate so a
caller can hold the clean frame and the corrupted one at the same time —
which is precisely what training a denoiser requires.
"""

from __future__ import annotations

from sentry_ai.common.exceptions import ConfigValidationError
from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.schema import SensorConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.map import CityMap
from sentry_ai.sensors.camera import CameraView, OnboardCamera
from sentry_ai.sensors.frame import CameraFrame
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rasterizer import FrameRasterizer

logger = get_logger(__name__)


class SensorRig:
    """Every camera watching the city: fixed CCTV plus the vehicle's own."""

    def __init__(
        self,
        cctv_views: tuple[CameraView, ...],
        onboard: OnboardCamera,
        rasterizer: FrameRasterizer,
    ) -> None:
        """Wire the rig.

        Args:
            cctv_views: Fixed camera footprints, in a stable order — the
                order frames come back in, and the order the command center
                merges them in.
            onboard: The vehicle-mounted camera.
            rasterizer: Turns a view of the city into a labelled frame.

        Raises:
            ConfigValidationError: If two cameras share an id, which would
                make a frame's provenance ambiguous.
        """
        ids = [view.camera_id for view in cctv_views] + [onboard.camera_id]
        duplicates = {name for name in ids if ids.count(name) > 1}
        if duplicates:
            raise ConfigValidationError(
                f"camera ids must be unique, repeated: {', '.join(sorted(duplicates))}"
            )
        self._cctv_views = cctv_views
        self._onboard = onboard
        self._rasterizer = rasterizer

    @classmethod
    def from_config(cls, config: SensorConfig, palette: SensorPalette) -> SensorRig:
        """Build a rig from a loaded sensors config."""
        views = tuple(
            CameraView(
                camera_id=camera.camera_id,
                origin=Position(camera.origin_x, camera.origin_y),
                width_tiles=camera.width_tiles,
                height_tiles=camera.height_tiles,
                tile_size_px=config.tile_size_px,
            )
            for camera in config.cameras
        )
        onboard = OnboardCamera(
            camera_id=config.onboard.camera_id,
            span_tiles=config.onboard.span_tiles,
            tile_size_px=config.onboard.tile_size_px,
        )
        return cls(cctv_views=views, onboard=onboard, rasterizer=FrameRasterizer(palette))

    @property
    def cctv_views(self) -> tuple[CameraView, ...]:
        """The fixed camera footprints, in capture order."""
        return self._cctv_views

    def capture_cctv(self, city_map: CityMap) -> list[CameraFrame]:
        """One frame from every fixed camera — the command center's input."""
        return [self._rasterizer.render(city_map, view) for view in self._cctv_views]

    def capture_onboard(self, city_map: CityMap) -> CameraFrame:
        """One frame from the vehicle's camera, framed on where it is now."""
        view = self._onboard.view_for(city_map.vehicle, city_map.width, city_map.height)
        return self._rasterizer.render(city_map, view)

    def capture_all(self, city_map: CityMap) -> list[CameraFrame]:
        """Every camera at once, CCTV first and the onboard view last."""
        return [*self.capture_cctv(city_map), self.capture_onboard(city_map)]

    def coverage(self, city_map: CityMap) -> float:
        """Fraction of the city at least one fixed camera can see.

        Reported because a blind spot in the CCTV network is invisible in
        the occupancy grid — uncovered tiles simply keep whatever the grid
        last believed, and that is worth knowing before trusting a map.
        """
        seen: set[Position] = set()
        for view in self._cctv_views:
            seen.update(tile for tile in view.world_tiles() if city_map.in_bounds(tile))
        return len(seen) / float(city_map.width * city_map.height)
