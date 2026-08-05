"""Where a camera is pointed, and how its pixels map back to the city.

The projection here is the specification's "convert detections to world
coordinates" step. A detector returns a box in *frame* pixels; the command
center needs a tile on the map. :meth:`CameraView.to_world` is the only
place that conversion happens, so a camera's footprint and the meaning of
its output can never drift apart.

Cameras are orthographic and axis-aligned: the city is a tile grid viewed
from directly overhead. That keeps the projection exact and invertible,
which matters more here than photographic realism — a wrong projection
would poison every downstream phase with mislocated victims.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentry_ai.common.exceptions import ConfigValidationError
from sentry_ai.domain.entities import Position, Vehicle


@dataclass(frozen=True)
class CameraView:
    """The rectangular slice of the city one camera covers.

    Attributes:
        camera_id: Stable identifier, used as the frame's provenance and as
            the seed for that camera's fixed sensor texture.
        origin: Top-left tile of the covered region, in world tile space.
        width_tiles: Region width, in tiles.
        height_tiles: Region height, in tiles.
        tile_size_px: How many pixels one tile occupies in the frame.
    """

    camera_id: str
    origin: Position
    width_tiles: int
    height_tiles: int
    tile_size_px: int

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ConfigValidationError("CameraView.camera_id must not be empty")
        for name, value in (
            ("width_tiles", self.width_tiles),
            ("height_tiles", self.height_tiles),
            ("tile_size_px", self.tile_size_px),
        ):
            if value <= 0:
                raise ConfigValidationError(f"CameraView.{name} must be positive, got {value}")

    @property
    def frame_width(self) -> int:
        """Frame width in pixels."""
        return self.width_tiles * self.tile_size_px

    @property
    def frame_height(self) -> int:
        """Frame height in pixels."""
        return self.height_tiles * self.tile_size_px

    def covers(self, position: Position) -> bool:
        """Whether ``position`` falls inside this camera's footprint."""
        return (
            self.origin.x <= position.x < self.origin.x + self.width_tiles
            and self.origin.y <= position.y < self.origin.y + self.height_tiles
        )

    def tile_rect(self, position: Position) -> tuple[int, int, int, int]:
        """Pixel bounds ``(x_min, y_min, x_max, y_max)`` of a world tile.

        The result is exclusive of ``x_max``/``y_max``, matching numpy
        slicing. It is *not* clipped to the frame — callers that may be
        looking at an off-camera tile should check :meth:`covers` first.
        """
        x_min = (position.x - self.origin.x) * self.tile_size_px
        y_min = (position.y - self.origin.y) * self.tile_size_px
        return (x_min, y_min, x_min + self.tile_size_px, y_min + self.tile_size_px)

    def to_world(self, x_px: int, y_px: int) -> Position:
        """Map a frame pixel back to the world tile it was painted from.

        This is the inverse of :meth:`tile_rect` and the step that turns a
        detection into something the occupancy grid can record.

        Raises:
            ConfigValidationError: If the pixel lies outside the frame.
        """
        if not 0 <= x_px < self.frame_width or not 0 <= y_px < self.frame_height:
            raise ConfigValidationError(
                f"Pixel ({x_px}, {y_px}) is outside camera '{self.camera_id}'s "
                f"{self.frame_width}x{self.frame_height} frame"
            )
        return Position(
            x=self.origin.x + x_px // self.tile_size_px,
            y=self.origin.y + y_px // self.tile_size_px,
        )

    def world_tiles(self) -> list[Position]:
        """Every world tile this camera sees, in row-major order."""
        return [
            Position(self.origin.x + dx, self.origin.y + dy)
            for dy in range(self.height_tiles)
            for dx in range(self.width_tiles)
        ]


@dataclass(frozen=True)
class OnboardCamera:
    """The vehicle's forward-looking camera, recomputed as it drives.

    Unlike a CCTV camera it has no fixed footprint — it is a square window
    that follows the vehicle. The window is clamped to the map edges rather
    than allowed to hang off them, so every frame it produces has identical
    dimensions. A convolutional network cannot accept a frame that changes
    shape when the vehicle reaches a corner.

    Attributes:
        camera_id: Identifier stamped onto the frames it produces.
        span_tiles: Side length of the square window, in tiles. Odd values
            keep the vehicle exactly centred away from the map edges.
        tile_size_px: Pixels per tile, usually finer than the CCTV cameras
            since this view is what local obstacle detection runs on.
    """

    camera_id: str
    span_tiles: int
    tile_size_px: int

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ConfigValidationError("OnboardCamera.camera_id must not be empty")
        if self.span_tiles <= 0:
            raise ConfigValidationError(
                f"OnboardCamera.span_tiles must be positive, got {self.span_tiles}"
            )
        if self.tile_size_px <= 0:
            raise ConfigValidationError(
                f"OnboardCamera.tile_size_px must be positive, got {self.tile_size_px}"
            )

    def view_for(self, vehicle: Vehicle, map_width: int, map_height: int) -> CameraView:
        """The window centred on ``vehicle``, clamped inside the map."""
        half = self.span_tiles // 2
        return CameraView(
            camera_id=self.camera_id,
            origin=Position(
                x=_clamp(vehicle.position.x - half, map_width - self.span_tiles),
                y=_clamp(vehicle.position.y - half, map_height - self.span_tiles),
            ),
            width_tiles=self.span_tiles,
            height_tiles=self.span_tiles,
            tile_size_px=self.tile_size_px,
        )


def _clamp(value: int, upper: int) -> int:
    """Clamp ``value`` into ``[0, upper]``, tolerating a negative ``upper``.

    ``upper`` goes negative when the window is larger than the map, in
    which case the only sensible origin is 0 and the view simply overhangs
    — the rasterizer paints those tiles as off-map void.
    """
    return max(0, min(value, upper)) if upper > 0 else 0
