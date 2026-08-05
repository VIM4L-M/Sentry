"""Shows what the cameras see, beside the world they are watching.

The point of this panel is that it will not change in Phase 3. Today it
displays raw frames from :class:`~sentry_ai.sensors.rig.SensorRig`; once a
detector exists it displays the same frames with boxes drawn on them. The
window layout, the thumbnail geometry, and the labels all stay put, because
what arrives here is a :class:`~sentry_ai.sensors.frame.CameraFrame` either
way — and a frame already carries its detections.

Frames are drawn through their ground-truth annotations right now, which
makes the panel double as a check that the labels line up with the pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from sentry_ai.common.color import Color
from sentry_ai.rendering.theme import Theme
from sentry_ai.sensors.frame import CameraFrame

#: Point size for the per-thumbnail camera label.
_LABEL_SIZE_PX = 12

#: Colour per detector class, so a box says what it found without a caption.
_LABEL_COLORS: dict[str, Color] = {
    "victim": Color(235, 96, 150),
    "fire": Color(240, 150, 60),
    "obstacle": Color(190, 170, 120),
}


@dataclass(frozen=True)
class CameraPanelLayout:
    """Pixel geometry of the camera strip.

    Attributes:
        width_px: Total panel width, including padding.
        padding_px: Inset from the panel's edges.
        label_height_px: Vertical space reserved above each thumbnail.
        gap_px: Vertical space between thumbnails.
    """

    width_px: int = 240
    padding_px: int = 8
    label_height_px: int = 14
    gap_px: int = 6


class CameraPanelRenderer:
    """Draws camera frames as a labelled vertical strip."""

    def __init__(self, theme: Theme, layout: CameraPanelLayout | None = None) -> None:
        """Create the renderer.

        Args:
            theme: Palette; only ``theme.hud`` is used.
            layout: Pixel geometry. Defaults to :class:`CameraPanelLayout`.
        """
        self._theme = theme
        self._layout = layout or CameraPanelLayout()
        self._font: pygame.font.Font | None = None

    @property
    def width_px(self) -> int:
        """How much horizontal space this panel needs."""
        return self._layout.width_px

    def draw(
        self,
        surface: pygame.Surface,
        frames: list[CameraFrame],
        left: int,
        top: int,
        height: int,
    ) -> None:
        """Draw ``frames`` stacked in the strip at ``left``.

        Thumbnails are sized to share ``height`` equally, so adding a fifth
        camera shrinks all five rather than overflowing the window.
        """
        layout = self._layout
        pygame.draw.rect(
            surface,
            self._theme.hud.panel.as_tuple(),
            pygame.Rect(left, top, layout.width_px, height),
        )
        if not frames:
            return

        slot_height = height // len(frames)
        for index, frame in enumerate(frames):
            self._draw_slot(surface, frame, left, top + index * slot_height, slot_height)

    def _draw_slot(
        self, surface: pygame.Surface, frame: CameraFrame, left: int, top: int, height: int
    ) -> None:
        """Draw one label plus its thumbnail, letterboxed into the slot."""
        layout = self._layout
        self._draw_label(surface, frame, left + layout.padding_px, top + 2)

        available = pygame.Rect(
            left + layout.padding_px,
            top + layout.label_height_px,
            layout.width_px - 2 * layout.padding_px,
            height - layout.label_height_px - layout.gap_px,
        )
        if available.width <= 0 or available.height <= 0:
            return

        thumbnail = self._thumbnail(frame, available)
        surface.blit(thumbnail, thumbnail.get_rect(center=available.center))

    def _draw_label(
        self, surface: pygame.Surface, frame: CameraFrame, x: int, y: int
    ) -> None:
        """Camera id plus a per-class tally of what it can currently see."""
        counts: dict[str, int] = {}
        for detection in frame.annotations:
            counts[detection.label.value] = counts.get(detection.label.value, 0) + 1
        summary = " ".join(f"{name[:3]}:{count}" for name, count in sorted(counts.items()))
        text = f"{frame.camera_id}  {summary}" if summary else frame.camera_id
        surface.blit(self._label_font().render(text, True, self._theme.hud.text.as_tuple()), (x, y))

    def _thumbnail(self, frame: CameraFrame, available: pygame.Rect) -> pygame.Surface:
        """A scaled copy of ``frame``, with its detections boxed."""
        # Pygame surfaces are column-major (x, y); numpy frames are (y, x).
        source = pygame.surfarray.make_surface(np.transpose(frame.pixels, (1, 0, 2)))
        self._draw_boxes(source, frame)

        scale = min(available.width / frame.width, available.height / frame.height)
        size = (max(1, int(frame.width * scale)), max(1, int(frame.height * scale)))
        return pygame.transform.smoothscale(source, size)

    def _draw_boxes(self, source: pygame.Surface, frame: CameraFrame) -> None:
        """Outline every detection, at full resolution before scaling.

        Drawn before the downscale so a one-pixel box does not vanish into
        the interpolation.
        """
        for detection in frame.annotations:
            color = _LABEL_COLORS.get(detection.label.value, self._theme.hud.accent)
            box = detection.bbox
            pygame.draw.rect(
                source,
                color.as_tuple(),
                pygame.Rect(
                    box.x_min, box.y_min, box.x_max - box.x_min, box.y_max - box.y_min
                ),
                width=1,
            )

    def _label_font(self) -> pygame.font.Font:
        """The label font, created on first use."""
        if self._font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._font = pygame.font.SysFont("consolas,dejavusansmono,monospace", _LABEL_SIZE_PX)
        return self._font
