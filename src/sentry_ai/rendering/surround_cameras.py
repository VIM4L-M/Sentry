"""Four surround cameras around the vehicle: front, left, right, rear (Phase 9).

The column beside the drive view on a city-sized map, where the fixed CCTV
cameras cover only one corner of the city and tell a viewer nothing. Each
view looks out from the vehicle in one direction and turns with it, like
the camera grid on a car's display, so a car pulling up behind or a person
stepping off the pavement on the left is visible before it matters. The
views are drawn by :meth:`DriveViewRenderer.draw_camera`; this panel only
lays them out and labels them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from sentry_ai.domain.map import CityMap
from sentry_ai.domain.traffic import AgentKind
from sentry_ai.rendering.drive_view import DriveViewRenderer, TrafficView
from sentry_ai.rendering.motion import VehiclePose
from sentry_ai.rendering.theme import Theme

#: Each camera: its label and how far it turns from the vehicle's heading.
CAMERAS: tuple[tuple[str, float], ...] = (
    ("FRONT", 0.0),
    ("LEFT", -math.pi / 2),
    ("RIGHT", math.pi / 2),
    ("REAR", math.pi),
)

_ALERT = (206, 36, 36)

#: The views are redrawn on every this-many-th frame.
REFRESH_FRAMES = 2


@dataclass(frozen=True)
class SurroundLayout:
    """Pixel geometry of the surround-camera column."""

    width_px: int = 240
    padding_px: int = 8
    title_px: int = 22
    label_px: int = 18


class SurroundCameraPanel:
    """Lays out the four surround cameras in one column."""

    def __init__(
        self, theme: Theme, renderer: DriveViewRenderer, layout: SurroundLayout | None = None
    ) -> None:
        """Create the panel. ``pygame.font`` must already be initialised."""
        self._theme = theme
        self._renderer = renderer
        self._layout = layout or SurroundLayout()
        self._font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 13, bold=True)
        self._cache: pygame.Surface | None = None
        self._frame = 0

    @property
    def width_px(self) -> int:
        """How much horizontal space this panel needs."""
        return self._layout.width_px

    def draw(
        self,
        surface: pygame.Surface,
        left: int,
        height: int,
        city_map: CityMap,
        pose: VehiclePose,
        traffic: TrafficView | None,
        background: pygame.Surface | None = None,
    ) -> None:
        """Draw the column with its left edge at ``left``, ``height`` tall.

        The four views are redrawn every :data:`REFRESH_FRAMES` frames and
        reused in between: side cameras at half the frame rate are
        indistinguishable, and it keeps the main view smooth.
        """
        column = pygame.Rect(left, 0, self._layout.width_px, height)
        self._frame += 1
        if (
            self._cache is None
            or self._cache.get_size() != column.size
            or (self._frame % REFRESH_FRAMES == 0)
        ):
            self._cache = pygame.Surface(column.size)
            self._draw_views(self._cache, height, city_map, pose, traffic, background)
        surface.blit(self._cache, column)

    def _draw_views(
        self,
        surface: pygame.Surface,
        height: int,
        city_map: CityMap,
        pose: VehiclePose,
        traffic: TrafficView | None,
        background: pygame.Surface | None,
    ) -> None:
        layout = self._layout
        hud = self._theme.hud
        surface.fill(hud.panel.as_tuple())
        left = 0
        x, y = left + layout.padding_px, layout.padding_px
        surface.blit(self._font.render("SURROUND CAMERAS", True, hud.accent.as_tuple()), (x, y))
        y += layout.title_px
        inner = layout.width_px - 2 * layout.padding_px
        view_height = (height - y) // len(CAMERAS) - layout.label_px - layout.padding_px
        for label, facing in CAMERAS:
            rect = pygame.Rect(x, y + layout.label_px, inner, view_height)
            braking = traffic is not None and traffic.brake is not None and label == "FRONT"
            text = f"{label}  {_count(traffic, pose, facing)}"
            color = _ALERT if braking else hud.text.as_tuple()
            surface.blit(self._font.render(text, True, color), (x, y))
            self._renderer.draw_camera(surface, rect, city_map, pose, facing, traffic, background)
            border = _ALERT if braking else hud.accent.blended_with(hud.panel, 0.5).as_tuple()
            pygame.draw.rect(surface, border, rect, 2 if braking else 1)
            y = rect.bottom + layout.padding_px


def _count(traffic: TrafficView | None, pose: VehiclePose, facing: float) -> str:
    """How many tracked cars and people lie in this camera's quarter."""
    if traffic is None:
        return ""
    cars = people = 0
    direction = pose.angle + facing
    ax, ay = math.sin(direction), -math.cos(direction)
    for agent in traffic.near(pose.x, pose.y, traffic.sensor_range):
        x, y = traffic.where(agent)
        dx, dy = x - pose.x, y - pose.y
        distance = math.hypot(dx, dy)
        if distance == 0 or distance > traffic.sensor_range:
            continue
        if (dx * ax + dy * ay) / distance < math.cos(math.pi / 4):
            continue
        if agent.kind is AgentKind.CAR:
            cars += 1
        else:
            people += 1
    return f"{cars} car{'s' * (cars != 1)} · {people} {'person' if people == 1 else 'people'}"
