"""The street-view column: a real photo of where the vehicle is (Phase 9, key ``P``).

Shown only on maps with street photos fetched (``fetch_street_photos.py``).
Top: the Mapillary photo for the vehicle's tile, the one whose camera faced
closest to the vehicle's heading; if the tile has none, the nearest tile's
within :data:`SEARCH_TILES`, labelled with how far away it is. Bottom: the
simulated onboard camera — what the detector actually sees — so the gap
between the real street and the model's world is on screen, not hidden.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pygame

from sentry_ai.common.color import Color
from sentry_ai.domain.entities import Position, Vehicle
from sentry_ai.rendering.theme import Theme
from sentry_ai.sensors.frame import CameraFrame
from sentry_ai.sequence.behaviour import heading_to_degrees

#: How many tiles away the nearest photo may be when the vehicle's own tile has none.
SEARCH_TILES = 4


@dataclass(frozen=True)
class StreetViewLayout:
    """Pixel geometry of the street-view column."""

    width_px: int = 340
    padding_px: int = 10
    font_size_px: int = 13


class StreetPhotoLibrary:
    """The fetched photos for one map, indexed by road tile."""

    def __init__(self, folder: Path, tile_metres: float) -> None:
        """Load ``folder/index.json`` written by ``fetch_street_photos.py``."""
        index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
        self.attribution: str = index["attribution"]
        self._folder = folder
        self._tile_metres = tile_metres
        self._tiles: dict[tuple[int, int], list[tuple[str, float]]] = {
            tuple(int(v) for v in key.split(",")): [(p["id"], p["compass"]) for p in photos]  # type: ignore[misc]
            for key, photos in index["tiles"].items()
        }
        self._cache: dict[tuple[str, tuple[int, int]], pygame.Surface] = {}

    @classmethod
    def for_map(
        cls, root: Path, map_path: Path, map_data: dict[str, object]
    ) -> StreetPhotoLibrary | None:
        """The library for ``map_path``, or ``None`` if nothing was fetched or it has no geo box."""
        folder = root / "data" / "maps" / "street" / map_path.stem
        geo = map_data.get("geo")
        if not (folder / "index.json").is_file() or not isinstance(geo, dict):
            return None
        width = int(str(map_data["width"]))
        span = (geo["east"] - geo["west"]) * 111_320.0 * math.cos(math.radians(geo["north"]))
        return cls(folder, span / width)

    def lookup(self, vehicle: Vehicle) -> tuple[str, float] | None:
        """``(image_id, metres away)`` for the vehicle's pose, or ``None``."""
        tile = self._nearest_tile(vehicle.position)
        if tile is None:
            return None
        bearing = heading_to_degrees(vehicle.heading)
        image_id, _ = min(
            self._tiles[tile], key=lambda p: abs((p[1] - bearing + 180.0) % 360.0 - 180.0)
        )
        steps = abs(tile[0] - vehicle.position.x) + abs(tile[1] - vehicle.position.y)
        return image_id, steps * self._tile_metres

    def path(self, image_id: str) -> Path:
        """The photo's file on disk, for a detector to read."""
        return self._folder / f"{image_id}.jpg"

    def image(self, image_id: str, size: tuple[int, int]) -> pygame.Surface:
        """The photo scaled to ``size``, loaded once."""
        key = (image_id, size)
        if key not in self._cache:
            raw = pygame.image.load(str(self._folder / f"{image_id}.jpg"))
            self._cache[key] = pygame.transform.smoothscale(raw, size)
        return self._cache[key]

    def _nearest_tile(self, position: Position) -> tuple[int, int] | None:
        here = (position.x, position.y)
        if here in self._tiles:
            return here
        candidates = [
            t for t in self._tiles
            if abs(t[0] - here[0]) + abs(t[1] - here[1]) <= SEARCH_TILES
        ]  # fmt: skip
        if not candidates:
            return None
        return min(candidates, key=lambda t: (abs(t[0] - here[0]) + abs(t[1] - here[1]), t))


class StreetViewPanel:
    """Draws the real photo above the simulated onboard frame."""

    def __init__(self, theme: Theme, layout: StreetViewLayout | None = None) -> None:
        """Create the panel. ``pygame.font`` must already be initialised."""
        self._theme = theme
        self._layout = layout or StreetViewLayout()
        family = "consolas,dejavusansmono,monospace"
        self._font = pygame.font.SysFont(family, self._layout.font_size_px)
        self._bold = pygame.font.SysFont(family, self._layout.font_size_px + 2, bold=True)

    @property
    def width_px(self) -> int:
        """How much horizontal space this panel needs."""
        return self._layout.width_px

    def draw(
        self,
        surface: pygame.Surface,
        library: StreetPhotoLibrary,
        vehicle: Vehicle,
        onboard: CameraFrame | None,
        left: int,
        height: int,
    ) -> None:
        """Draw the column with its left edge at ``left``."""
        layout = self._layout
        hud = self._theme.hud
        pygame.draw.rect(
            surface, hud.panel.as_tuple(), pygame.Rect(left, 0, layout.width_px, height)
        )
        pygame.draw.line(surface, hud.accent.as_tuple(), (left, 0), (left, height), 2)
        x, y = left + layout.padding_px, layout.padding_px
        inner = layout.width_px - 2 * layout.padding_px
        y = self._title(surface, "STREET VIEW  real photo", x, y)
        photo_size = (inner, inner * 9 // 16)
        found = library.lookup(vehicle)
        frame = pygame.Rect(x, y, *photo_size)
        if found is None:
            pygame.draw.rect(surface, hud.panel.blended_with(hud.text, 0.12).as_tuple(), frame)
            self._text(surface, "no street photo near here", x + 8, y + photo_size[1] // 2 - 8)
        else:
            image_id, metres = found
            surface.blit(library.image(image_id, photo_size), frame)
            where = "at the vehicle" if metres == 0 else f"{metres:.0f} m from the vehicle"
            self._text(surface, where, x, frame.bottom + 4)
        y = frame.bottom + 24
        muted = hud.text.blended_with(hud.panel, 0.45)
        for line in library.attribution.split(", ", 1):
            self._text(surface, line, x, y, muted)
            y += self._layout.font_size_px + 3
        y += 14
        y = self._title(surface, "WHAT THE AI SEES  simulated", x, y)
        if onboard is not None:
            side = min(inner, height - y - layout.padding_px)
            raw = pygame.surfarray.make_surface(onboard.pixels.swapaxes(0, 1))
            surface.blit(pygame.transform.scale(raw, (side, side)), (x + (inner - side) // 2, y))

    def _title(self, surface: pygame.Surface, text: str, x: int, y: int) -> int:
        surface.blit(self._bold.render(text, True, self._theme.hud.accent.as_tuple()), (x, y))
        return y + self._layout.font_size_px + 10

    def _text(
        self, surface: pygame.Surface, text: str, x: int, y: int, color: Color | None = None
    ) -> None:
        tone = color if color is not None else self._theme.hud.text
        surface.blit(self._font.render(text, True, tone.as_tuple()), (x, y))
