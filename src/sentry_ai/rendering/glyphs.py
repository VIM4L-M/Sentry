"""Recognisable shapes for the operator display, drawn procedurally.

Flat coloured circles told a viewer that *something* was at a tile but not
what, which made a 30-second demo impossible to follow without narration.
These glyphs fix that: a victim reads as a person, fire as a flame, debris
as broken slabs.

Drawn with Pygame primitives rather than loaded from image files, on
purpose. Sprite assets would mean binary files in the repo, an asset
loader, a search path, scaling policy, and a licence question — a
surprising amount of machinery for what is fundamentally "draw a stick
figure". Every glyph here scales from the rect it is given, so changing
``render.tile_size_px`` needs no new artwork.

This module is the *operator's* view only. What the cameras see is
:mod:`sentry_ai.sensors.rasterizer`, which is deliberately separate and
deliberately never animated — a detector must be trained on stable imagery
even while the human-facing map flickers.
"""

from __future__ import annotations

import math

import pygame

from sentry_ai.common.color import Color
from sentry_ai.domain.enums import Heading

#: Flame outline as fractions of the tile rect, drawn tip-first. Asymmetric
#: on purpose: a symmetrical flame reads as an arrow or a tree.
_FLAME_POINTS: tuple[tuple[float, float], ...] = (
    (0.50, 0.05),
    (0.72, 0.34),
    (0.64, 0.44),
    (0.85, 0.68),
    (0.72, 0.92),
    (0.28, 0.92),
    (0.15, 0.66),
    (0.36, 0.42),
    (0.30, 0.30),
)

#: Debris slabs as (x, y, w, h) fractions — a heap, not a neat square.
_DEBRIS_SLABS: tuple[tuple[float, float, float, float], ...] = (
    (0.08, 0.55, 0.40, 0.30),
    (0.45, 0.62, 0.45, 0.25),
    (0.24, 0.28, 0.34, 0.30),
    (0.58, 0.34, 0.28, 0.24),
)

#: How far a flame shrinks at the bottom of its flicker cycle.
_FLICKER_DEPTH = 0.12

#: Vehicle body as a chevron pointing along its heading.
_CHEVRON_POINTS: tuple[tuple[float, float], ...] = (
    (0.50, 0.06),
    (0.92, 0.86),
    (0.50, 0.66),
    (0.08, 0.86),
)


def draw_victim(surface: pygame.Surface, rect: pygame.Rect, color: Color) -> None:
    """A small standing figure: head, body, and arms."""
    rgb = color.as_tuple()
    head_radius = max(1, rect.width // 7)
    head_y = rect.top + rect.height // 4
    pygame.draw.circle(surface, rgb, (rect.centerx, head_y), head_radius)

    thickness = max(1, rect.width // 9)
    body_top = head_y + head_radius
    body_bottom = rect.bottom - rect.height // 6
    pygame.draw.line(
        surface, rgb, (rect.centerx, body_top), (rect.centerx, body_bottom), thickness
    )
    arm_y = body_top + (body_bottom - body_top) // 3
    pygame.draw.line(
        surface,
        rgb,
        (rect.centerx - rect.width // 4, arm_y),
        (rect.centerx + rect.width // 4, arm_y),
        thickness,
    )


def draw_fire(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: Color,
    core_color: Color,
    flicker: float = 0.0,
) -> None:
    """A flame with a brighter core.

    Args:
        surface: Target surface.
        rect: Tile bounds to draw within.
        color: Outer flame colour.
        core_color: Inner, hotter colour.
        flicker: Animation phase in ``[0, 1)``. Scales the flame's height a
            few percent so the fire looks alive without moving tile to tile.
    """
    # Always <= 1.0: the flame shrinks and recovers rather than growing past
    # full height. Scaling above 1.0 would push the tip out of the tile and
    # into the one above, which misreports where the fire actually is.
    wobble = 1.0 - _FLICKER_DEPTH * (1.0 + math.sin(flicker * math.tau)) / 2.0
    points = [
        (
            rect.left + fx * rect.width,
            rect.bottom - (1.0 - fy) * rect.height * wobble,
        )
        for fx, fy in _FLAME_POINTS
    ]
    pygame.draw.polygon(surface, color.as_tuple(), points)

    core = rect.inflate(-int(rect.width * 0.66), -int(rect.height * 0.60))
    core.centery = rect.centery + rect.height // 5
    pygame.draw.ellipse(surface, core_color.as_tuple(), core)


def draw_debris(surface: pygame.Surface, rect: pygame.Rect, color: Color) -> None:
    """A heap of broken slabs, each outlined so the pile reads as rubble."""
    rgb = color.as_tuple()
    edge = _shade(color, 0.55).as_tuple()
    for fx, fy, fw, fh in _DEBRIS_SLABS:
        slab = pygame.Rect(
            rect.left + int(fx * rect.width),
            rect.top + int(fy * rect.height),
            max(2, int(fw * rect.width)),
            max(2, int(fh * rect.height)),
        )
        pygame.draw.rect(surface, rgb, slab)
        pygame.draw.rect(surface, edge, slab, width=1)


def draw_tree(surface: pygame.Surface, rect: pygame.Rect, color: Color) -> None:
    """A canopy on a trunk.

    Trees are ``Obstacle`` entities like rubble is, but drawing them as
    rubble made a park read as a collapsed building. They block movement
    for the same reason and mean something completely different.
    """
    trunk_width = max(2, rect.width // 6)
    trunk = pygame.Rect(
        rect.centerx - trunk_width // 2,
        rect.centery,
        trunk_width,
        rect.bottom - rect.centery - rect.height // 8,
    )
    pygame.draw.rect(surface, _shade(color, 0.5).as_tuple(), trunk)
    pygame.draw.circle(
        surface,
        color.as_tuple(),
        (rect.centerx, rect.centery - rect.height // 10),
        max(2, rect.width // 3),
    )


def draw_vehicle(
    surface: pygame.Surface, rect: pygame.Rect, color: Color, heading: Heading
) -> None:
    """A chevron pointing the way the vehicle faces.

    The shape carries the heading, so the separate heading stub the old
    renderer drew is no longer needed — one glyph, one fact.
    """
    draw_vehicle_facing(surface, rect, color, _heading_radians(heading))


def draw_vehicle_facing(
    surface: pygame.Surface, rect: pygame.Rect, color: Color, radians: float
) -> None:
    """The vehicle chevron turned ``radians`` clockwise from north.

    For a vehicle drawn mid-turn, between the four headings.
    """
    points = [
        (rect.left + fx * rect.width, rect.top + fy * rect.height) for fx, fy in _CHEVRON_POINTS
    ]
    rotated = [_rotate_about(point, rect.center, radians) for point in points]
    pygame.draw.polygon(surface, color.as_tuple(), rotated)
    pygame.draw.polygon(surface, _shade(color, 0.6).as_tuple(), rotated, width=1)


def draw_hospital(surface: pygame.Surface, rect: pygame.Rect, color: Color) -> None:
    """A medical cross marking the safe zone."""
    rgb = color.as_tuple()
    arm = max(2, rect.width // 4)
    vertical = pygame.Rect(
        rect.centerx - arm // 2, rect.top + arm // 2, arm, rect.height - arm
    )
    horizontal = pygame.Rect(
        rect.left + arm // 2, rect.centery - arm // 2, rect.width - arm, arm
    )
    pygame.draw.rect(surface, rgb, vertical)
    pygame.draw.rect(surface, rgb, horizontal)


def _heading_radians(heading: Heading) -> float:
    """Rotation needed to point a north-facing glyph along ``heading``."""
    return {
        Heading.NORTH: 0.0,
        Heading.EAST: math.pi / 2,
        Heading.SOUTH: math.pi,
        Heading.WEST: 3 * math.pi / 2,
    }[heading]


def _rotate_about(
    point: tuple[float, float], centre: tuple[int, int], radians: float
) -> tuple[float, float]:
    """Rotate ``point`` clockwise about ``centre`` in screen space."""
    dx, dy = point[0] - centre[0], point[1] - centre[1]
    cos, sin = math.cos(radians), math.sin(radians)
    return (centre[0] + dx * cos - dy * sin, centre[1] + dx * sin + dy * cos)


def _shade(color: Color, factor: float) -> Color:
    """A darker (``factor`` < 1) or lighter version of ``color``."""
    return Color(
        r=min(255, round(color.r * factor)),
        g=min(255, round(color.g * factor)),
        b=min(255, round(color.b * factor)),
    )
