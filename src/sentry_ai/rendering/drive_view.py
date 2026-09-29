"""A Tesla-style driving view that follows the vehicle (Phase 9 display, key ``F``).

For city-sized maps, where the whole map shown at once makes every street a
pixel wide. The camera follows the vehicle and the map turns with it, so the
vehicle always points up the screen and the road ahead runs up the view, the
way a car's autopilot display shows it:

* the street plan in dark, low-contrast colours (or the satellite photo),
  with kerbs and centre-line dashes on the roads;
* the planned route as a blue ribbon from the vehicle to its goal;
* what the onboard camera's model detected this tick, boxed and labelled
  with its confidence — the AI's view, not the ground truth;
* victims, fires, damage and the hospital, drawn over the street;
* a banner with where the vehicle is going and how far, and a minimap of
  the whole city in the corner.

Flat 2D, pure pygame. The map is drawn north-up into a square patch around
the vehicle and the patch is rotated as one image, so the per-tile work does
not depend on the heading.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pygame

from sentry_ai.common.color import Color
from sentry_ai.decision.emergency_brake import BrakeEvent
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind, TerrainType, VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.traffic import AgentKind, TrafficAgent
from sentry_ai.interfaces.navigation import Route
from sentry_ai.perception.scene_evidence import Sighting
from sentry_ai.rendering import glyphs
from sentry_ai.rendering.motion import VehiclePose

_ASPHALT = Color(52, 55, 62)
_KERB = Color(92, 97, 108)
_LANE = Color(150, 154, 162)
_LANE_ON_IMAGERY = Color(236, 226, 180)
_BLOCK = Color(30, 33, 40)
_BLOCK_EDGE = Color(44, 48, 58)
_GROUND = Color(22, 24, 29)
_TREE = Color(34, 70, 48)
_DAMAGE = Color(120, 72, 40)
_ROUTE = Color(58, 132, 255)
_TEXT = Color(232, 234, 238)
_MUTED = Color(150, 156, 168)
_WHITE = Color(255, 255, 255)

#: Terrain the vehicle drives on, drawn as asphalt.
_ROADS = frozenset({TerrainType.ROAD, TerrainType.SAFE_ZONE, TerrainType.BLOCKED_ROAD})

#: Terrain drawn as damage over the street plan.
_DAMAGED = frozenset({TerrainType.COLLAPSED_BUILDING, TerrainType.RUBBLE, TerrainType.BLOCKED_ROAD})

_CAR = Color(196, 202, 212)
_PERSON = Color(246, 206, 96)
#: Chennai's autorickshaws: yellow body, black canopy.
_AUTO = Color(238, 200, 40)
_AUTO_TOP = Color(30, 30, 30)
_BIKE = Color(200, 70, 70)
_COW = Color(236, 232, 222)
_TRACKED = Color(120, 230, 255)

#: How far, in tiles, a vehicle sits left of the road's centre line.
_LANE_OFFSET = 0.22

#: Nearest tracked road users that get a distance tag.
_TAGGED_ROAD_USERS = 3

#: Tiles across one surround-camera view.
_CAMERA_TILES_ACROSS = 9.0

#: Map tile coordinates to pixel coordinates in the patch.
_Projector = Callable[[float, float], tuple[int, int]]

#: What each detection class is called on screen.
_LABELS = {EntityKind.VICTIM: "VICTIM", EntityKind.FIRE: "FIRE", EntityKind.OBSTACLE: "DEBRIS"}


@dataclass(frozen=True)
class DriveViewLayout:
    """How much city the view shows and where the vehicle sits in it."""

    tiles_across: float = 30.0
    #: The vehicle's height up the view, as a share of it: more road ahead than behind.
    vehicle_height: float = 0.3
    minimap_px: int = 180
    font_size_px: int = 14
    #: Darkening laid over satellite imagery, 0-255, so the overlays stay readable.
    imagery_dim: int = 110


@dataclass(frozen=True)
class DriveStatus:
    """The banner's content: what the vehicle is doing and how far it has to go."""

    phase: str
    goal: str
    #: ``None`` on maps with no real-world scale: distances are given in tiles.
    metres_per_tile: float | None
    #: The vehicle's real-world speed right now, when the map has a real scale.
    speed_kmh: float | None = None
    #: How many times faster than real life the replay is running.
    time_factor: float | None = None


@dataclass(frozen=True)
class TrafficView:
    """The road users to draw, and where each is along its current move.

    Attributes:
        agents: Every car and pedestrian.
        clock: The traffic clock, including the fraction of the tick in
            progress, so agents glide between tiles like the vehicle does.
        step_ticks: Ticks one move spans, for each kind of road user.
        sensor_range: Tiles within which the surround sensors track an agent;
            tracked agents are outlined and the nearest are tagged.
        brake: The emergency brake's intervention this tick, if any.
    """

    agents: Sequence[TrafficAgent]
    clock: float
    step_ticks: Mapping[AgentKind, int]
    sensor_range: float = 5.0
    brake: BrakeEvent | None = None

    def near(self, x: float, y: float, reach: float) -> list[TrafficAgent]:
        """Agents whose tile lies within ``reach`` tiles (square) of ``(x, y)``.

        A cheap cut on the tile alone, before :meth:`where` does the
        interpolation: a city has hundreds of agents and a view shows a few.
        """
        limit = reach + 1.5
        return [
            agent
            for agent in self.agents
            if abs(agent.position.x + 0.5 - x) <= limit and abs(agent.position.y + 0.5 - y) <= limit
        ]

    def where(self, agent: TrafficAgent) -> tuple[float, float]:
        """The agent's drawn centre, in tiles, part way along its last move."""
        step = self.step_ticks.get(agent.kind, 1)
        t = min(1.0, max(0.0, (self.clock - agent.moved_at) / step))
        x = agent.previous.x + (agent.position.x - agent.previous.x) * t
        y = agent.previous.y + (agent.position.y - agent.previous.y) * t
        if agent.kind.is_vehicle:
            # Keep left, as Indian traffic drives: each vehicle sits in the
            # left half of its road tile relative to where it is heading, so
            # oncoming traffic passes on the other side of the centre line.
            lx, ly = agent.heading.turn_left().delta
            x, y = x + lx * _LANE_OFFSET, y + ly * _LANE_OFFSET
        return x + 0.5, y + 0.5


class DriveViewRenderer:
    """Draws the follow-camera view of a :class:`CityMap` into a rectangle."""

    def __init__(self, layout: DriveViewLayout | None = None) -> None:
        """Create the renderer. ``pygame.font`` must be initialised before drawing."""
        self._layout = layout or DriveViewLayout()
        self._flicker = 0.0
        self._fonts: tuple[pygame.font.Font, pygame.font.Font] | None = None
        self._minimap: pygame.Surface | None = None
        self._minimap_age = math.inf
        self._imagery: tuple[int, pygame.Surface] | None = None

    def advance_animation(self, delta_seconds: float) -> None:
        """Advance the fire flicker and the victim beacons by frame time."""
        self._flicker = (self._flicker + delta_seconds) % 1.0
        self._minimap_age += delta_seconds

    def draw(
        self,
        surface: pygame.Surface,
        area: pygame.Rect,
        city_map: CityMap,
        pose: VehiclePose,
        route: Route | None = None,
        sightings: Sequence[Sighting] = (),
        background: pygame.Surface | None = None,
        status: DriveStatus | None = None,
        traffic: TrafficView | None = None,
    ) -> None:
        """Draw the view of ``city_map`` around ``pose`` into ``area``."""
        scale = area.width / self._layout.tiles_across
        anchor = (area.centerx, area.bottom - int(area.height * self._layout.vehicle_height))
        reach = _reach(area, anchor) / scale + 1.5
        patch = self._patch(city_map, pose, reach, scale, route, sightings, background, traffic)
        # rotate, not rotozoom: no anti-aliasing, but 3-4x cheaper on a
        # ~1,300 px patch, and exact at the four headings the vehicle cruises at.
        rotated = pygame.transform.rotate(patch, math.degrees(pose.angle))
        previous_clip = surface.get_clip()
        surface.set_clip(area)
        surface.fill(_GROUND.as_tuple(), area)
        surface.blit(rotated, rotated.get_rect(center=anchor))
        self._draw_vehicle(surface, anchor, scale)
        self._draw_labels(surface, city_map, pose, sightings, anchor, scale)
        if traffic is not None:
            self._draw_traffic_tags(surface, pose, traffic, anchor, scale, status)
        self._draw_banner(surface, area, city_map, route, status)
        if traffic is not None and traffic.brake is not None:
            self._draw_brake_alert(surface, area, traffic.brake)
        self._draw_minimap(surface, area, city_map, pose, route)
        surface.set_clip(previous_clip)

    def draw_camera(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        city_map: CityMap,
        pose: VehiclePose,
        facing: float,
        traffic: TrafficView | None = None,
        background: pygame.Surface | None = None,
    ) -> None:
        """One surround camera: the street on one side of the vehicle, facing out.

        ``facing`` turns the camera from the vehicle's heading (0 front, pi
        rear, -pi/2 left, pi/2 right). The vehicle sits at the bottom edge
        of ``rect``, so the view shows what lies that way.
        """
        scale = rect.width / _CAMERA_TILES_ACROSS
        anchor = (rect.centerx, rect.bottom - int(scale * 0.4))
        reach = _reach(rect, anchor) / scale + 1.5
        patch = self._patch(city_map, pose, reach, scale, None, (), background, traffic)
        rotated = pygame.transform.rotate(patch, math.degrees(pose.angle + facing))
        previous_clip = surface.get_clip()
        surface.set_clip(rect)
        surface.fill(_GROUND.as_tuple(), rect)
        surface.blit(rotated, rotated.get_rect(center=anchor))
        body = pygame.Rect(0, 0, int(scale * 0.55), int(scale * 0.55))
        body.center = anchor
        pygame.draw.rect(surface, (236, 238, 242), body, border_radius=4)
        surface.set_clip(previous_clip)

    # ------------------------------------------------------------------
    # The rotating patch
    # ------------------------------------------------------------------

    def _patch(
        self,
        city_map: CityMap,
        pose: VehiclePose,
        reach: float,
        scale: float,
        route: Route | None,
        sightings: Sequence[Sighting],
        background: pygame.Surface | None,
        traffic: TrafficView | None = None,
    ) -> pygame.Surface:
        """The map north-up in a square of ``2 * reach`` tiles centred on the vehicle."""
        side = int(2 * reach * scale)
        patch = pygame.Surface((side, side))
        patch.fill(_GROUND.as_tuple())
        origin = (pose.x - reach, pose.y - reach)

        def to_patch(x: float, y: float) -> tuple[int, int]:
            return round((x - origin[0]) * scale), round((y - origin[1]) * scale)

        tiles = _tiles_within(city_map, pose, reach)
        if background is not None:
            self._draw_imagery(patch, city_map, background, origin, scale, side)
        self._draw_streets(patch, city_map, tiles, to_patch, scale, background is not None)
        self._draw_route(patch, city_map, route, pose, to_patch, scale)
        self._draw_things(patch, city_map, pose, reach, to_patch, scale)
        if traffic is not None:
            self._draw_road_users(patch, pose, reach, traffic, to_patch, scale)
        self._draw_sightings(patch, sightings, to_patch, scale)
        return patch

    def _draw_imagery(
        self,
        patch: pygame.Surface,
        city_map: CityMap,
        background: pygame.Surface,
        origin: tuple[float, float],
        scale: float,
        side: int,
    ) -> None:
        """The satellite photo under the patch, dimmed to the view's dark palette."""
        per_tile = background.get_width() / city_map.width
        source = pygame.Rect(
            round(origin[0] * per_tile), round(origin[1] * per_tile),
            round(side / scale * per_tile), round(side / scale * per_tile),
        )  # fmt: skip
        clipped = source.clip(background.get_rect())
        if clipped.width <= 0 or clipped.height <= 0:
            return
        crop = background.subsurface(clipped)
        size = (round(clipped.width / per_tile * scale), round(clipped.height / per_tile * scale))
        at = (
            round((clipped.left - source.left) / per_tile * scale),
            round((clipped.top - source.top) / per_tile * scale),
        )
        patch.blit(pygame.transform.scale(crop, size), at)
        shade = pygame.Surface((side, side), pygame.SRCALPHA)
        shade.fill((*_GROUND.as_tuple(), self._layout.imagery_dim))
        patch.blit(shade, (0, 0))

    def _draw_streets(
        self,
        patch: pygame.Surface,
        city_map: CityMap,
        tiles: list[Position],
        to_patch: _Projector,
        scale: float,
        over_imagery: bool,
    ) -> None:
        """Asphalt with kerbs and centre dashes; dark blocks; trees; damage."""
        size = math.ceil(scale) + 1
        for position in tiles:
            terrain = city_map.tile_at(position)
            left, top = to_patch(position.x, position.y)
            rect = pygame.Rect(left, top, size, size)
            if terrain in _DAMAGED:
                pygame.draw.rect(patch, _DAMAGE.as_tuple(), rect)
                pygame.draw.line(patch, _BLOCK.as_tuple(), rect.topleft, rect.bottomright, 2)
            elif over_imagery:
                # The photo already shows the street; add only the lane line.
                if terrain is TerrainType.ROAD:
                    self._draw_centre_line(patch, city_map, position, rect)
                continue
            elif terrain in _ROADS:
                pygame.draw.rect(patch, _ASPHALT.as_tuple(), rect)
                self._draw_road_marks(patch, city_map, position, rect)
            elif terrain is TerrainType.BUILDING:
                pygame.draw.rect(patch, _BLOCK.as_tuple(), rect)
                pygame.draw.rect(patch, _BLOCK_EDGE.as_tuple(), rect, 1)
            elif terrain is TerrainType.TREE:
                pygame.draw.circle(patch, _TREE.as_tuple(), rect.center, max(2, size // 3))

    @staticmethod
    def _draw_centre_line(
        patch: pygame.Surface, city_map: CityMap, position: Position, rect: pygame.Rect
    ) -> None:
        """A dashed centre line on a straight road tile: the two lanes of keep-left traffic."""
        east_west = _is_road(city_map, position.x + 1, position.y) and _is_road(
            city_map, position.x - 1, position.y
        )
        north_south = _is_road(city_map, position.x, position.y + 1) and _is_road(
            city_map, position.x, position.y - 1
        )
        dash = max(3, rect.width // 3)
        colour = _LANE_ON_IMAGERY.as_tuple()
        if east_west and not north_south:
            pygame.draw.line(
                patch, colour, (rect.centerx - dash // 2, rect.centery),
                (rect.centerx + dash // 2, rect.centery), 2,
            )  # fmt: skip
        elif north_south and not east_west:
            pygame.draw.line(
                patch, colour, (rect.centerx, rect.centery - dash // 2),
                (rect.centerx, rect.centery + dash // 2), 2,
            )  # fmt: skip

    @staticmethod
    def _draw_road_marks(
        patch: pygame.Surface, city_map: CityMap, position: Position, rect: pygame.Rect
    ) -> None:
        """Kerbs on sides that leave the road; a dash along the way the road runs."""
        road = {
            (dx, dy): _is_road(city_map, position.x + dx, position.y + dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        }
        kerb, lane = _KERB.as_tuple(), _LANE.as_tuple()
        edges = {
            (0, -1): (rect.topleft, rect.topright),
            (0, 1): (rect.bottomleft, rect.bottomright),
            (-1, 0): (rect.topleft, rect.bottomleft),
            (1, 0): (rect.topright, rect.bottomright),
        }
        for side, (a, b) in edges.items():
            if not road[side]:
                pygame.draw.line(patch, kerb, a, b, 2)
        dash = max(2, rect.width // 3)
        if road[(1, 0)] and road[(-1, 0)] and not (road[(0, 1)] or road[(0, -1)]):
            pygame.draw.line(
                patch, lane, (rect.centerx - dash // 2, rect.centery),
                (rect.centerx + dash // 2, rect.centery), 1,
            )  # fmt: skip
        elif road[(0, 1)] and road[(0, -1)] and not (road[(1, 0)] or road[(-1, 0)]):
            pygame.draw.line(
                patch, lane, (rect.centerx, rect.centery - dash // 2),
                (rect.centerx, rect.centery + dash // 2), 1,
            )  # fmt: skip

    def _draw_route(
        self,
        patch: pygame.Surface,
        city_map: CityMap,
        route: Route | None,
        pose: VehiclePose,
        to_patch: _Projector,
        scale: float,
    ) -> None:
        """The rest of the route as a blue ribbon, starting at the (gliding) vehicle."""
        ahead = remaining(route, city_map.vehicle.position)
        if not ahead:
            return
        points = [to_patch(pose.x, pose.y)] + [to_patch(p.x + 0.5, p.y + 0.5) for p in ahead]
        width = max(3, int(scale * 0.32))
        ribbon = _ROUTE.blended_with(_GROUND, 0.25).as_tuple()
        if len(points) > 1:
            pygame.draw.lines(patch, ribbon, False, points, width)
        for point in points:
            pygame.draw.circle(patch, ribbon, point, width // 2)
        goal = points[-1]
        pygame.draw.circle(patch, _ROUTE.as_tuple(), goal, width)
        pygame.draw.circle(patch, _WHITE.as_tuple(), goal, width, 2)

    def _draw_things(
        self,
        patch: pygame.Surface,
        city_map: CityMap,
        pose: VehiclePose,
        reach: float,
        to_patch: _Projector,
        scale: float,
    ) -> None:
        """Hospital, fires and victims near the vehicle."""
        size = max(6, int(scale))

        def rect_at(position: Position) -> pygame.Rect:
            left, top = to_patch(position.x, position.y)
            return pygame.Rect(left, top, size, size)

        def near(position: Position) -> bool:
            return max(abs(position.x + 0.5 - pose.x), abs(position.y + 0.5 - pose.y)) <= reach

        zone = city_map.safe_zone.position
        if near(zone):
            box = rect_at(zone).inflate(size // 2, size // 2)
            pygame.draw.rect(patch, _WHITE.as_tuple(), box, border_radius=4)
            glyphs.draw_hospital(patch, box.inflate(-4, -4), Color(214, 40, 48))
        fire_color = Color(255, 120, 40)
        for fire in city_map.fires:
            if near(fire.position):
                glow = rect_at(fire.position)
                pygame.draw.circle(
                    patch, fire_color.blended_with(_GROUND, 0.6).as_tuple(), glow.center, size
                )
                glyphs.draw_fire(
                    patch, glow, fire_color, fire_color.blended_with(_WHITE, 0.6), self._flicker
                )
        for victim in city_map.victims:
            if victim.status in (VictimStatus.TRAPPED, VictimStatus.LOST) and near(victim.position):
                self._draw_victim(patch, rect_at(victim.position), victim.status, size)

    def _draw_victim(
        self, patch: pygame.Surface, rect: pygame.Rect, status: VictimStatus, size: int
    ) -> None:
        color = Color(255, 92, 160) if status is VictimStatus.TRAPPED else _MUTED
        if status is VictimStatus.TRAPPED:
            pulse = 0.6 + 0.4 * math.sin(self._flicker * math.tau)
            pygame.draw.circle(patch, color.as_tuple(), rect.center, int(size * (0.8 + pulse)), 2)
        glyphs.draw_victim(patch, rect, color)

    @staticmethod
    def _draw_road_users(
        patch: pygame.Surface,
        pose: VehiclePose,
        reach: float,
        traffic: TrafficView,
        to_patch: _Projector,
        scale: float,
    ) -> None:
        """Cars as small bodies along their heading, people as dots; tracked ones outlined."""
        for agent in traffic.near(pose.x, pose.y, reach):
            x, y = traffic.where(agent)
            if max(abs(x - pose.x), abs(y - pose.y)) > reach:
                continue
            tracked = math.hypot(x - pose.x, y - pose.y) <= traffic.sensor_range
            centre = to_patch(x, y)
            outline = _draw_road_user(patch, agent, centre, scale)
            if tracked:
                pygame.draw.rect(patch, _TRACKED.as_tuple(), outline.inflate(6, 6), 2, 5)

    def _draw_traffic_tags(
        self,
        surface: pygame.Surface,
        pose: VehiclePose,
        traffic: TrafficView,
        anchor: tuple[int, int],
        scale: float,
        status: DriveStatus | None,
    ) -> None:
        """A distance tag on the few nearest tracked road users."""
        small, _ = self._font_pair()
        metres = status.metres_per_tile if status is not None else None
        tracked = sorted(
            (
                (math.hypot(x - pose.x, y - pose.y), agent, (x, y))
                for agent in traffic.near(pose.x, pose.y, traffic.sensor_range)
                for x, y in [traffic.where(agent)]
                if math.hypot(x - pose.x, y - pose.y) <= traffic.sensor_range
            ),
            key=lambda item: item[0],
        )
        placed: list[pygame.Rect] = []
        for distance, agent, (x, y) in tracked[:_TAGGED_ROAD_USERS]:
            sx, sy = _to_screen(x, y, pose, anchor, scale)
            name = agent.kind.label
            far = f"{distance * metres:.0f} m" if metres else f"{distance:.1f} tiles"
            text = small.render(f"{name} {far}", True, _GROUND.as_tuple())
            tag = text.get_rect().inflate(8, 2)
            tag.midleft = (sx + int(scale * 0.5), sy)
            while tag.collidelist(placed) != -1:  # stack tags that would overlap
                tag.top = placed[tag.collidelist(placed)].bottom + 2
            placed.append(tag)
            pygame.draw.rect(surface, _TRACKED.as_tuple(), tag, border_radius=4)
            surface.blit(text, text.get_rect(center=tag.center))

    def _draw_brake_alert(
        self, surface: pygame.Surface, area: pygame.Rect, brake: BrakeEvent
    ) -> None:
        """Red strip across the view: the brake just stopped the vehicle, and for whom."""
        _, large = self._font_pair()
        who = "PEDESTRIAN" if brake.kind is AgentKind.PEDESTRIAN else brake.kind.label
        text = large.render(f"EMERGENCY BRAKE · {who} AHEAD", True, _WHITE.as_tuple())
        strip = pygame.Rect(0, 0, text.get_width() + 40, text.get_height() + 16)
        strip.midtop = (area.centerx, area.top + 100)
        pygame.draw.rect(surface, (206, 36, 36), strip, border_radius=6)
        surface.blit(text, text.get_rect(center=strip.center))

    @staticmethod
    def _draw_sightings(
        patch: pygame.Surface, sightings: Sequence[Sighting], to_patch: _Projector, scale: float
    ) -> None:
        """Corner brackets round each group of tiles the onboard model detected something on."""
        for group in group_sightings(sightings):
            left, top = to_patch(group.left, group.top)
            right, bottom = to_patch(group.right + 1, group.bottom + 1)
            box = pygame.Rect(left, top, right - left, bottom - top).inflate(4, 4)
            arm = max(3, min(box.width, box.height) // 4)
            color = _sighting_color(group.kind).as_tuple()
            for cx, cy, sx, sy in (
                (box.left, box.top, 1, 1), (box.right, box.top, -1, 1),
                (box.left, box.bottom, 1, -1), (box.right, box.bottom, -1, -1),
            ):  # fmt: skip
                pygame.draw.line(patch, color, (cx, cy), (cx + sx * arm, cy), 2)
                pygame.draw.line(patch, color, (cx, cy), (cx, cy + sy * arm), 2)

    # ------------------------------------------------------------------
    # Screen-space overlays
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_vehicle(surface: pygame.Surface, anchor: tuple[int, int], scale: float) -> None:
        """The vehicle, always pointing up: a white car with a soft shadow."""
        length, width = scale * 0.95, scale * 0.55
        body = pygame.Rect(0, 0, int(width), int(length))
        body.center = anchor
        shadow = body.inflate(int(scale * 0.5), int(scale * 0.5))
        halo = pygame.Surface(shadow.size, pygame.SRCALPHA)
        pygame.draw.ellipse(halo, (*_ROUTE.as_tuple(), 70), halo.get_rect())
        surface.blit(halo, shadow)
        pygame.draw.rect(surface, (236, 238, 242), body, border_radius=max(3, body.width // 3))
        glass = pygame.Rect(0, 0, body.width - 6, max(3, body.height // 4))
        glass.midtop = (body.centerx, body.top + body.height // 5)
        pygame.draw.rect(surface, (40, 46, 58), glass, border_radius=3)

    def _draw_labels(
        self,
        surface: pygame.Surface,
        city_map: CityMap,
        pose: VehiclePose,
        sightings: Sequence[Sighting],
        anchor: tuple[int, int],
        scale: float,
    ) -> None:
        """One upright tag per detected group: what the model saw and how sure it was."""
        small, _ = self._font_pair()
        for group in group_sightings(sightings):
            centre_x = (group.left + group.right + 1) / 2
            centre_y = (group.top + group.bottom + 1) / 2
            x, y = _to_screen(centre_x, centre_y, pose, anchor, scale)
            text = small.render(
                f"{_LABELS.get(group.kind, group.kind.value.upper())} {group.confidence:.0%}",
                True,
                _WHITE.as_tuple(),
            )
            tag = text.get_rect().inflate(10, 4)
            half = max(group.right - group.left, group.bottom - group.top) + 1
            tag.midbottom = (x, y - int(scale * (half / 2 + 0.4)))
            pygame.draw.rect(surface, _sighting_color(group.kind).as_tuple(), tag, border_radius=4)
            surface.blit(text, text.get_rect(center=tag.center))

    def _draw_banner(
        self,
        surface: pygame.Surface,
        area: pygame.Rect,
        city_map: CityMap,
        route: Route | None,
        status: DriveStatus | None,
    ) -> None:
        """Top-left: autopilot state, where the vehicle is going and how far."""
        if status is None:
            return
        small, large = self._font_pair()
        left_tiles = len(remaining(route, city_map.vehicle.position))
        scale = status.metres_per_tile
        distance = _distance_text(left_tiles * scale) if scale else f"{left_tiles} tiles"
        lines = [
            (large, "AUTOPILOT", _ROUTE),
            (small, status.phase, _TEXT),
            (small, f"{status.goal} · {distance}" if left_tiles else status.goal, _MUTED),
        ]
        if status.speed_kmh is not None:
            pace = f" · replay {status.time_factor:.0f}x real time" if status.time_factor else ""
            lines.append((small, f"{status.speed_kmh:.0f} km/h{pace}", _TEXT))
        height = 16 + sum(font.get_height() + 2 for font, _, _ in lines)
        panel = pygame.Surface((300, height), pygame.SRCALPHA)
        panel.fill((10, 12, 16, 190))
        surface.blit(panel, (area.left + 10, area.top + 10))
        y = area.top + 16
        for font, text, color in lines:
            surface.blit(font.render(text, True, color.as_tuple()), (area.left + 20, y))
            y += font.get_height() + 2

    def _draw_minimap(
        self,
        surface: pygame.Surface,
        area: pygame.Rect,
        city_map: CityMap,
        pose: VehiclePose,
        route: Route | None,
    ) -> None:
        """The whole city in the top-right corner: streets, route, victims, the vehicle."""
        width = self._layout.minimap_px
        height = int(width * city_map.height / city_map.width)
        if self._minimap is None or self._minimap_age > 1.0:
            self._minimap = _street_plan(city_map, width, height)
            self._minimap_age = 0.0
        frame = pygame.Rect(area.right - width - 10, area.top + 10, width, height)
        surface.blit(self._minimap, frame)
        sx, sy = width / city_map.width, height / city_map.height

        def at(x: float, y: float) -> tuple[int, int]:
            return frame.left + int(x * sx), frame.top + int(y * sy)

        ahead = remaining(route, city_map.vehicle.position)
        if len(ahead) > 1:
            points = [at(p.x + 0.5, p.y + 0.5) for p in ahead]
            pygame.draw.lines(surface, _ROUTE.as_tuple(), False, points, 2)
        for victim in city_map.victims:
            if victim.status is VictimStatus.TRAPPED:
                pygame.draw.circle(
                    surface, (255, 92, 160), at(victim.position.x, victim.position.y), 3
                )
        for fire in city_map.fires:
            pygame.draw.circle(surface, (255, 120, 40), at(fire.position.x, fire.position.y), 2)
        pygame.draw.circle(surface, _WHITE.as_tuple(), at(pose.x, pose.y), 4)
        pygame.draw.rect(surface, _KERB.as_tuple(), frame, 1)

    def _font_pair(self) -> tuple[pygame.font.Font, pygame.font.Font]:
        if self._fonts is None:
            family = "segoeui,arial,dejavusans"
            size = self._layout.font_size_px
            self._fonts = (
                pygame.font.SysFont(family, size, bold=True),
                pygame.font.SysFont(family, size + 6, bold=True),
            )
        return self._fonts


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SightingGroup:
    """Touching sightings of one kind, as one object: its tile bounds and best confidence."""

    kind: EntityKind
    left: int
    top: int
    right: int
    bottom: int
    confidence: float


def group_sightings(sightings: Sequence[Sighting]) -> list[SightingGroup]:
    """Merge sightings of the same kind on touching tiles (8-neighbours) into groups.

    A fire box covers several tiles and yields one sighting per tile; drawn
    one by one they bury the view in labels. A driver's display shows one box
    per object, so this does too.
    """
    best: dict[tuple[EntityKind, int, int], float] = {}
    for s in sightings:
        key = (s.kind, s.position.x, s.position.y)
        best[key] = max(best.get(key, 0.0), s.confidence)
    groups: list[SightingGroup] = []
    seen: set[tuple[EntityKind, int, int]] = set()
    for start in sorted(best, key=lambda k: (k[0].value, k[1], k[2])):
        if start in seen:
            continue
        seen.add(start)
        members, frontier = [start], [start]
        while frontier:
            kind, x, y = frontier.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbour = (kind, x + dx, y + dy)
                    if neighbour in best and neighbour not in seen:
                        seen.add(neighbour)
                        members.append(neighbour)
                        frontier.append(neighbour)
        xs, ys = [m[1] for m in members], [m[2] for m in members]
        groups.append(
            SightingGroup(
                start[0], min(xs), min(ys), max(xs), max(ys), max(best[m] for m in members)
            )
        )
    return groups


#: Body length and width of each kind, in tiles.
_BODIES: dict[AgentKind, tuple[float, float]] = {
    AgentKind.CAR: (0.8, 0.46),
    AgentKind.AUTO_RICKSHAW: (0.6, 0.42),
    AgentKind.TWO_WHEELER: (0.55, 0.2),
    AgentKind.COW: (0.62, 0.3),
}


def _draw_road_user(
    patch: pygame.Surface, agent: TrafficAgent, centre: tuple[int, int], scale: float
) -> pygame.Rect:
    """Draw one road user at ``centre`` along its heading; return its outline."""
    if agent.kind is AgentKind.PEDESTRIAN:
        radius = max(3, int(scale * 0.16))
        pygame.draw.circle(patch, _PERSON.as_tuple(), centre, radius)
        box = pygame.Rect(0, 0, radius * 2, radius * 2)
        box.center = centre
        return box
    length, width = _BODIES[agent.kind]
    dx, _ = agent.heading.delta
    size = (int(scale * length), int(scale * width))
    body = pygame.Rect(0, 0, *(size if dx else size[::-1]))
    body.center = centre
    if agent.kind is AgentKind.COW:
        pygame.draw.ellipse(patch, _COW.as_tuple(), body)
        return body
    color = {AgentKind.CAR: _CAR, AgentKind.AUTO_RICKSHAW: _AUTO, AgentKind.TWO_WHEELER: _BIKE}
    radius = max(2, min(body.size) // 3)
    pygame.draw.rect(patch, color[agent.kind].as_tuple(), body, border_radius=radius)
    if agent.kind is AgentKind.AUTO_RICKSHAW:
        pygame.draw.rect(
            patch, _AUTO_TOP.as_tuple(), body.inflate(-body.w // 3, -body.h // 3), 0, 3
        )
    return body


def remaining(route: Route | None, position: Position) -> tuple[Position, ...]:
    """The waypoints still ahead of a vehicle at ``position`` (empty without a route)."""
    if route is None or route.is_empty:
        return ()
    waypoints = route.waypoints
    if position in waypoints:
        return waypoints[waypoints.index(position) + 1 :]
    nearest = min(
        range(len(waypoints)),
        key=lambda i: abs(waypoints[i].x - position.x) + abs(waypoints[i].y - position.y),
    )
    return waypoints[nearest:]


def _reach(area: pygame.Rect, anchor: tuple[int, int]) -> float:
    """Pixels from the anchor to the farthest corner of ``area``."""
    return max(
        math.hypot(cx - anchor[0], cy - anchor[1])
        for cx in (area.left, area.right)
        for cy in (area.top, area.bottom)
    )


def _tiles_within(city_map: CityMap, pose: VehiclePose, reach: float) -> list[Position]:
    x0, x1 = max(0, int(pose.x - reach)), min(city_map.width - 1, int(pose.x + reach))
    y0, y1 = max(0, int(pose.y - reach)), min(city_map.height - 1, int(pose.y + reach))
    return [Position(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def _is_road(city_map: CityMap, x: int, y: int) -> bool:
    if not (0 <= x < city_map.width and 0 <= y < city_map.height):
        return False
    return city_map.tile_at(Position(x, y)) in _ROADS


def _to_screen(
    x: float, y: float, pose: VehiclePose, anchor: tuple[int, int], scale: float
) -> tuple[int, int]:
    """A map point's place on screen, with the vehicle at ``anchor`` pointing up."""
    dx, dy = (x - pose.x) * scale, (y - pose.y) * scale
    cos, sin = math.cos(-pose.angle), math.sin(-pose.angle)
    return round(anchor[0] + dx * cos - dy * sin), round(anchor[1] + dx * sin + dy * cos)


def _sighting_color(kind: EntityKind) -> Color:
    return {
        EntityKind.VICTIM: Color(255, 120, 190),
        EntityKind.FIRE: Color(255, 150, 60),
    }.get(kind, Color(255, 214, 90))


def _distance_text(metres: float) -> str:
    return f"{metres / 1000:.1f} km" if metres >= 1000 else f"{metres:.0f} m"


def _street_plan(city_map: CityMap, width: int, height: int) -> pygame.Surface:
    """The whole map at minimap size: roads light, everything else dark."""
    plan = pygame.Surface((city_map.width, city_map.height))
    plan.fill(_BLOCK.as_tuple())
    road = _KERB.as_tuple()
    damage = _DAMAGE.as_tuple()
    for y in range(city_map.height):
        for x in range(city_map.width):
            terrain = city_map.tile_at(Position(x, y))
            if terrain in _DAMAGED:
                plan.set_at((x, y), damage)
            elif terrain in _ROADS:
                plan.set_at((x, y), road)
    return pygame.transform.scale(plan, (width, height))
