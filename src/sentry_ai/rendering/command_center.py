"""The SENTRY command center: one screen for a whole mission (Phase 9 dashboard).

City-sized maps open in this layout. It answers, at a glance, the questions a
reviewer asks of an autonomous vehicle — where is it, what does it see, what
did the AI detect, what is it planning, what did the policy choose, is it
safe, and how is the mission going::

    ┌──────────────────────────── header: mission · time · distance · status ───┐
    │ LIVE VEHICLE VIEW (drive view, speedometer)     │ REAL-WORLD MAP · routes  │
    │                                                 │ ROUTE COMPARISON A/B/C   │
    │                                                 │ ENVIRONMENT · HAZARDS    │
    ├──────────── FRONT CAMERA ─ PERCEPTION ─ RL + CONTROLS ─┬── MISSION LOG ───┤

Every number comes from the running simulation or a trained model:

* the front camera is a real Mapillary photo of the street, with a stock
  COCO YOLOv8 finding the real vehicles and people in it (display only);
* perception is the vehicle's simulated onboard frame with what the
  mission's own YOLO reported;
* the route comparison is three real A* plans scored on traffic and fire
  risk (:mod:`sentry_ai.navigation.route_advisor`);
* the RL chart is the traffic-trained DQN's own held-out evaluation log,
  and the controls are its live action scores.

Drawing only: the dashboard reads state, it never changes it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pygame

from sentry_ai.common.color import Color
from sentry_ai.decision.emergency_brake import EmergencyBrake
from sentry_ai.decision.fused_controller import FusedLocalController
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind, TerrainType, VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.traffic import AgentKind
from sentry_ai.interfaces.navigation import LOCAL_ACTION_ORDER, LocalAction
from sentry_ai.navigation.route_advisor import RouteOption
from sentry_ai.perception.scene_evidence import OnboardEvidenceSource
from sentry_ai.perception.street_detector import StreetPhotoDetector
from sentry_ai.rendering.drive_view import (
    DriveStatus,
    DriveViewRenderer,
    TrafficView,
    remaining,
)
from sentry_ai.rendering.motion import VehiclePose
from sentry_ai.rendering.street_view import StreetPhotoLibrary
from sentry_ai.sensors.frame import CameraFrame
from sentry_ai.simulation.mission import MissionController, MissionPhase

#: Window size of the command-center layout.
SIZE = (1600, 900)

_BG = Color(8, 13, 20)
_PANEL = Color(13, 21, 31)
_EDGE = Color(34, 52, 70)
_TEXT = Color(226, 234, 242)
_MUTED = Color(128, 146, 166)
_CYAN = Color(80, 200, 255)
_GREEN = Color(64, 220, 130)
_AMBER = Color(255, 186, 60)
_RED = Color(236, 64, 64)
_BLUE = Color(70, 140, 255)

_HEADER_H = 64
_MAIN = pygame.Rect(0, _HEADER_H, 1040, 560)
_RIGHT_X = 1040
_BOTTOM_Y = _HEADER_H + 560

#: The environment panel's rows: label and the kind of road user it counts.
_ENVIRONMENT_ROWS = (
    ("Cars", AgentKind.CAR),
    ("Autos", AgentKind.AUTO_RICKSHAW),
    ("Bikes", AgentKind.TWO_WHEELER),
    ("People", AgentKind.PEDESTRIAN),
    ("Cattle", AgentKind.COW),
)

#: Road users within this many tiles count toward the environment panel.
_ENVIRONMENT_RADIUS = 15


@dataclass
class CommandCenterState:
    """Everything the dashboard shows, gathered by the window each frame."""

    city_name: str
    city_map: CityMap
    mission: MissionController
    pose: VehiclePose
    metres_per_tile: float
    speed_kmh: float
    time_factor: float
    real_seconds: float
    #: Real-world seconds per simulated second, for the log's timestamps.
    real_per_sim_second: float
    brain: FusedLocalController | None
    brake: EmergencyBrake | None
    camera: OnboardEvidenceSource | None
    onboard_frame: CameraFrame | None
    traffic: TrafficView | None
    routes: Sequence[RouteOption]
    collisions: int
    satellite: pygame.Surface | None
    street: StreetPhotoLibrary | None
    rl_curve: Sequence[tuple[int, float, int]]


class CommandCenterRenderer:
    """Draws :class:`CommandCenterState` as the full-window dashboard."""

    def __init__(
        self, drive: DriveViewRenderer, street_detector: StreetPhotoDetector | None = None
    ) -> None:
        """Create the renderer. ``pygame.font`` must be initialised first."""
        self._drive = drive
        self._detector = street_detector
        family = "segoeui,arial,dejavusans"
        self._small = pygame.font.SysFont(family, 13)
        self._text = pygame.font.SysFont(family, 15, bold=True)
        self._big = pygame.font.SysFont(family, 26, bold=True)
        self._title = pygame.font.SysFont(family, 34, bold=True)
        self._map_cache: tuple[int, pygame.Surface] | None = None

    def draw(self, surface: pygame.Surface, state: CommandCenterState) -> None:
        """Draw the whole dashboard onto ``surface`` (see :data:`SIZE`)."""
        surface.fill(_BG.as_tuple())
        self._header(surface, state)
        self._drive.draw(
            surface,
            _MAIN,
            state.city_map,
            state.pose,
            route=state.mission.route,
            sightings=state.camera.last_sightings if state.camera is not None else (),
            background=state.satellite,
            status=DriveStatus("", "", state.metres_per_tile),
            traffic=state.traffic,
            overlays=False,
        )
        self._speedometer(surface, state)
        y = _HEADER_H
        y = self._map_panel(surface, pygame.Rect(_RIGHT_X, y, SIZE[0] - _RIGHT_X, 300), state)
        y = self._routes_panel(surface, pygame.Rect(_RIGHT_X, y, SIZE[0] - _RIGHT_X, 132), state)
        self._environment_panel(surface, pygame.Rect(_RIGHT_X, y, SIZE[0] - _RIGHT_X, 128), state)
        bottom = SIZE[1] - _BOTTOM_Y
        self._front_camera(surface, pygame.Rect(0, _BOTTOM_Y, 350, bottom), state)
        self._perception(surface, pygame.Rect(350, _BOTTOM_Y, 290, bottom), state)
        self._learning(surface, pygame.Rect(640, _BOTTOM_Y, 400, bottom), state)
        self._log(surface, pygame.Rect(_RIGHT_X, _BOTTOM_Y, SIZE[0] - _RIGHT_X, bottom), state)

    # ------------------------------------------------------------------
    # Header and speedometer
    # ------------------------------------------------------------------

    def _header(self, surface: pygame.Surface, state: CommandCenterState) -> None:
        pygame.draw.rect(surface, _PANEL.as_tuple(), pygame.Rect(0, 0, SIZE[0], _HEADER_H))
        badge = pygame.Rect(12, 10, 44, 44)
        pygame.draw.rect(surface, _RED.as_tuple(), badge, border_radius=9)
        pygame.draw.rect(surface, (255, 255, 255), badge.inflate(-30, -12))
        pygame.draw.rect(surface, (255, 255, 255), badge.inflate(-12, -30))
        surface.blit(self._title.render("SENTRY AI", True, _TEXT.as_tuple()), (66, 4))
        subtitle = "AUTONOMOUS EMERGENCY RESCUE VEHICLE  ·  RL-DRIVEN SAFE NAVIGATION"
        surface.blit(self._small.render(subtitle, True, _MUTED.as_tuple()), (68, 42))
        mission = state.mission
        goal = _goal_text(state)
        self._card(surface, pygame.Rect(560, 8, 440, 48), "MISSION", goal)
        minutes, seconds = divmod(int(state.real_seconds), 60)
        self._card(
            surface, pygame.Rect(1010, 8, 170, 48), "TIME ELAPSED", f"{minutes:02d}:{seconds:02d}"
        )
        left = len(remaining(mission.route, state.city_map.vehicle.position))
        km = left * state.metres_per_tile / 1000
        self._card(surface, pygame.Rect(1190, 8, 180, 48), "DISTANCE TO GOAL", f"{km:.1f} km")
        phase = mission.phase.value.replace("_", " ").upper()
        colour = _GREEN if mission.phase is MissionPhase.COMPLETED else _CYAN
        self._card(surface, pygame.Rect(1380, 8, 210, 48), "STATUS", phase, colour)

    def _card(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        value: str,
        colour: Color = _TEXT,
    ) -> None:
        pygame.draw.rect(surface, _BG.as_tuple(), rect, border_radius=6)
        pygame.draw.rect(surface, _EDGE.as_tuple(), rect, 1, 6)
        surface.blit(self._small.render(label, True, _MUTED.as_tuple()), (rect.x + 10, rect.y + 4))
        text = _fit(self._text, value, rect.width - 20)
        surface.blit(self._text.render(text, True, colour.as_tuple()), (rect.x + 10, rect.y + 23))

    def _speedometer(self, surface: pygame.Surface, state: CommandCenterState) -> None:
        centre = (_MAIN.left + 80, _MAIN.bottom - 70)
        pygame.draw.circle(surface, (*_BG.as_tuple(),), centre, 62)
        pygame.draw.circle(surface, _EDGE.as_tuple(), centre, 62, 2)
        fraction = min(1.0, state.speed_kmh / 80.0)
        start = math.radians(220)
        for i in range(40):
            angle = start - math.radians(260) * i / 39
            colour = _CYAN if i / 39 <= fraction else _EDGE
            point = (centre[0] + math.cos(angle) * 54, centre[1] - math.sin(angle) * 54)
            pygame.draw.circle(surface, colour.as_tuple(), point, 3)
        speed = self._big.render(f"{state.speed_kmh:.0f}", True, _TEXT.as_tuple())
        surface.blit(speed, speed.get_rect(center=(centre[0], centre[1] - 6)))
        unit = self._small.render("km/h", True, _MUTED.as_tuple())
        surface.blit(unit, unit.get_rect(center=(centre[0], centre[1] + 18)))
        driving = not state.mission.phase.is_terminal
        backdrop = pygame.Rect(centre[0] + 70, centre[1] - 30, 190, 62)
        panel = pygame.Surface(backdrop.size, pygame.SRCALPHA)
        panel.fill((8, 13, 20, 200))
        surface.blit(panel, backdrop)
        pace = f"replay {state.time_factor:.0f}x real time"
        surface.blit(
            self._small.render(pace, True, _MUTED.as_tuple()), (backdrop.x + 8, backdrop.y + 6)
        )
        x = backdrop.x + 8
        for gear in "PRND":
            live = gear == ("D" if driving else "P")
            colour = _GREEN if live else _MUTED
            surface.blit(self._text.render(gear, True, colour.as_tuple()), (x, backdrop.y + 30))
            x += 22

    # ------------------------------------------------------------------
    # Right column
    # ------------------------------------------------------------------

    def _panel(self, surface: pygame.Surface, rect: pygame.Rect, title: str) -> pygame.Rect:
        """Draw a panel frame and title; return the content area."""
        pygame.draw.rect(surface, _PANEL.as_tuple(), rect)
        pygame.draw.rect(surface, _EDGE.as_tuple(), rect, 1)
        surface.blit(self._text.render(title, True, _TEXT.as_tuple()), (rect.x + 10, rect.y + 6))
        return pygame.Rect(rect.x + 8, rect.y + 28, rect.width - 16, rect.height - 34)

    def _map_panel(
        self, surface: pygame.Surface, rect: pygame.Rect, state: CommandCenterState
    ) -> int:
        area = self._panel(surface, rect, f"REAL-WORLD MAP  ({state.city_name})")
        city = state.city_map
        points = [city.vehicle.position, city.safe_zone.position]
        points += [v.position for v in city.victims if v.status is VictimStatus.TRAPPED]
        for option in state.routes:
            points += list(option.route.waypoints)
        x0, x1 = min(p.x for p in points) - 6, max(p.x for p in points) + 6
        y0, y1 = min(p.y for p in points) - 6, max(p.y for p in points) + 6
        span = max(x1 - x0, (y1 - y0) * area.width / area.height, 12)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_w, half_h = span / 2, span / 2 * area.height / area.width
        scale = area.width / span

        def at(x: float, y: float) -> tuple[int, int]:
            return (
                round(area.x + (x - (cx - half_w)) * scale),
                round(area.y + (y - (cy - half_h)) * scale),
            )

        clip = surface.get_clip()
        surface.set_clip(area)
        self._map_ground(surface, area, state, (cx - half_w, cy - half_h, span))
        selected = state.mission.route
        for option in state.routes:
            if option.route.waypoints != selected.waypoints:
                _dashed(surface, [at(p.x + 0.5, p.y + 0.5) for p in option.route], _BLUE)
        if len(selected) > 1:
            pts = [at(p.x + 0.5, p.y + 0.5) for p in selected]
            pygame.draw.lines(surface, _GREEN.as_tuple(), False, pts, 4)
        for fire in city.fires:
            pygame.draw.circle(
                surface, _AMBER.as_tuple(), at(fire.position.x + 0.5, fire.position.y + 0.5), 4
            )
        for victim in city.victims:
            if victim.status is VictimStatus.TRAPPED:
                pin = at(victim.position.x + 0.5, victim.position.y + 0.5)
                pygame.draw.circle(surface, _RED.as_tuple(), pin, 6)
                pygame.draw.circle(surface, (255, 255, 255), pin, 6, 2)
        home = at(city.safe_zone.position.x + 0.5, city.safe_zone.position.y + 0.5)
        pygame.draw.rect(surface, (255, 255, 255), pygame.Rect(home[0] - 6, home[1] - 6, 12, 12))
        pygame.draw.rect(surface, _RED.as_tuple(), pygame.Rect(home[0] - 1, home[1] - 5, 3, 10))
        pygame.draw.rect(surface, _RED.as_tuple(), pygame.Rect(home[0] - 5, home[1] - 1, 10, 3))
        car = at(state.pose.x, state.pose.y)
        pygame.draw.circle(surface, _CYAN.as_tuple(), car, 7)
        pygame.draw.circle(surface, (255, 255, 255), car, 7, 2)
        surface.set_clip(clip)
        self._legend(surface, area)
        return rect.bottom

    def _map_ground(
        self,
        surface: pygame.Surface,
        area: pygame.Rect,
        state: CommandCenterState,
        window: tuple[float, float, float],
    ) -> None:
        """Satellite (dimmed) or the street plan, for the map window in tiles."""
        left, top, span = window
        if state.satellite is None:
            pygame.draw.rect(surface, _BG.as_tuple(), area)
            return
        per_tile = state.satellite.get_width() / state.city_map.width
        source = pygame.Rect(
            round(left * per_tile),
            round(top * per_tile),
            round(span * per_tile),
            round(span * per_tile * area.height / area.width),
        )
        clipped = source.clip(state.satellite.get_rect())
        pygame.draw.rect(surface, _BG.as_tuple(), area)
        if clipped.width > 0 and clipped.height > 0:
            factor = area.width / source.width
            image = pygame.transform.smoothscale(
                state.satellite.subsurface(clipped),
                (max(1, round(clipped.width * factor)), max(1, round(clipped.height * factor))),
            )
            surface.blit(
                image,
                (
                    area.x + round((clipped.x - source.x) * factor),
                    area.y + round((clipped.y - source.y) * factor),
                ),
            )
        shade = pygame.Surface(area.size, pygame.SRCALPHA)
        shade.fill((8, 13, 20, 90))
        surface.blit(shade, area)

    def _legend(self, surface: pygame.Surface, area: pygame.Rect) -> None:
        box = pygame.Rect(area.right - 150, area.y + 4, 146, 66)
        panel = pygame.Surface(box.size, pygame.SRCALPHA)
        panel.fill((8, 13, 20, 200))
        surface.blit(panel, box)
        rows = (
            (_GREEN, "Selected route", False),
            (_BLUE, "Alternative route", True),
            (_RED, "Victim", False),
            (_AMBER, "Fire", False),
        )
        for i, (colour, text, dashed) in enumerate(rows):
            y = box.y + 8 + i * 15
            if dashed:
                _dashed(surface, [(box.x + 8, y), (box.x + 30, y)], colour)
            else:
                pygame.draw.line(surface, colour.as_tuple(), (box.x + 8, y), (box.x + 30, y), 3)
            surface.blit(self._small.render(text, True, _TEXT.as_tuple()), (box.x + 38, y - 8))

    def _routes_panel(
        self, surface: pygame.Surface, rect: pygame.Rect, state: CommandCenterState
    ) -> int:
        area = self._panel(surface, rect, "ROUTE COMPARISON  (AI evaluated)")
        if not state.routes:
            surface.blit(
                self._small.render("no route to evaluate", True, _MUTED.as_tuple()), area.topleft
            )
            return rect.bottom
        selected = state.mission.route.waypoints
        levels = {"Low": _GREEN, "Moderate": _AMBER, "Heavy": _RED}
        risks = {"Safe": _GREEN, "Moderate": _AMBER, "High": _RED}
        for i, option in enumerate(state.routes[:3]):
            row = pygame.Rect(area.x, area.y + i * 32, area.width, 28)
            chosen = option.route.waypoints == selected
            if chosen:
                pygame.draw.rect(surface, _GREEN.as_tuple(), row, 2, 5)
            cells = (
                (f"Route {option.name}", _TEXT, 0),
                (f"{option.length_m / 1000:.1f} km", _MUTED, 90),
                (f"{option.traffic_level} traffic", levels[option.traffic_level], 170),
                (
                    option.risk_level if option.risk_level != "Safe" else "Safe",
                    risks[option.risk_level],
                    310,
                ),
                (f"{option.eta_minutes:.1f} min", _TEXT, 400),
            )
            for text, colour, dx in cells:
                surface.blit(
                    self._text.render(text, True, colour.as_tuple()), (row.x + 10 + dx, row.y + 5)
                )
            if chosen:
                mark = self._text.render(">", True, _GREEN.as_tuple())
                surface.blit(mark, (row.right - 18, row.y + 5))
        return rect.bottom

    def _environment_panel(
        self, surface: pygame.Surface, rect: pygame.Rect, state: CommandCenterState
    ) -> None:
        half = rect.width // 2
        left = self._panel(surface, pygame.Rect(rect.x, rect.y, half, rect.height), "ENVIRONMENT")
        counts = _nearby_counts(state)
        for i, (label, kind) in enumerate(_ENVIRONMENT_ROWS):
            x = left.x + (i % 3) * 88
            y = left.y + (i // 3) * 44
            surface.blit(self._small.render(label, True, _MUTED.as_tuple()), (x, y))
            surface.blit(
                self._text.render(str(counts.get(kind, 0)), True, _TEXT.as_tuple()), (x, y + 16)
            )
        right = self._panel(
            surface, pygame.Rect(rect.x + half, rect.y, rect.width - half, rect.height), "HAZARDS"
        )
        city = state.city_map
        debris = sum(1 for o in city.obstacles if o.kind is not TerrainType.TREE)
        values = (
            ("Fire", len(city.fires), _AMBER),
            ("Debris", debris, _RED),
            ("Collisions", state.collisions, _RED if state.collisions else _GREEN),
        )
        for i, (label, value, colour) in enumerate(values):
            x = right.x + i * 88
            surface.blit(self._small.render(label, True, _MUTED.as_tuple()), (x, right.y))
            surface.blit(self._big.render(str(value), True, colour.as_tuple()), (x, right.y + 16))
        brake = state.brake
        if brake is not None:
            text = f"brake {'ON' if brake.enabled else 'OFF'} · stops {sum(brake.brakes.values())}"
            colour = _GREEN if brake.enabled else _RED
            surface.blit(self._small.render(text, True, colour.as_tuple()), (right.x, right.y + 58))

    # ------------------------------------------------------------------
    # Bottom row
    # ------------------------------------------------------------------

    def _front_camera(
        self, surface: pygame.Surface, rect: pygame.Rect, state: CommandCenterState
    ) -> None:
        area = self._panel(surface, rect, "FRONT CAMERA  (real street photo)")
        library = state.street
        found = library.lookup(state.city_map.vehicle) if library is not None else None
        if library is None or found is None:
            self._frame(surface, area, state.onboard_frame)
            note = "no street photo here · simulated frame"
            surface.blit(
                self._small.render(note, True, _MUTED.as_tuple()), (area.x, area.bottom - 16)
            )
            return
        image_id, metres = found
        photo = library.image(image_id, area.size)
        surface.blit(photo, area)
        if self._detector is not None:
            raw = pygame.image.load(str(library.path(image_id)))
            sx, sy = area.width / raw.get_width(), area.height / raw.get_height()
            for detection in self._detector.detect(library.path(image_id)):
                x0, y0, x1, y1 = detection.box
                box = pygame.Rect(
                    area.x + x0 * sx, area.y + y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy
                )
                colour = _box_colour(detection.label)
                pygame.draw.rect(surface, colour.as_tuple(), box, 2)
                tag = self._small.render(
                    f"{detection.label} {detection.confidence:.2f}", True, (0, 0, 0)
                )
                back = tag.get_rect(bottomleft=(box.x, box.y))
                pygame.draw.rect(surface, colour.as_tuple(), back.inflate(4, 0))
                surface.blit(tag, back)
        where = "at the vehicle" if metres == 0 else f"{metres:.0f} m away"
        credit = f"YOLOv8 (COCO) · {where} · © Mapillary, CC BY-SA"
        strip = pygame.Surface((area.width, 16), pygame.SRCALPHA)
        strip.fill((0, 0, 0, 170))
        surface.blit(strip, (area.x, area.bottom - 16))
        surface.blit(
            self._small.render(credit, True, _TEXT.as_tuple()), (area.x + 4, area.bottom - 16)
        )

    def _perception(
        self, surface: pygame.Surface, rect: pygame.Rect, state: CommandCenterState
    ) -> None:
        area = self._panel(surface, rect, "PERCEPTION  (YOLOv8, onboard)")
        side = min(area.width, area.height - 18)
        frame_rect = pygame.Rect(area.x + (area.width - side) // 2, area.y, side, side)
        self._frame(surface, frame_rect, state.onboard_frame)
        frame = state.onboard_frame
        camera = state.camera
        if frame is not None and camera is not None:
            view = frame.view
            per = side / max(1, view.width_tiles)
            for sighting in camera.last_sightings:
                box = pygame.Rect(
                    frame_rect.x + (sighting.position.x - view.origin.x) * per,
                    frame_rect.y + (sighting.position.y - view.origin.y) * per,
                    per,
                    per,
                )
                colour = {EntityKind.VICTIM: _RED, EntityKind.FIRE: _AMBER}.get(
                    sighting.kind, _BLUE
                )
                pygame.draw.rect(surface, colour.as_tuple(), box, 2)
        legend = "red victim · amber fire · blue debris"
        surface.blit(
            self._small.render(legend, True, _MUTED.as_tuple()), (area.x, area.bottom - 14)
        )

    def _frame(self, surface: pygame.Surface, rect: pygame.Rect, frame: CameraFrame | None) -> None:
        if frame is None:
            pygame.draw.rect(surface, _BG.as_tuple(), rect)
            return
        image = pygame.surfarray.make_surface(frame.pixels.swapaxes(0, 1))
        surface.blit(pygame.transform.scale(image, rect.size), rect)

    def _learning(
        self, surface: pygame.Surface, rect: pygame.Rect, state: CommandCenterState
    ) -> None:
        area = self._panel(surface, rect, "REINFORCEMENT LEARNING  (DQN)")
        chart = pygame.Rect(area.x, area.y + 14, area.width, 96)
        self._chart(surface, chart, state.rl_curve)
        controls = pygame.Rect(
            area.x, chart.bottom + 12, area.width, area.bottom - chart.bottom - 12
        )
        self._controls(surface, controls, state)

    def _chart(
        self, surface: pygame.Surface, rect: pygame.Rect, curve: Sequence[tuple[int, float, int]]
    ) -> None:
        pygame.draw.rect(surface, _BG.as_tuple(), rect)
        key = "rescues vs rule-based driver (green) · collisions (red) · held-out missions"
        surface.blit(self._small.render(key, True, _MUTED.as_tuple()), (rect.x, rect.y - 15))
        if len(curve) < 2:
            return
        steps = [c[0] for c in curve]
        top_ratio = max(1.5, max(c[1] for c in curve))
        top_hits = max(1, max(c[2] for c in curve))

        def point(step: int, value: float, top: float) -> tuple[int, int]:
            x = rect.x + 6 + (step - steps[0]) / max(1, steps[-1] - steps[0]) * (rect.width - 12)
            return round(x), round(rect.bottom - 6 - value / top * (rect.height - 12))

        pygame.draw.lines(
            surface, _GREEN.as_tuple(), False, [point(s, r, top_ratio) for s, r, _ in curve], 2
        )
        pygame.draw.lines(
            surface, _RED.as_tuple(), False, [point(s, h, top_hits) for s, _, h in curve], 2
        )
        last = curve[-1]
        text = f"{last[0] // 1000}k steps · {last[1]:.0%} rescues · {last[2]} collisions"
        surface.blit(self._small.render(text, True, _TEXT.as_tuple()), (rect.x + 6, rect.y + 4))

    def _controls(
        self, surface: pygame.Surface, rect: pygame.Rect, state: CommandCenterState
    ) -> None:
        surface.blit(
            self._text.render("VEHICLE CONTROLS  (RL output)", True, _TEXT.as_tuple()), rect.topleft
        )
        trace = state.brain.last if state.brain is not None else None
        scores = {a: 0.0 for a in LOCAL_ACTION_ORDER}
        if trace is not None:
            values = [trace.local_decision.q_values.get(a, 0.0) for a in LOCAL_ACTION_ORDER]
            peak = max(values)
            weights = [math.exp((v - peak) * 3) for v in values]
            total = sum(weights)
            scores = {a: w / total for a, w in zip(LOCAL_ACTION_ORDER, weights, strict=True)}
        braking = state.brake is not None and state.brake.last is not None
        steering = scores[LocalAction.TURN_RIGHT] - scores[LocalAction.TURN_LEFT]
        throttle = scores[LocalAction.MOVE_FORWARD]
        stop = 1.0 if braking else scores[LocalAction.STOP]
        rows = (
            (
                "Steering",
                abs(steering),
                f"{'R' if steering > 0 else 'L'} {abs(steering):.2f}",
                _GREEN,
            ),
            ("Throttle", throttle, f"{throttle:.2f}", _BLUE),
            ("Brake", stop, "EMERGENCY" if braking else f"{stop:.2f}", _RED),
        )
        for i, (label, share, value, colour) in enumerate(rows):
            y = rect.y + 22 + i * 20
            surface.blit(self._small.render(label, True, _MUTED.as_tuple()), (rect.x, y))
            track = pygame.Rect(rect.x + 70, y + 4, rect.width - 170, 9)
            pygame.draw.rect(surface, _EDGE.as_tuple(), track, border_radius=4)
            filled = track.copy()
            filled.width = max(2, round(track.width * min(1.0, share)))
            pygame.draw.rect(surface, colour.as_tuple(), filled, border_radius=4)
            surface.blit(self._small.render(value, True, _TEXT.as_tuple()), (track.right + 8, y))

    def _log(self, surface: pygame.Surface, rect: pygame.Rect, state: CommandCenterState) -> None:
        split = rect.width - 200
        area = self._panel(surface, pygame.Rect(rect.x, rect.y, split, rect.height), "MISSION LOG")
        events = state.mission.events.recent(12)
        for i, event in enumerate(events[-12:]):
            minutes, seconds = divmod(int(event.at_seconds * state.real_per_sim_second), 60)
            line = _fit(self._small, f"{minutes:02d}:{seconds:02d}  {event.message}", area.width)
            surface.blit(
                self._small.render(line, True, _TEXT.as_tuple()), (area.x, area.y + i * 17)
            )
        goals = self._panel(
            surface, pygame.Rect(rect.x + split, rect.y, 200, rect.height), "OBJECTIVES"
        )
        city = state.city_map
        rescued = sum(1 for v in city.victims if v.status is VictimStatus.RESCUED)
        onboard = len(city.vehicle.onboard_victims)
        total = len(city.victims)
        done = state.mission.phase is MissionPhase.COMPLETED
        items = (
            ("Navigate to victims", rescued + onboard > 0),
            (f"Rescue victims {rescued}/{total}", rescued == total),
            ("Return to safe zone", done),
            ("Harm no one", state.brake is None or not sum(state.brake.hits.values())),
        )
        for i, (text, ticked) in enumerate(items):
            y = goals.y + i * 26
            box = pygame.Rect(goals.x, y, 16, 16)
            pygame.draw.rect(surface, (_GREEN if ticked else _MUTED).as_tuple(), box, 2, 3)
            if ticked:
                pygame.draw.line(
                    surface, _GREEN.as_tuple(), (box.x + 3, box.y + 8), (box.x + 7, box.y + 12), 2
                )
                pygame.draw.line(
                    surface, _GREEN.as_tuple(), (box.x + 7, box.y + 12), (box.x + 13, box.y + 3), 2
                )
            surface.blit(self._small.render(text, True, _TEXT.as_tuple()), (box.right + 8, y))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _goal_text(state: CommandCenterState) -> str:
    mission = state.mission
    goal = mission.route.goal
    if mission.phase is MissionPhase.COMPLETED:
        return "Mission complete: every reachable victim delivered"
    for victim in state.city_map.victims:
        if goal is not None and victim.position == goal:
            return f"Reach and rescue {victim.victim_id} safely through real traffic"
    if goal is not None and goal == state.city_map.safe_zone.position:
        return "Return victims to the hospital safely"
    return "Planning the safest route"


def _nearby_counts(state: CommandCenterState) -> dict[AgentKind, int]:
    counts: dict[AgentKind, int] = {}
    if state.traffic is None:
        return counts
    vehicle: Position = state.city_map.vehicle.position
    for agent in state.traffic.near(vehicle.x + 0.5, vehicle.y + 0.5, _ENVIRONMENT_RADIUS):
        counts[agent.kind] = counts.get(agent.kind, 0) + 1
    return counts


def _box_colour(label: str) -> Color:
    return {
        "Pedestrian": _BLUE,
        "Bike": _AMBER,
        "Cycle": _AMBER,
        "Bus": _RED,
        "Truck": _RED,
        "Auto/Truck": _AMBER,
    }.get(label, _GREEN)


def _dashed(surface: pygame.Surface, points: list[tuple[int, int]], colour: Color) -> None:
    for i, (a, b) in enumerate(zip(points, points[1:], strict=False)):
        if i % 2 == 0:
            pygame.draw.line(surface, colour.as_tuple(), a, b, 3)


def _fit(font: pygame.font.Font, text: str, width: int) -> str:
    if font.size(text)[0] <= width:
        return text
    while text and font.size(f"{text}…")[0] > width:
        text = text[:-1]
    return f"{text}…"
