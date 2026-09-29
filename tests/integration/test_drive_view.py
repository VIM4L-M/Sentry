"""Integration tests for the follow-camera drive view (sentry_ai.rendering.drive_view)."""

from __future__ import annotations

import math
from pathlib import Path

import pygame
import pytest

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.navigation import Route
from sentry_ai.perception.scene_evidence import Sighting
from sentry_ai.rendering.drive_view import (
    DriveStatus,
    DriveViewRenderer,
    _to_screen,
    group_sightings,
    remaining,
)
from sentry_ai.rendering.motion import VehiclePose, heading_angle


@pytest.fixture
def city(project_root: Path) -> CityMap:
    loader = ConfigLoader(project_root=project_root)
    return CityMap.from_config(loader.load_yaml("configs/maps/city_default.yaml"))


def _pose(city: CityMap) -> VehiclePose:
    vehicle = city.vehicle
    return VehiclePose(vehicle.position.x + 0.5, vehicle.position.y + 0.5, heading_angle(vehicle))


def test_draws_inside_its_area_only(city: CityMap) -> None:
    pygame.init()
    surface = pygame.Surface((1000, 700))
    surface.fill((255, 0, 255))
    area = pygame.Rect(50, 30, 900, 640)
    start = city.vehicle.position
    route = Route(waypoints=(start, Position(start.x + 1, start.y)), cost=1.0)
    sighting = Sighting(EntityKind.FIRE, Position(start.x + 2, start.y), 0.9)
    DriveViewRenderer().draw(
        surface,
        area,
        city,
        _pose(city),
        route=route,
        sightings=(sighting,),
        status=DriveStatus("EN ROUTE TO VICTIM", "to victim_01", 20.0),
    )
    outside = [surface.get_at(p)[:3] for p in ((5, 5), (990, 690), (20, 300))]
    assert all(color == (255, 0, 255) for color in outside)
    assert surface.get_at(area.center)[:3] != (255, 0, 255)


def test_draws_with_imagery(city: CityMap) -> None:
    pygame.init()
    imagery = pygame.Surface((city.width * 8, city.height * 8))
    imagery.fill((10, 120, 10))
    surface = pygame.Surface((640, 480))
    DriveViewRenderer().draw(surface, surface.get_rect(), city, _pose(city), background=imagery)


def test_the_heading_points_up_the_screen() -> None:
    anchor = (400, 300)
    for angle, ahead in ((0.0, (0.0, -1.0)), (math.pi / 2, (1.0, 0.0)), (math.pi, (0.0, 1.0))):
        pose = VehiclePose(10.0, 10.0, angle)
        x, y = _to_screen(10.0 + ahead[0] * 3, 10.0 + ahead[1] * 3, pose, anchor, 20.0)
        assert x == pytest.approx(anchor[0], abs=1)
        assert y < anchor[1]


def test_touching_sightings_of_one_kind_are_one_group() -> None:
    fire = [Sighting(EntityKind.FIRE, Position(x, y), 0.5 + x / 10) for x in (3, 4) for y in (5, 6)]
    far_fire = Sighting(EntityKind.FIRE, Position(9, 9), 0.7)
    person = Sighting(EntityKind.VICTIM, Position(4, 6), 0.8)
    groups = group_sightings([*fire, far_fire, person])
    by_kind = sorted(groups, key=lambda g: (g.kind.value, g.left))
    assert len(groups) == 3
    first = by_kind[0]
    assert (first.left, first.top, first.right, first.bottom) == (3, 5, 4, 6)
    assert first.confidence == pytest.approx(0.9)


def test_remaining_is_what_lies_ahead_of_the_vehicle() -> None:
    route = Route(waypoints=(Position(0, 0), Position(1, 0), Position(2, 0)), cost=2.0)
    assert remaining(route, Position(1, 0)) == (Position(2, 0),)
    assert remaining(route, Position(0, 1)) == route.waypoints
    assert remaining(None, Position(0, 0)) == ()
    assert remaining(Route.unreachable(), Position(0, 0)) == ()


def test_vehicles_keep_left_and_people_do_not() -> None:
    from sentry_ai.domain.enums import Heading
    from sentry_ai.domain.traffic import AgentKind, TrafficAgent
    from sentry_ai.rendering.drive_view import TrafficView

    here = Position(5, 5)
    car = TrafficAgent("c", AgentKind.CAR, here, Heading.NORTH, here)
    person = TrafficAgent("p", AgentKind.PEDESTRIAN, here, Heading.NORTH, here)
    view = TrafficView(agents=(car, person), clock=10.0, step_ticks={k: 2 for k in AgentKind})
    car_x, car_y = view.where(car)
    assert car_x < 5.5 and car_y == pytest.approx(5.5)  # heading north, left is west
    assert view.where(person) == (5.5, 5.5)


def test_the_banner_shows_real_speed(city: CityMap) -> None:
    pygame.init()
    surface = pygame.Surface((640, 480))
    status = DriveStatus("EN ROUTE", "to victim_01", 20.0, speed_kmh=40.0, time_factor=5.4)
    DriveViewRenderer().draw(surface, surface.get_rect(), city, _pose(city), status=status)


def test_cruise_speed_must_be_positive() -> None:
    from sentry_ai.common.exceptions import ConfigValidationError
    from sentry_ai.config.schema import VehicleConfig

    with pytest.raises(ConfigValidationError):
        VehicleConfig(cruise_speed_kmh=0.0)


def test_the_command_center_draws_a_running_city_mission(project_root: Path) -> None:
    import sys

    sys.path.insert(0, str(project_root / "scripts"))
    import run_simulation as rs
    from sentry_ai.rendering.command_center import SIZE, CommandCenterRenderer
    from sentry_ai.rendering.simulation_app import SimulationApp
    from sentry_ai.rendering.theme import Theme

    pygame.init()
    loader = ConfigLoader(project_root=project_root)
    sys.argv = ["x", "--config", "configs/app_traffic.yaml", "--full", "--device", "cpu"]
    args = rs._parse_args()
    app_config = loader.load_app_config(args.config)
    scene = rs._build_scene(loader, app_config, args)
    map_data = dict(loader.load_yaml(app_config.map_config_path))
    map_data["geo"] = {"south": 0.0, "west": 0.0, "north": 0.004, "east": 0.006}
    app = SimulationApp(
        city_map=scene.city_map,
        engine=scene.engine,
        render_config=app_config.render,
        theme=Theme.from_config(loader, app_config.render.palette_config_path),
        scene=scene,
        metres_per_tile=20.0,
        cruise_kmh=40.0,
        dashboard=rs._dashboard_inputs(loader, app_config, args, map_data),
    )
    surface = pygame.Surface(SIZE)
    renderer = CommandCenterRenderer(app._drive)
    for _ in range(40):
        app._advance(0.2)
        renderer.draw(surface, app._command_center_state(0.2))
    assert surface.get_at((800, 30))[:3] != (0, 0, 0)
