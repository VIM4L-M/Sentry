"""Unit tests for the drawn vehicle's glide (sentry_ai.rendering.motion)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.entities import Position, Vehicle
from sentry_ai.domain.enums import Heading
from sentry_ai.domain.map import CityMap
from sentry_ai.rendering.motion import VehicleGlide, heading_angle


@pytest.fixture
def vehicle(project_root: Path) -> Vehicle:
    loader = ConfigLoader(project_root=project_root)
    city = CityMap.from_config(loader.load_yaml("configs/maps/city_default.yaml"))
    city.vehicle.position = Position(5, 5)
    city.vehicle.heading = Heading.EAST
    return city.vehicle


def test_first_update_places_the_vehicle_on_its_tile(vehicle: Vehicle) -> None:
    pose = VehicleGlide().update(vehicle, 0.0, 0.1)
    assert (pose.x, pose.y) == (5.5, 5.5)
    assert math.isclose(pose.angle, math.pi / 2)


def test_glides_at_constant_speed_and_arrives_on_time(vehicle: Vehicle) -> None:
    glide = VehicleGlide()
    glide.update(vehicle, 0.0, 0.2)
    vehicle.position = Position(6, 5)
    assert glide.update(vehicle, 0.0, 0.2).x == pytest.approx(5.5)
    assert glide.update(vehicle, 0.1, 0.2).x == pytest.approx(6.0)
    assert glide.update(vehicle, 0.1, 0.2).x == pytest.approx(6.5)
    assert glide.update(vehicle, 0.5, 0.2).x == pytest.approx(6.5)


def test_turns_the_short_way(vehicle: Vehicle) -> None:
    glide = VehicleGlide()
    vehicle.heading = Heading.NORTH
    glide.update(vehicle, 0.0, 0.2)
    vehicle.heading = Heading.WEST
    glide.update(vehicle, 0.0, 0.2)
    halfway = glide.update(vehicle, 0.1, 0.2)
    assert halfway.angle == pytest.approx(-math.pi / 4)


def test_snaps_on_a_restart_sized_jump(vehicle: Vehicle) -> None:
    glide = VehicleGlide()
    glide.update(vehicle, 0.0, 0.2)
    vehicle.position = Position(20, 15)
    pose = glide.update(vehicle, 0.0, 0.2)
    assert (pose.x, pose.y) == (20.5, 15.5)


def test_reset_snaps_the_next_update(vehicle: Vehicle) -> None:
    glide = VehicleGlide()
    glide.update(vehicle, 0.0, 0.2)
    vehicle.position = Position(6, 5)
    glide.reset()
    assert glide.update(vehicle, 0.0, 0.2).x == pytest.approx(6.5)


def test_heading_angle_is_clockwise_from_north(vehicle: Vehicle) -> None:
    expected = {Heading.NORTH: 0.0, Heading.EAST: 90.0, Heading.SOUTH: 180.0, Heading.WEST: -90.0}
    for heading, degrees in expected.items():
        vehicle.heading = heading
        assert math.degrees(heading_angle(vehicle)) == pytest.approx(degrees)
