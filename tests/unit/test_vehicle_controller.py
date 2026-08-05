"""Unit tests for sentry_ai.simulation.vehicle_controller."""

from __future__ import annotations

import pytest

from sentry_ai.config.schema import BatteryConfig, VehicleConfig
from sentry_ai.domain.entities import Position, Vehicle
from sentry_ai.domain.enums import Heading
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.interfaces.navigation import LocalAction
from sentry_ai.simulation.vehicle_controller import VehicleController

_CONFIG = VehicleConfig(
    battery=BatteryConfig(drain_per_move=1.0, drain_per_turn=0.5, drain_per_idle_tick=0.1),
    collision_damage_percent=5.0,
    fire_damage_per_tick=2.0,
)


@pytest.fixture
def grid() -> OccupancyGrid:
    """A 3x3 open grid with a building at (1, 0)."""
    grid = OccupancyGrid.empty(width=3, height=3)
    grid.mark(Position(1, 0), OccupancyCode.BUILDING)
    return grid


@pytest.fixture
def controller() -> VehicleController:
    return VehicleController(_CONFIG)


class TestTurning:
    def test_turn_left_rotates_and_spends_turn_battery(
        self, controller: VehicleController, grid: OccupancyGrid
    ) -> None:
        vehicle = Vehicle(position=Position(1, 1), heading=Heading.NORTH)
        outcome = controller.apply(vehicle, LocalAction.TURN_LEFT, grid)
        assert vehicle.heading is Heading.WEST
        assert vehicle.position == Position(1, 1)
        assert outcome.battery_spent == pytest.approx(0.5)
        assert outcome.moved is False

    def test_turn_right_rotates_clockwise(
        self, controller: VehicleController, grid: OccupancyGrid
    ) -> None:
        vehicle = Vehicle(position=Position(1, 1), heading=Heading.NORTH)
        controller.apply(vehicle, LocalAction.TURN_RIGHT, grid)
        assert vehicle.heading is Heading.EAST


class TestMovement:
    def test_forward_moves_one_tile_along_the_heading(
        self, controller: VehicleController, grid: OccupancyGrid
    ) -> None:
        vehicle = Vehicle(position=Position(1, 1), heading=Heading.SOUTH)
        outcome = controller.apply(vehicle, LocalAction.MOVE_FORWARD, grid)
        assert vehicle.position == Position(1, 2)
        assert outcome.moved is True
        assert outcome.battery_spent == pytest.approx(1.0)

    def test_reverse_moves_against_the_heading_without_turning(
        self, controller: VehicleController, grid: OccupancyGrid
    ) -> None:
        vehicle = Vehicle(position=Position(1, 1), heading=Heading.NORTH)
        controller.apply(vehicle, LocalAction.REVERSE, grid)
        assert vehicle.position == Position(1, 2)
        assert vehicle.heading is Heading.NORTH

    def test_stop_holds_position_and_spends_idle_battery(
        self, controller: VehicleController, grid: OccupancyGrid
    ) -> None:
        vehicle = Vehicle(position=Position(1, 1), heading=Heading.NORTH)
        outcome = controller.apply(vehicle, LocalAction.STOP, grid)
        assert vehicle.position == Position(1, 1)
        assert outcome.battery_spent == pytest.approx(0.1)


class TestCollisions:
    def test_driving_into_a_building_is_refused_and_damages(
        self, controller: VehicleController, grid: OccupancyGrid
    ) -> None:
        vehicle = Vehicle(position=Position(1, 1), heading=Heading.NORTH)
        outcome = controller.apply(vehicle, LocalAction.MOVE_FORWARD, grid)
        assert vehicle.position == Position(1, 1)
        assert outcome.collided is True
        assert outcome.moved is False
        assert vehicle.health_percent == pytest.approx(95.0)

    def test_driving_off_the_map_edge_is_refused(
        self, controller: VehicleController, grid: OccupancyGrid
    ) -> None:
        vehicle = Vehicle(position=Position(0, 0), heading=Heading.WEST)
        outcome = controller.apply(vehicle, LocalAction.MOVE_FORWARD, grid)
        assert vehicle.position == Position(0, 0)
        assert outcome.collided is True

    def test_negative_target_coordinates_never_construct_a_position(
        self, controller: VehicleController, grid: OccupancyGrid
    ) -> None:
        """Position forbids negatives, so the edge case must be caught first."""
        vehicle = Vehicle(position=Position(0, 0), heading=Heading.NORTH)
        outcome = controller.apply(vehicle, LocalAction.MOVE_FORWARD, grid)  # must not raise
        assert outcome.collided is True


class TestHazardsAndClamping:
    def test_standing_in_fire_costs_health_every_tick(
        self, controller: VehicleController, grid: OccupancyGrid
    ) -> None:
        grid.mark(Position(1, 1), OccupancyCode.FIRE)
        vehicle = Vehicle(position=Position(1, 1), heading=Heading.NORTH)
        controller.apply(vehicle, LocalAction.STOP, grid)
        assert vehicle.health_percent == pytest.approx(98.0)

    def test_battery_never_goes_negative(
        self, controller: VehicleController, grid: OccupancyGrid
    ) -> None:
        vehicle = Vehicle(position=Position(1, 1), heading=Heading.SOUTH, battery_percent=0.2)
        controller.apply(vehicle, LocalAction.MOVE_FORWARD, grid)
        assert vehicle.battery_percent == 0.0
        assert vehicle.is_operational() is False

    def test_health_never_goes_negative(
        self, controller: VehicleController, grid: OccupancyGrid
    ) -> None:
        vehicle = Vehicle(position=Position(1, 1), heading=Heading.NORTH, health_percent=1.0)
        controller.apply(vehicle, LocalAction.MOVE_FORWARD, grid)  # collides, 5.0 damage
        assert vehicle.health_percent == 0.0
