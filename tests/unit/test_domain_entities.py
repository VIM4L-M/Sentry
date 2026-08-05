"""Unit tests for sentry_ai.domain.entities."""

from __future__ import annotations

import pytest

from sentry_ai.common.exceptions import DomainValidationError
from sentry_ai.domain.entities import FireSource, Obstacle, Position, SafeZone, Vehicle, Victim
from sentry_ai.domain.enums import EntityKind, TerrainType, VictimStatus


class TestPosition:
    def test_rejects_negative_x(self) -> None:
        with pytest.raises(DomainValidationError):
            Position(x=-1, y=0)

    def test_rejects_negative_y(self) -> None:
        with pytest.raises(DomainValidationError):
            Position(x=0, y=-1)

    def test_distance_to_is_euclidean(self) -> None:
        assert Position(0, 0).distance_to(Position(3, 4)) == pytest.approx(5.0)

    def test_as_tuple(self) -> None:
        assert Position(2, 5).as_tuple() == (2, 5)

    def test_is_hashable_for_dict_keys(self) -> None:
        d = {Position(1, 1): "a"}
        assert d[Position(1, 1)] == "a"


class TestVictim:
    def test_defaults(self) -> None:
        victim = Victim(victim_id="v1", position=Position(0, 0))
        assert victim.status is VictimStatus.TRAPPED
        assert victim.health == 100
        assert victim.entity_kind is EntityKind.VICTIM

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(DomainValidationError):
            Victim(victim_id="  ", position=Position(0, 0))

    @pytest.mark.parametrize("health", [-1, 101])
    def test_rejects_out_of_range_health(self, health: int) -> None:
        with pytest.raises(DomainValidationError):
            Victim(victim_id="v1", position=Position(0, 0), health=health)


class TestFireSource:
    def test_rejects_empty_id(self) -> None:
        with pytest.raises(DomainValidationError):
            FireSource(fire_id="", position=Position(0, 0))

    @pytest.mark.parametrize("intensity", [-0.1, 1.1])
    def test_rejects_out_of_range_intensity(self, intensity: float) -> None:
        with pytest.raises(DomainValidationError):
            FireSource(fire_id="f1", position=Position(0, 0), intensity=intensity)

    def test_rejects_negative_radius(self) -> None:
        with pytest.raises(DomainValidationError):
            FireSource(fire_id="f1", position=Position(0, 0), radius=-1)


class TestObstacle:
    def test_blocks_movement_matches_terrain_kind(self) -> None:
        rubble = Obstacle(obstacle_id="o1", position=Position(0, 0), kind=TerrainType.RUBBLE)
        assert rubble.blocks_movement is True

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(DomainValidationError):
            Obstacle(obstacle_id="", position=Position(0, 0), kind=TerrainType.TREE)

    def test_entity_kind_is_obstacle(self) -> None:
        obstacle = Obstacle(obstacle_id="o1", position=Position(0, 0), kind=TerrainType.TREE)
        assert obstacle.entity_kind is EntityKind.OBSTACLE


class TestSafeZone:
    def test_contains_within_radius(self) -> None:
        zone = SafeZone(position=Position(5, 5), radius=2)
        assert zone.contains(Position(6, 6)) is True

    def test_does_not_contain_outside_radius(self) -> None:
        zone = SafeZone(position=Position(5, 5), radius=1)
        assert zone.contains(Position(10, 10)) is False

    def test_rejects_negative_radius(self) -> None:
        with pytest.raises(DomainValidationError):
            SafeZone(position=Position(0, 0), radius=-1)

    def test_rejects_non_positive_capacity(self) -> None:
        with pytest.raises(DomainValidationError):
            SafeZone(position=Position(0, 0), capacity=0)


class TestVehicle:
    def test_defaults_are_fully_operational(self) -> None:
        vehicle = Vehicle(position=Position(0, 0))
        assert vehicle.is_operational() is True
        assert vehicle.onboard_victims == []

    @pytest.mark.parametrize("battery_percent", [-1.0, 101.0])
    def test_rejects_out_of_range_battery(self, battery_percent: float) -> None:
        with pytest.raises(DomainValidationError):
            Vehicle(position=Position(0, 0), battery_percent=battery_percent)

    def test_zero_battery_is_not_operational(self) -> None:
        vehicle = Vehicle(position=Position(0, 0), battery_percent=0.0)
        assert vehicle.is_operational() is False

    def test_rejects_more_onboard_victims_than_capacity(self) -> None:
        victims = [Victim(victim_id=f"v{i}", position=Position(0, 0)) for i in range(3)]
        with pytest.raises(DomainValidationError):
            Vehicle(position=Position(0, 0), capacity=2, onboard_victims=victims)

    def test_rejects_non_positive_capacity(self) -> None:
        with pytest.raises(DomainValidationError):
            Vehicle(position=Position(0, 0), capacity=0)
