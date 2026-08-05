"""Unit tests for sentry_ai.domain.map.CityMap."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from sentry_ai.common.exceptions import DomainValidationError
from sentry_ai.domain.entities import Position, SafeZone, Vehicle, Victim
from sentry_ai.domain.enums import TerrainType
from sentry_ai.domain.map import CityMap

# A small 4x3 map:
#   row0: ....
#   row1: .##.
#   row2: .r.t
_BASE_MAP_DATA: dict[str, Any] = {
    "width": 4,
    "height": 3,
    "grid": [
        "....",
        ".##.",
        ".r.t",
    ],
    "safe_zone": {"position": [0, 0], "radius": 1, "capacity": 2},
    "vehicle_start": [0, 1],
    "victims": [{"id": "v1", "position": [2, 2]}],
    "fires": [{"id": "f1", "position": [3, 0], "intensity": 0.8, "radius": 2}],
}


def _map_data(**overrides: Any) -> dict[str, Any]:
    data = copy.deepcopy(_BASE_MAP_DATA)
    data.update(overrides)
    return data


class TestFromConfig:
    def test_builds_expected_dimensions_and_entities(self) -> None:
        city_map = CityMap.from_config(_map_data())
        assert (city_map.width, city_map.height) == (4, 3)
        assert len(city_map.victims) == 1
        assert len(city_map.fires) == 1
        assert len(city_map.obstacles) == 2  # one rubble, one tree

    def test_missing_width_raises(self) -> None:
        data = _map_data()
        del data["width"]
        with pytest.raises(DomainValidationError):
            CityMap.from_config(data)

    def test_grid_row_count_mismatch_raises(self) -> None:
        with pytest.raises(DomainValidationError):
            CityMap.from_config(_map_data(height=5))  # grid still has 3 rows

    def test_grid_row_length_mismatch_raises(self) -> None:
        data = _map_data()
        data["grid"] = ["...", ".##.", ".r.t"]  # first row too short
        with pytest.raises(DomainValidationError):
            CityMap.from_config(data)

    def test_unknown_terrain_symbol_raises(self) -> None:
        data = _map_data()
        data["grid"] = ["..Z.", ".##.", ".r.t"]
        with pytest.raises(DomainValidationError):
            CityMap.from_config(data)

    def test_missing_safe_zone_raises(self) -> None:
        data = _map_data()
        del data["safe_zone"]
        with pytest.raises(DomainValidationError):
            CityMap.from_config(data)

    def test_missing_vehicle_start_raises(self) -> None:
        data = _map_data()
        del data["vehicle_start"]
        with pytest.raises(DomainValidationError):
            CityMap.from_config(data)

    def test_vehicle_on_blocking_terrain_raises(self) -> None:
        data = _map_data(vehicle_start=[1, 1])  # a '#' building tile
        with pytest.raises(DomainValidationError):
            CityMap.from_config(data)

    def test_custom_terrain_legend_is_merged_with_default(self) -> None:
        data = _map_data()
        data["grid"] = ["S...", ".##.", ".r.t"]
        data["terrain_legend"] = {"S": "safe_zone"}
        city_map = CityMap.from_config(data)
        assert city_map.tile_at(Position(0, 0)) is TerrainType.SAFE_ZONE


class TestQueries:
    def test_tile_at_defaults_to_open_ground(self) -> None:
        city_map = CityMap.from_config(_map_data())
        assert city_map.tile_at(Position(3, 2)) is TerrainType.TREE
        assert city_map.tile_at(Position(0, 2)) is TerrainType.OPEN_GROUND

    def test_is_walkable_false_for_building(self) -> None:
        city_map = CityMap.from_config(_map_data())
        assert city_map.is_walkable(Position(1, 1)) is False

    def test_is_walkable_false_out_of_bounds(self) -> None:
        city_map = CityMap.from_config(_map_data())
        assert city_map.is_walkable(Position(99, 99)) is False

    def test_is_walkable_true_for_open_ground(self) -> None:
        city_map = CityMap.from_config(_map_data())
        assert city_map.is_walkable(Position(0, 2)) is True

    def test_entities_near_returns_within_radius(self) -> None:
        city_map = CityMap.from_config(_map_data())
        nearby = city_map.entities_near(Position(3, 0), radius=0.5)
        assert any(isinstance(e, type(city_map.fires[0])) for e in nearby)
        assert city_map.vehicle not in nearby  # vehicle starts at (0, 1), too far


class TestValidate:
    def test_rejects_out_of_bounds_vehicle(self) -> None:
        with pytest.raises(DomainValidationError):
            CityMap(
                width=2,
                height=2,
                terrain={},
                vehicle=Vehicle(position=Position(5, 5)),
                safe_zone=SafeZone(position=Position(0, 0)),
            )

    def test_rejects_non_positive_dimensions(self) -> None:
        with pytest.raises(DomainValidationError):
            CityMap(
                width=0,
                height=2,
                terrain={},
                vehicle=Vehicle(position=Position(0, 0)),
                safe_zone=SafeZone(position=Position(0, 0)),
            )

    def test_rejects_duplicate_victim_ids(self) -> None:
        with pytest.raises(DomainValidationError):
            CityMap(
                width=3,
                height=3,
                terrain={},
                vehicle=Vehicle(position=Position(0, 0)),
                safe_zone=SafeZone(position=Position(1, 1)),
                victims=[
                    Victim(victim_id="dup", position=Position(0, 0)),
                    Victim(victim_id="dup", position=Position(1, 0)),
                ],
            )


def test_real_default_city_map_loads_and_validates(project_root: Any) -> None:
    """The actual configs/maps/city_default.yaml shipped in the repo must be valid."""
    from sentry_ai.config.loader import ConfigLoader

    loader = ConfigLoader(project_root=project_root)
    data = loader.load_yaml("configs/maps/city_default.yaml")
    city_map = CityMap.from_config(data)

    assert city_map.width == 30
    assert city_map.height == 20
    assert len(city_map.victims) == 4
    assert len(city_map.fires) == 2
    assert city_map.is_walkable(city_map.vehicle.position)
