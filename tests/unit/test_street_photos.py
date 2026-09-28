"""Unit tests for sentry_ai.mapping.street_photos and the street-view library (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentry_ai.domain.entities import Position, Vehicle
from sentry_ai.domain.enums import Heading
from sentry_ai.mapping.street_photos import (
    PHOTOS_PER_TILE,
    StreetPhoto,
    assign,
    best_for_heading,
    distance_m,
    write_index,
)
from sentry_ai.rendering.street_view import StreetPhotoLibrary


def _photo(image_id: str, lat: float, lon: float, compass: float = 0.0) -> StreetPhoto:
    return StreetPhoto(image_id, lat, lon, compass, captured_at=0)


class TestGeometry:
    def test_distance_is_about_111_km_per_degree_of_latitude(self) -> None:
        assert distance_m(13.0, 80.0, 14.0, 80.0) == pytest.approx(111_320, rel=0.001)

    def test_best_for_heading_wraps_round_north(self) -> None:
        photos = [_photo("east", 0, 0, 90.0), _photo("almost_north", 0, 0, 350.0)]
        assert best_for_heading(photos, 5.0).image_id == "almost_north"


class TestAssign:
    def test_only_photos_within_the_radius_count(self) -> None:
        tiles = {(0, 0): (13.0, 80.0)}
        photos = [_photo("near", 13.0001, 80.0), _photo("far", 13.01, 80.0)]
        assigned = assign(photos, tiles, radius_metres=40)
        assert [p.image_id for p in assigned[(0, 0)]] == ["near"]

    def test_tiles_without_photos_are_left_out(self) -> None:
        assert assign([], {(0, 0): (13.0, 80.0)}, 40) == {}

    def test_keeps_at_most_a_few_photos_per_tile_nearest_first(self) -> None:
        photos = [_photo(str(i), 13.0 + i * 1e-5, 80.0) for i in range(10)]
        chosen = assign(photos, {(0, 0): (13.0, 80.0)}, 100)[(0, 0)]
        assert len(chosen) == PHOTOS_PER_TILE
        assert chosen[0].image_id == "0"


class TestLibrary:
    def _library(self, tmp_path: Path) -> StreetPhotoLibrary:
        assigned = {(2, 2): [_photo("north", 0, 0, 0.0), _photo("south", 0, 0, 180.0)]}
        write_index(tmp_path / "index.json", assigned)
        return StreetPhotoLibrary(tmp_path, tile_metres=20.0)

    def test_picks_the_photo_facing_the_vehicle_heading(self, tmp_path: Path) -> None:
        library = self._library(tmp_path)
        vehicle = Vehicle(position=Position(2, 2), heading=Heading.SOUTH)
        assert library.lookup(vehicle) == ("south", 0.0)

    def test_falls_back_to_a_nearby_tile_and_says_how_far(self, tmp_path: Path) -> None:
        library = self._library(tmp_path)
        vehicle = Vehicle(position=Position(4, 2), heading=Heading.NORTH)
        assert library.lookup(vehicle) == ("north", 40.0)

    def test_nothing_when_no_photo_is_close(self, tmp_path: Path) -> None:
        library = self._library(tmp_path)
        assert library.lookup(Vehicle(position=Position(20, 20))) is None

    def test_index_is_plain_json(self, tmp_path: Path) -> None:
        self._library(tmp_path)
        data = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
        assert "2,2" in data["tiles"] and "Mapillary" in data["attribution"]
