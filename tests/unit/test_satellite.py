"""Unit tests for sentry_ai.mapping.satellite (tile maths only; no network)."""

from __future__ import annotations

import pytest

from sentry_ai.mapping.overpass import BoundingBox
from sentry_ai.mapping.satellite import TILE_PX, mercator_pixel, zoom_for


class TestMercator:
    def test_origin_is_the_centre_of_the_world(self) -> None:
        x, y = mercator_pixel(0.0, 0.0, 0)
        assert (x, y) == pytest.approx((TILE_PX / 2, TILE_PX / 2))

    def test_east_is_right_and_north_is_up(self) -> None:
        x0, y0 = mercator_pixel(13.0, 80.0, 15)
        x1, y1 = mercator_pixel(13.1, 80.1, 15)
        assert x1 > x0 and y1 < y0

    def test_each_zoom_doubles_the_scale(self) -> None:
        x14, _ = mercator_pixel(13.0, 80.0, 14)
        x15, _ = mercator_pixel(13.0, 80.0, 15)
        assert x15 == pytest.approx(2 * x14)


class TestZoomFor:
    def test_picks_enough_detail_for_the_target_width(self) -> None:
        box = BoundingBox(12.968, 80.039, 12.975, 80.049)
        zoom = zoom_for(box, 1680)
        west, _ = mercator_pixel(box.north, box.west, zoom)
        east, _ = mercator_pixel(box.north, box.east, zoom)
        assert east - west >= 1680
        west, _ = mercator_pixel(box.north, box.west, zoom - 1)
        east, _ = mercator_pixel(box.north, box.east, zoom - 1)
        assert east - west < 1680
