"""Unit tests for sentry_ai.sensors.camera geometry and projection."""

from __future__ import annotations

import pytest

from sentry_ai.common.exceptions import ConfigValidationError
from sentry_ai.domain.entities import Position, Vehicle
from sentry_ai.interfaces.perception import BoundingBox
from sentry_ai.sensors.camera import CameraView, OnboardCamera


def _view(**overrides: object) -> CameraView:
    defaults: dict[str, object] = {
        "camera_id": "cam",
        "origin": Position(4, 2),
        "width_tiles": 5,
        "height_tiles": 3,
        "tile_size_px": 10,
    }
    defaults.update(overrides)
    return CameraView(**defaults)  # type: ignore[arg-type]


class TestCameraViewValidation:
    def test_an_empty_id_is_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="camera_id"):
            _view(camera_id="  ")

    @pytest.mark.parametrize("field", ["width_tiles", "height_tiles", "tile_size_px"])
    def test_non_positive_dimensions_are_rejected(self, field: str) -> None:
        with pytest.raises(ConfigValidationError, match=field):
            _view(**{field: 0})


class TestCameraViewGeometry:
    def test_frame_size_is_tiles_times_tile_size(self) -> None:
        view = _view()
        assert (view.frame_width, view.frame_height) == (50, 30)

    def test_covers_its_own_footprint(self) -> None:
        view = _view()
        assert view.covers(Position(4, 2)) is True
        assert view.covers(Position(8, 4)) is True

    def test_does_not_cover_tiles_past_its_edges(self) -> None:
        view = _view()
        assert view.covers(Position(3, 2)) is False
        assert view.covers(Position(9, 2)) is False
        assert view.covers(Position(4, 5)) is False

    def test_the_origin_tile_starts_at_pixel_zero(self) -> None:
        assert _view().tile_rect(Position(4, 2)) == (0, 0, 10, 10)

    def test_tile_rects_advance_by_one_tile(self) -> None:
        assert _view().tile_rect(Position(6, 3)) == (20, 10, 30, 20)

    def test_world_tiles_are_row_major_and_complete(self) -> None:
        tiles = _view().world_tiles()
        assert len(tiles) == 15
        assert tiles[0] == Position(4, 2)
        assert tiles[1] == Position(5, 2)
        assert tiles[-1] == Position(8, 4)


class TestProjectionBackToTheMap:
    """to_world is how a detection becomes a map coordinate — it must be exact."""

    def test_the_first_pixel_maps_to_the_origin_tile(self) -> None:
        assert _view().to_world(0, 0) == Position(4, 2)

    def test_every_pixel_of_a_tile_maps_to_that_tile(self) -> None:
        view = _view()
        corners = [(20, 10), (29, 10), (20, 19), (29, 19), (24, 14)]
        assert {view.to_world(x, y) for x, y in corners} == {Position(6, 3)}

    def test_projection_round_trips_for_every_tile(self) -> None:
        view = _view()
        for tile in view.world_tiles():
            x_min, y_min, _, _ = view.tile_rect(tile)
            assert view.to_world(x_min, y_min) == tile

    @pytest.mark.parametrize(("x", "y"), [(-1, 0), (0, -1), (50, 0), (0, 30)])
    def test_pixels_outside_the_frame_are_rejected(self, x: int, y: int) -> None:
        with pytest.raises(ConfigValidationError, match="outside camera"):
            _view().to_world(x, y)


class TestProjectingABox:
    """tiles_of_box is what Phase 3.2 merges on.

    The footprint matters, not just the centre: a fire is annotated across
    its whole visible disc and is clipped differently by each camera that
    sees it, so two views of one fire agree on tiles that touch even when
    their box centres do not.
    """

    def test_a_box_inside_one_tile_yields_that_tile(self) -> None:
        # Tile (6, 3) occupies pixels x 20..30, y 10..20.
        assert _view().tiles_of_box(BoundingBox(22, 12, 28, 18)) == {Position(6, 3)}

    def test_a_box_spanning_tiles_yields_all_of_them(self) -> None:
        assert _view().tiles_of_box(BoundingBox(20, 10, 40, 20)) == {
            Position(6, 3),
            Position(7, 3),
        }

    def test_a_box_covering_the_frame_yields_every_tile(self) -> None:
        view = _view()
        assert view.tiles_of_box(BoundingBox(0, 0, 50, 30)) == set(view.world_tiles())

    def test_the_exclusive_maximum_does_not_leak_into_the_next_tile(self) -> None:
        """A box ending exactly on a boundary stops at the tile before it."""
        assert _view().tiles_of_box(BoundingBox(20, 10, 30, 20)) == {Position(6, 3)}

    def test_a_box_overhanging_the_frame_is_clamped(self) -> None:
        """A detector predicting past the edge cannot invent unseen tiles."""
        view = _view()
        tiles = view.tiles_of_box(BoundingBox(40, 20, 999, 999))
        assert tiles == {Position(8, 4)}
        assert all(view.covers(tile) for tile in tiles)

    def test_every_returned_tile_is_one_this_camera_sees(self) -> None:
        view = _view()
        assert all(view.covers(tile) for tile in view.tiles_of_box(BoundingBox(0, 0, 50, 30)))

    def test_the_result_agrees_with_to_world_at_the_corners(self) -> None:
        view = _view()
        box = BoundingBox(21, 11, 39, 19)
        tiles = view.tiles_of_box(box)
        assert view.to_world(box.x_min, box.y_min) in tiles
        assert view.to_world(box.x_max - 1, box.y_max - 1) in tiles


class TestOnboardCamera:
    _CAMERA = OnboardCamera(camera_id="onboard", span_tiles=5, tile_size_px=4)

    def _vehicle(self, x: int, y: int) -> Vehicle:
        return Vehicle(position=Position(x, y))

    def test_the_window_centres_on_the_vehicle(self) -> None:
        view = self._CAMERA.view_for(self._vehicle(10, 8), 30, 20)
        assert view.origin == Position(8, 6)
        assert view.covers(Position(10, 8))

    def test_the_window_is_clamped_at_the_top_left_corner(self) -> None:
        view = self._CAMERA.view_for(self._vehicle(0, 0), 30, 20)
        assert view.origin == Position(0, 0)

    def test_the_window_is_clamped_at_the_bottom_right_corner(self) -> None:
        view = self._CAMERA.view_for(self._vehicle(29, 19), 30, 20)
        assert view.origin == Position(25, 15)

    def test_the_frame_size_never_changes_as_the_vehicle_moves(self) -> None:
        """A CNN cannot accept a frame that shrinks in the corners."""
        sizes = {
            (view.frame_width, view.frame_height)
            for view in (
                self._CAMERA.view_for(self._vehicle(x, y), 30, 20)
                for x, y in [(0, 0), (15, 10), (29, 19), (0, 19), (29, 0)]
            )
        }
        assert sizes == {(20, 20)}

    def test_a_window_larger_than_the_map_pins_to_the_origin(self) -> None:
        camera = OnboardCamera(camera_id="wide", span_tiles=9, tile_size_px=4)
        view = camera.view_for(self._vehicle(2, 2), 5, 5)
        assert view.origin == Position(0, 0)
        assert (view.frame_width, view.frame_height) == (36, 36)

    def test_an_empty_id_is_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="camera_id"):
            OnboardCamera(camera_id="", span_tiles=5, tile_size_px=4)

    @pytest.mark.parametrize("field", ["span_tiles", "tile_size_px"])
    def test_non_positive_dimensions_are_rejected(self, field: str) -> None:
        kwargs: dict[str, object] = {
            "camera_id": "onboard",
            "span_tiles": 5,
            "tile_size_px": 4,
            field: 0,
        }
        with pytest.raises(ConfigValidationError, match=field):
            OnboardCamera(**kwargs)  # type: ignore[arg-type]


class TestProjectingByCentre:
    """tiles_centred_in_box is the strict rule fire is projected with.

    ``tiles_of_box`` claims a tile on a single pixel of overlap, which is
    right for a marker painted well inside one tile and wrong for anything
    spanning several: a detector's box is a pixel or two off, and that error
    claims a whole extra ring. For fire that ring is impassable, so it walls
    off streets that are open. Requiring the tile's centre absorbs up to
    half a tile of error instead.
    """

    def test_a_box_inside_one_tile_yields_that_tile(self) -> None:
        # Tile (6, 3) occupies pixels x 20..30, y 10..20; its centre is (25, 15).
        assert _view().tiles_centred_in_box(BoundingBox(22, 12, 28, 18)) == {Position(6, 3)}

    def test_a_box_that_misses_every_centre_yields_nothing(self) -> None:
        """The reason victims are never projected this way."""
        assert _view().tiles_centred_in_box(BoundingBox(26, 16, 29, 19)) == frozenset()

    def test_a_sliver_into_the_next_tile_does_not_claim_it(self) -> None:
        """The exact over-claim that doubled fire's footprint."""
        generous = _view().tiles_of_box(BoundingBox(22, 12, 32, 18))
        strict = _view().tiles_centred_in_box(BoundingBox(22, 12, 32, 18))
        assert generous == {Position(6, 3), Position(7, 3)}
        assert strict == {Position(6, 3)}

    def test_reaching_past_a_centre_does_claim_that_tile(self) -> None:
        assert _view().tiles_centred_in_box(BoundingBox(22, 12, 36, 18)) == {
            Position(6, 3),
            Position(7, 3),
        }

    def test_half_a_tile_of_error_is_tolerated(self) -> None:
        """A box 4px too wide on each side still yields the same tiles."""
        exact = _view().tiles_centred_in_box(BoundingBox(20, 10, 40, 20))
        sloppy = _view().tiles_centred_in_box(BoundingBox(16, 6, 44, 24))
        assert exact == sloppy == {Position(6, 3), Position(7, 3)}

    def test_a_box_covering_the_frame_yields_every_tile(self) -> None:
        view = _view()
        assert view.tiles_centred_in_box(BoundingBox(0, 0, 50, 30)) == set(view.world_tiles())

    def test_it_never_claims_more_than_the_generous_rule(self) -> None:
        view = _view()
        for box in (
            BoundingBox(0, 0, 50, 30),
            BoundingBox(22, 12, 28, 18),
            BoundingBox(13, 7, 47, 27),
        ):
            assert view.tiles_centred_in_box(box) <= view.tiles_of_box(box)

    def test_every_returned_tile_is_one_this_camera_sees(self) -> None:
        view = _view()
        tiles = view.tiles_centred_in_box(BoundingBox(0, 0, 50, 30))
        assert all(view.covers(tile) for tile in tiles)

    def test_a_box_overhanging_the_frame_invents_nothing(self) -> None:
        view = _view()
        tiles = view.tiles_centred_in_box(BoundingBox(40, 20, 999, 999))
        assert tiles == {Position(8, 4)}
