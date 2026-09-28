"""Aerial imagery for an imported map, as a picture to draw the city on.

Downloads the Web Mercator tiles covering a map's lat/lon box from Esri
World Imagery, stitches them, and crops the result to exactly that box, so
pixel (0, 0) of the image is the map's top-left tile corner. The image is
scenery for the operator display only. The simulated cameras and every
model keep seeing the rasterised city they were trained on; satellite
pixels never reach a detector.

Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community.
Tiles are cached, so the review runs offline once an image is fetched.
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path

from sentry_ai.common.logging_config import get_logger
from sentry_ai.mapping.overpass import USER_AGENT, BoundingBox

logger = get_logger(__name__)

TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ATTRIBUTION = "Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community"

#: Tile edge in pixels, fixed by the Web Mercator tiling scheme.
TILE_PX = 256


def mercator_pixel(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Global Web Mercator pixel coordinates of a point at ``zoom``."""
    scale = TILE_PX * 2**zoom
    x = (lon + 180.0) / 360.0 * scale
    siny = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * scale
    return x, y


def zoom_for(box: BoundingBox, target_width_px: int, max_zoom: int = 19) -> int:
    """The smallest zoom whose imagery is at least ``target_width_px`` across ``box``."""
    for zoom in range(12, max_zoom + 1):
        west, _ = mercator_pixel(box.north, box.west, zoom)
        east, _ = mercator_pixel(box.north, box.east, zoom)
        if east - west >= target_width_px:
            return zoom
    return max_zoom


def fetch_image(box: BoundingBox, size: tuple[int, int], cache_dir: Path) -> object:
    """A ``size`` PIL image of ``box``, stitched from cached or downloaded tiles.

    Returns a ``PIL.Image.Image``; typed loosely so importing this module
    does not import Pillow.
    """
    from PIL import Image  # noqa: PLC0415 - only the fetch script needs Pillow

    zoom = zoom_for(box, size[0])
    left, top = mercator_pixel(box.north, box.west, zoom)
    right, bottom = mercator_pixel(box.south, box.east, zoom)
    tiles_x = range(int(left // TILE_PX), int(right // TILE_PX) + 1)
    tiles_y = range(int(top // TILE_PX), int(bottom // TILE_PX) + 1)
    mosaic = Image.new("RGB", (len(tiles_x) * TILE_PX, len(tiles_y) * TILE_PX))
    for column, tx in enumerate(tiles_x):
        for row, ty in enumerate(tiles_y):
            tile = Image.open(_tile(zoom, tx, ty, cache_dir)).convert("RGB")
            mosaic.paste(tile, (column * TILE_PX, row * TILE_PX))
    origin_x, origin_y = tiles_x[0] * TILE_PX, tiles_y[0] * TILE_PX
    crop = (left - origin_x, top - origin_y, right - origin_x, bottom - origin_y)
    box_px = (round(crop[0]), round(crop[1]), round(crop[2]), round(crop[3]))
    return mosaic.crop(box_px).resize(size, Image.Resampling.LANCZOS)


def _tile(zoom: int, x: int, y: int, cache_dir: Path) -> Path:
    path = cache_dir / f"{zoom}_{x}_{y}.jpg"
    if not path.is_file():
        url = TILE_URL.format(z=zoom, x=x, y=y)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path
