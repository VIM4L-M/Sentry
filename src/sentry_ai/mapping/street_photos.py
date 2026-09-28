"""Real street-level photos along an imported map's roads (Phase 9 display).

For every road tile of an OpenStreetMap-imported map, this finds the nearest
Mapillary photo within a radius, downloads its 1024 px thumbnail once, and
records which photo belongs to which tile in an index. At run time the
window shows the photo for the tile the vehicle is on, choosing among the
nearby photos the one whose camera faced closest to the vehicle's heading —
so driving north shows the street ahead going north.

Display only, like the satellite view: the simulated cameras and every model
keep seeing the rendered city. Photos are © their Mapillary contributors,
licensed CC BY-SA 4.0; access needs a free token in ``MAPILLARY_TOKEN``.
Everything is cached, so the review needs no network once fetched.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentry_ai.common.logging_config import get_logger
from sentry_ai.mapping.overpass import BoundingBox

logger = get_logger(__name__)

GRAPH_URL = "https://graph.mapillary.com/images"
ATTRIBUTION = "Street photos © Mapillary contributors, CC BY-SA 4.0"

#: Photos kept per tile, so a heading can be matched.
PHOTOS_PER_TILE = 4

_METRES_PER_DEGREE = 111_320.0


@dataclass(frozen=True)
class StreetPhoto:
    """One Mapillary photo: where it was taken and which way the camera faced."""

    image_id: str
    lat: float
    lon: float
    compass: float
    captured_at: int


def search(box: BoundingBox, token: str, splits: int = 3) -> list[StreetPhoto]:
    """Every photo in ``box``, queried in ``splits`` x ``splits`` cells under the API cap."""
    photos: dict[str, StreetPhoto] = {}
    dlat = (box.north - box.south) / splits
    dlon = (box.east - box.west) / splits
    for i in range(splits):
        for j in range(splits):
            south, west = box.south + i * dlat, box.west + j * dlon
            query = urllib.parse.urlencode(
                {
                    "access_token": token,
                    "fields": "id,computed_geometry,compass_angle,captured_at",
                    "bbox": f"{west},{south},{west + dlon},{south + dlat}",
                    "limit": 2000,
                }
            )
            for item in _get(f"{GRAPH_URL}?{query}").get("data", []):
                photo = _photo(item)
                if photo is not None:
                    photos[photo.image_id] = photo
    return list(photos.values())


def assign(
    photos: list[StreetPhoto],
    tiles: dict[tuple[int, int], tuple[float, float]],
    radius_metres: float,
) -> dict[tuple[int, int], list[StreetPhoto]]:
    """For each tile, up to ``PHOTOS_PER_TILE`` photos within ``radius_metres``, nearest first."""
    result: dict[tuple[int, int], list[StreetPhoto]] = {}
    for tile, (lat, lon) in tiles.items():
        near = sorted(
            (distance_m(lat, lon, p.lat, p.lon), p.captured_at * -1, p.image_id, p) for p in photos
        )
        chosen = [p for d, _, _, p in near if d <= radius_metres][:PHOTOS_PER_TILE]
        if chosen:
            result[tile] = chosen
    return result


def best_for_heading(photos: list[StreetPhoto], bearing: float) -> StreetPhoto:
    """The photo whose camera faced closest to ``bearing`` (degrees, 0 = north)."""
    return min(photos, key=lambda p: abs((p.compass - bearing + 180.0) % 360.0 - 180.0))


def download(photo: StreetPhoto, token: str, folder: Path) -> Path:
    """The photo's 1024 px thumbnail, downloaded once into ``folder``."""
    path = folder / f"{photo.image_id}.jpg"
    if not path.is_file():
        meta = _get(
            f"https://graph.mapillary.com/{photo.image_id}?fields=thumb_1024_url&access_token={token}"
        )
        request = urllib.request.Request(meta["thumb_1024_url"])
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        folder.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Ground distance in metres, flat-earth approximation (fine over a few km)."""
    dy = (lat1 - lat2) * _METRES_PER_DEGREE
    dx = (lon1 - lon2) * _METRES_PER_DEGREE * math.cos(math.radians(lat1))
    return math.hypot(dx, dy)


def write_index(path: Path, assigned: dict[tuple[int, int], list[StreetPhoto]]) -> None:
    """Save the tile -> photos table the window reads."""
    table = {
        f"{x},{y}": [{"id": p.image_id, "compass": p.compass} for p in photos]
        for (x, y), photos in sorted(assigned.items())
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"attribution": ATTRIBUTION, "tiles": table}, indent=1), encoding="utf-8"
    )


def _photo(item: dict[str, Any]) -> StreetPhoto | None:
    geometry = item.get("computed_geometry") or {}
    coordinates = geometry.get("coordinates")
    if not coordinates:
        return None
    return StreetPhoto(
        image_id=str(item["id"]),
        lat=float(coordinates[1]),
        lon=float(coordinates[0]),
        compass=float(item.get("compass_angle") or 0.0),
        captured_at=int(item.get("captured_at") or 0),
    )


def _get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())
