#!/usr/bin/env python3
"""Fetch real street-level photos for an imported map (Phase 9 display).

Finds the Mapillary photos near every road tile of an OpenStreetMap-imported
map, downloads them once, and writes ``data/maps/street/<map>/index.json``.
``run_simulation.py`` then shows the photo for the vehicle's tile in the
mission-control strip (key ``P``). Needs a free Mapillary client token in the
``MAPILLARY_TOKEN`` environment variable; never put the token in a file.

Usage:
    python scripts/fetch_street_photos.py --map configs/maps/osm_annanagar.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from sentry_ai.common.logging_config import setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.mapping.overpass import BoundingBox
from sentry_ai.mapping.street_photos import ATTRIBUTION, assign, download, search, write_index

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Glyph for a road tile in an imported map's grid.
ROAD = "="


def street_dir(map_path: Path) -> Path:
    """Where the photos and index for ``map_path`` live."""
    return PROJECT_ROOT / "data" / "maps" / "street" / map_path.stem


def main() -> int:
    """Search, assign to tiles, download, index."""
    args = _parse_args()
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise SystemExit(
            "set MAPILLARY_TOKEN to a Mapillary client token (mapillary.com/developer)"
        )
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    setup_logging(loader.load_app_config().logging_config_path)
    map_path = loader.resolve(args.map)
    data = loader.load_yaml(map_path)
    if "geo" not in data:
        raise SystemExit(
            f"{args.map} has no 'geo' box; re-import it with scripts/import_osm_map.py"
        )
    geo, width, height = data["geo"], int(data["width"]), int(data["height"])
    box = BoundingBox(geo["south"], geo["west"], geo["north"], geo["east"])

    def centre(x: int, y: int) -> tuple[float, float]:
        lat = box.north - (y + 0.5) / height * (box.north - box.south)
        return lat, box.west + (x + 0.5) / width * (box.east - box.west)

    roads = {
        (x, y): centre(x, y)
        for y, row in enumerate(data["grid"])
        for x, cell in enumerate(row)
        if cell == ROAD
    }
    photos = search(box, token)
    assigned = assign(photos, roads, args.radius)
    folder = street_dir(map_path)
    for tile_photos in assigned.values():
        for photo in tile_photos:
            download(photo, token, folder)
    write_index(folder / "index.json", assigned)
    print(
        f"{len(photos)} photos in the area; {len(assigned)} of {len(roads)} road tiles "
        f"({len(assigned) / max(1, len(roads)):.0%}) have one within {args.radius:g} m. "
        f"{ATTRIBUTION}"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Mapillary street photos for a map.")
    parser.add_argument("--map", required=True, help="An OSM-imported map config.")
    parser.add_argument("--radius", type=float, default=40.0, help="Metres from tile to photo.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
