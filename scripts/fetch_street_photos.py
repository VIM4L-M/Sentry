#!/usr/bin/env python3
"""Fetch real street-level photos for an imported map (Phase 9 display).

Finds the Mapillary photos near every road tile of an OpenStreetMap-imported
map, downloads them once, and writes ``data/maps/street/<map>/index.json``.
``run_simulation.py`` then shows the photo for the vehicle's tile in the
mission-control strip (key ``P``). Needs a free Mapillary client token in the
``MAPILLARY_TOKEN`` environment variable; never put the token in a file.

Usage:
    python scripts/fetch_street_photos.py --map configs/maps/osm_annanagar.yaml
    python scripts/fetch_street_photos.py --map configs/maps/osm_chicago.yaml
        --around-mission 12 --photos-per-tile 2 --max-width 640   (one command)
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sentry_ai.common.logging_config import get_logger, setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.mapping.overpass import BoundingBox
from sentry_ai.mapping.street_photos import (
    ATTRIBUTION,
    PHOTOS_PER_TILE,
    assign,
    distance_m,
    download,
    search,
    write_index,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger(__name__)

#: Glyph for a road tile in an imported map's grid.
ROAD = "="

#: Widest search cell; see ``_splits``.
SEARCH_CELL_METRES = 400.0

#: Parallel downloads: the work is waiting on the network, not the CPU.
DOWNLOAD_THREADS = 8


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
    if args.around_mission is not None:
        x0, y0, x1, y1 = _mission_area(data, args.around_mission)
        roads = {t: c for t, c in roads.items() if x0 <= t[0] <= x1 and y0 <= t[1] <= y1}
        north, west = centre(x0, y0)
        south, east = centre(x1, y1)
        box = BoundingBox(south, west, north, east)
    photos = search(box, token, splits=_splits(box))
    assigned = assign(photos, roads, args.radius, per_tile=args.photos_per_tile)
    folder = street_dir(map_path)
    wanted = [photo for tile_photos in assigned.values() for photo in tile_photos]
    print(f"{len(photos)} photos found; downloading {len(wanted)} ...", flush=True)
    failed: set[str] = set()
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
        jobs = {pool.submit(download, p, token, folder, args.max_width): p for p in wanted}
        for done, job in enumerate(as_completed(jobs), start=1):
            try:
                job.result()
            except OSError as exc:  # a timeout or a dropped connection costs one photo
                failed.add(jobs[job].image_id)
                logger.warning("Skipped photo %s: %s", jobs[job].image_id, exc)
            if done % 500 == 0:
                print(f"  {done}/{len(wanted)}", flush=True)
    if failed:
        print(f"  {len(failed)} photos failed to download and were left out", flush=True)
        assigned = {
            tile: kept
            for tile, photos_here in assigned.items()
            if (kept := [p for p in photos_here if p.image_id not in failed])
        }
    write_index(folder / "index.json", assigned)
    print(
        f"{len(photos)} photos in the area; {len(assigned)} of {len(roads)} road tiles "
        f"({len(assigned) / max(1, len(roads)):.0%}) have one within {args.radius:g} m. "
        f"{ATTRIBUTION}"
    )
    return 0


def _mission_area(data: dict[str, Any], margin: int) -> tuple[int, int, int, int]:
    """Tile bounds around the hospital, the start and every victim, plus ``margin``."""
    points = [data["safe_zone"]["position"], data["vehicle_start"]]
    points += [victim["position"] for victim in data.get("victims", [])]
    xs, ys = [int(p[0]) for p in points], [int(p[1]) for p in points]
    width, height = int(data["width"]), int(data["height"])
    return (
        max(0, min(xs) - margin),
        max(0, min(ys) - margin),
        min(width - 1, max(xs) + margin),
        min(height - 1, max(ys) + margin),
    )


def _splits(box: BoundingBox) -> int:
    """Search cells per side, so each is at most ``SEARCH_CELL_METRES`` across.

    The API returns at most 2,000 photos per query; a city centre can hold
    several hundred in 400 m.
    """
    span = max(
        distance_m(box.south, box.west, box.north, box.west),
        distance_m(box.south, box.west, box.south, box.east),
    )
    return max(3, math.ceil(span / SEARCH_CELL_METRES))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Mapillary street photos for a map.")
    parser.add_argument("--map", required=True, help="An OSM-imported map config.")
    parser.add_argument("--radius", type=float, default=40.0, help="Metres from tile to photo.")
    parser.add_argument(
        "--around-mission",
        type=int,
        metavar="TILES",
        help="Only the area around the hospital, start and victims, plus this margin "
        "(for city-sized maps).",
    )
    parser.add_argument(
        "--photos-per-tile",
        type=int,
        default=PHOTOS_PER_TILE,
        help=f"Photos kept per road tile, for matching the heading (default {PHOTOS_PER_TILE}).",
    )
    parser.add_argument(
        "--max-width", type=int, help="Shrink photos to this width in pixels as they download."
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
