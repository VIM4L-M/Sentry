#!/usr/bin/env python3
"""Fetch aerial imagery for an OpenStreetMap-imported map (Phase 9 display).

Reads the map's ``geo`` box (written by ``scripts/import_osm_map.py``),
downloads and stitches the matching Esri World Imagery tiles, and writes
``data/maps/satellite/<map>.png`` sized to the map at the window's tile
size. ``run_simulation.py`` picks it up automatically; press ``S`` to switch
between the satellite and the drawn map. Display only: the simulated cameras
and every model still see the rasterised city.

Usage:
    python scripts/fetch_satellite.py --map configs/maps/osm_cit.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentry_ai.common.logging_config import setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.mapping.overpass import BoundingBox
from sentry_ai.mapping.satellite import ATTRIBUTION, fetch_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Imagery is fetched at this multiple of the window's pixels, so it stays sharp.
OVERSAMPLE = 2

#: Longest side of a saved image, in pixels.
MAX_SIDE_PX = 4096


def satellite_path(map_path: Path) -> Path:
    """Where the imagery for ``map_path`` lives."""
    return PROJECT_ROOT / "data" / "maps" / "satellite" / f"{map_path.stem}.png"


def main() -> int:
    """Fetch, stitch, crop, save."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    app = loader.load_app_config()
    setup_logging(app.logging_config_path)
    map_path = loader.resolve(args.map)
    data = loader.load_yaml(map_path)
    if "geo" not in data:
        raise SystemExit(
            f"{args.map} has no 'geo' box; re-import it with scripts/import_osm_map.py"
        )
    geo = data["geo"]
    box = BoundingBox(geo["south"], geo["west"], geo["north"], geo["east"])
    width, height = int(data["width"]), int(data["height"])
    # A city-sized map at the window's scale would be a gigapixel image; cap
    # the long side (about 1 m per pixel on a 4 km map, still street-sharp).
    tile = min(app.render.tile_size_px * OVERSAMPLE, MAX_SIDE_PX / max(width, height))
    size = (round(width * tile), round(height * tile))
    image = fetch_image(box, size, PROJECT_ROOT / "data" / "maps" / "satellite" / "tiles")
    out = satellite_path(map_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)  # type: ignore[attr-defined]
    print(f"Wrote {out.relative_to(PROJECT_ROOT)} ({size[0]}x{size[1]}). {ATTRIBUTION}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch satellite imagery for an imported map.")
    parser.add_argument("--map", required=True, help="An OSM-imported map config.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
