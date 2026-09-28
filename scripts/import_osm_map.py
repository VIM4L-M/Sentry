#!/usr/bin/env python3
"""Turn a real place into a SENTRY disaster city, from OpenStreetMap.

Fetches the streets and buildings around a place (cached under
``data/maps/osm_cache``), rasterises them onto the shipped map's grid size,
stages a disaster on them, checks the result loads and every victim is
reachable, and writes ``configs/maps/osm_<name>.yaml``. Run once, with a
network; missions then load the file offline.

Usage:
    python scripts/import_osm_map.py --place "Kattankulathur, Chengalpattu" --name ktr
    python scripts/import_osm_map.py --lat 13.0378 --lon 80.2318 --name tnagar --seed 3
    python scripts/run_simulation.py --map configs/maps/osm_srm.yaml --full
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import yaml

from sentry_ai.common.logging_config import setup_logging
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.map import CityMap
from sentry_ai.mapping.osm import (
    DisasterSpec,
    box_around,
    connect,
    hospital_tile,
    infer_blocks,
    rasterise,
    stage_disaster,
)
from sentry_ai.mapping.overpass import OsmClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LEGEND = {
    ".": "open_ground",
    "#": "building",
    "=": "road",
    "x": "collapsed_building",
    "r": "rubble",
    "t": "tree",
    "!": "blocked_road",
}


def main() -> int:
    """Fetch, rasterise, stage, validate, write."""
    args = _parse_args()
    loader = ConfigLoader(project_root=PROJECT_ROOT)
    setup_logging(loader.load_app_config().logging_config_path)
    reference = loader.load_yaml(loader.load_app_config().map_config_path)
    width = args.width or int(reference["width"])
    height = args.height or int(reference["height"])

    # A district-sized box is tens of megabytes of OSM; give the server time.
    timeout = 60.0 if width * height <= 2_000 else 240.0
    client = OsmClient(PROJECT_ROOT / "data" / "maps" / "osm_cache", timeout_seconds=timeout)
    lat, lon = (args.lat, args.lon) if args.lat is not None else client.geocode(args.place)
    box = box_around(lat, lon, width, height, args.tile_metres)
    elements = client.ways(box)
    grid = rasterise(elements, box, width, height)
    grid = connect(infer_blocks(grid) if args.infer_blocks else grid)
    staged = stage_disaster(
        grid, random.Random(args.seed), _disaster(args), hospital_tile(elements, box, width, height)
    )
    geo = {"south": box.south, "west": box.west, "north": box.north, "east": box.east}
    config: dict[str, Any] = {
        "width": width,
        "height": height,
        "terrain_legend": LEGEND,
        **staged,
        # Where on Earth the grid sits; scripts/fetch_satellite.py reads it.
        "geo": geo,
    }
    CityMap.from_config(config)  # validates before anything is written

    path = PROJECT_ROOT / "configs" / "maps" / f"osm_{args.name}.yaml"
    _write(path, config, _header(args, lat, lon, len(elements)))
    print(f"Wrote {path.relative_to(PROJECT_ROOT)} from {len(elements)} OSM elements:")
    for row in staged["grid"]:
        print(f"  {row}")
    return 0


def _disaster(args: argparse.Namespace) -> DisasterSpec:
    """The default disaster, scaled up by the command line for a city-sized map."""
    default = DisasterSpec()
    return DisasterSpec(
        victims=args.victims or default.victims,
        fires=args.fires or default.fires,
        collapses=args.collapses or default.collapses,
        min_victim_distance=args.min_distance or default.min_victim_distance,
        max_victim_distance=args.max_distance,
        min_passable_share=args.min_street_share or default.min_passable_share,
    )


def _header(args: argparse.Namespace, lat: float, lon: float, elements: int) -> str:
    place = args.place or f"{lat:.5f}, {lon:.5f}"
    inferred = (
        "# Streets are from OpenStreetMap; most BUILDINGS ARE INFERRED (open land 2+ tiles\n"
        "# from a road filled in), because OSM has few building outlines for this area.\n"
        if args.infer_blocks
        else ""
    )
    return (
        f"# Disaster city generated from OpenStreetMap: {place}\n"
        f"# Centre {lat:.5f}, {lon:.5f}; {args.tile_metres:g} m per tile; "
        f"{elements} OSM elements;\n"
        f"# disaster staged with seed {args.seed} by scripts/import_osm_map.py.\n"
        f"{inferred}"
        "# Map data (c) OpenStreetMap contributors, available under the Open Database\n"
        "# License (ODbL): https://www.openstreetmap.org/copyright\n\n"
    )


def _write(path: Path, config: dict[str, Any], header: str) -> None:
    body = yaml.safe_dump(config, sort_keys=False, default_flow_style=None, width=200)
    path.write_text(header + body, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a real place as a SENTRY map.")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--place", help="A place name OpenStreetMap can geocode.")
    where.add_argument("--lat", type=float, help="Centre latitude (with --lon).")
    parser.add_argument("--lon", type=float, help="Centre longitude (with --lat).")
    parser.add_argument(
        "--name", required=True, help="Short name: writes configs/maps/osm_<name>.yaml."
    )
    parser.add_argument("--tile-metres", type=float, default=15.0, help="Real size of one tile.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for placing the disaster.")
    parser.add_argument(
        "--infer-blocks",
        action="store_true",
        help="Fill open land away from roads with building, where OSM lacks outlines.",
    )
    size = parser.add_argument_group("a larger map (default: the shipped map's size)")
    size.add_argument("--width", type=int, help="Tiles across.")
    size.add_argument("--height", type=int, help="Tiles down.")
    size.add_argument("--victims", type=int, help="Victims to place (default 4).")
    size.add_argument("--fires", type=int, help="Fires to start, at most one per victim.")
    size.add_argument("--collapses", type=int, help="Collapsed buildings (default 3).")
    size.add_argument("--min-distance", type=int, help="Nearest a victim is to the hospital.")
    size.add_argument("--max-distance", type=int, help="Farthest a victim is from the hospital.")
    size.add_argument(
        "--min-street-share", type=float, help="Share of the map that must be street (0.25)."
    )
    args = parser.parse_args()
    if args.lat is not None and args.lon is None:
        parser.error("--lat needs --lon")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
