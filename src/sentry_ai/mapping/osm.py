"""OpenStreetMap ways -> a SENTRY disaster-city map config (pure, no network).

The city keeps the shipped map's size (30x20 tiles by default) so every
trained model and the camera network work on it unchanged. Only the street
plan changes. Each tile is ``tile_metres`` on a side (15 m by default, so
about 450 m x 300 m of real city). A larger grid works too — the driving
models are egocentric — for a whole district seen through the drive view.

Steps:

1. **Rasterise.** Roads are drawn as lines of ``=`` tiles; building
   footprints fill ``#`` (a tile is a building when its centre lies inside
   one); parks get scattered trees; everything else is open ground.
2. **Connect.** Only the largest connected region of passable tiles is kept.
   Stranded courtyards become building, so a victim can never be placed
   somewhere no route reaches.
3. **Stage the disaster.** Collapsed buildings and rubble beside the
   streets, victims spread across the map (one beside a collapse, trapped),
   and fires next to them. The hospital sits on OSM's own hospital when the
   area has one. A candidate layout is accepted only if every victim is
   reachable from the start. The whole step is seeded, so the same place
   and seed always produce the same disaster.
"""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from sentry_ai.mapping.overpass import BoundingBox

ROAD, BUILDING, OPEN, TREE, COLLAPSED, RUBBLE = "=", "#", ".", "t", "x", "r"

#: OSM highway types a rescue vehicle can drive.
DRIVABLE = frozenset(
    {
        "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
        "residential", "service", "living_street", "road", "pedestrian",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    }
)  # fmt: skip

_PASSABLE = frozenset({ROAD, OPEN})
_METRES_PER_DEGREE_LAT = 111_320.0
_STEPS = ((0, 1), (1, 0), (0, -1), (-1, 0))

Grid = list[list[str]]
Tile = tuple[int, int]


@dataclass(frozen=True)
class DisasterSpec:
    """How much disaster to stage on an imported street plan."""

    victims: int = 4
    fires: int = 2
    collapses: int = 3
    min_victim_distance: int = 6
    attempts: int = 200
    #: Farthest a victim may be from the hospital, in tiles; ``None`` = anywhere.
    #: On a city-sized map this keeps a mission within one battery charge.
    max_victim_distance: int | None = None
    #: Share of the map that must be one connected street network. A city grid
    #: of 20 m tiles is mostly buildings, so large maps lower this.
    min_passable_share: float = 0.25


def box_around(lat: float, lon: float, width: int, height: int, tile_metres: float) -> BoundingBox:
    """The lat/lon box a ``width`` x ``height`` tile grid centred on a point covers."""
    half_lat = height * tile_metres / 2 / _METRES_PER_DEGREE_LAT
    half_lon = width * tile_metres / 2 / (_METRES_PER_DEGREE_LAT * math.cos(math.radians(lat)))
    return BoundingBox(lat - half_lat, lon - half_lon, lat + half_lat, lon + half_lon)


def rasterise(ways: Iterable[dict[str, Any]], box: BoundingBox, width: int, height: int) -> Grid:
    """Paint roads, buildings and parks onto a ``height`` x ``width`` grid."""
    grid: Grid = [[OPEN] * width for _ in range(height)]
    ways = list(ways)

    def project(point: dict[str, float]) -> tuple[float, float]:
        x = (point["lon"] - box.west) / (box.east - box.west) * width
        y = (box.north - point["lat"]) / (box.north - box.south) * height
        return x, y

    for way in ways:
        tags = way.get("tags", {})
        points = [project(p) for p in way.get("geometry", [])]
        if len(points) < 2:
            continue
        if tags.get("building"):
            _fill(grid, points, BUILDING, only_on={OPEN, TREE})
        elif tags.get("leisure") == "park" or tags.get("landuse") == "grass":
            _fill(grid, points, TREE, only_on={OPEN}, sparse=True)
    for way in ways:
        if way.get("tags", {}).get("highway") in DRIVABLE:
            _polyline(grid, [project(p) for p in way.get("geometry", [])])
    return grid


def connect(grid: Grid) -> Grid:
    """Keep the largest connected passable region; wall off the rest."""
    region = _largest_region(grid)
    return [
        [
            cell if cell not in _PASSABLE or (x, y) in region else BUILDING
            for x, cell in enumerate(row)
        ]
        for y, row in enumerate(grid)
    ]


def infer_blocks(grid: Grid, setback: int = 2) -> Grid:
    """Fill open land ``setback`` or more tiles from any road with building.

    For areas where OpenStreetMap has the street network but few building
    footprints — common outside city centres. The land between streets in a
    town is built up; leaving it open gives a city with nothing to burn or
    collapse. Tiles within ``setback - 1`` of a road stay open (the verge a
    vehicle can pull onto). Callers must say the blocks are inferred.
    """
    height, width = len(grid), len(grid[0])
    roads = [(x, y) for y, row in enumerate(grid) for x, cell in enumerate(row) if cell == ROAD]
    distance = {tile: 0 for tile in roads}
    queue = deque(roads)
    while queue:
        x, y = queue.popleft()
        for dx, dy in _STEPS:
            tile = (x + dx, y + dy)
            if 0 <= tile[0] < width and 0 <= tile[1] < height and tile not in distance:
                distance[tile] = distance[(x, y)] + 1
                queue.append(tile)
    return [
        [BUILDING if cell == OPEN and distance.get((x, y), setback) >= setback else cell
         for x, cell in enumerate(row)]
        for y, row in enumerate(grid)
    ]  # fmt: skip


def hospital_tile(
    elements: Iterable[dict[str, Any]], box: BoundingBox, width: int, height: int
) -> Tile | None:
    """The tile of the first hospital OSM knows inside ``box``, if any."""
    for element in elements:
        if element.get("tags", {}).get("amenity") != "hospital":
            continue
        points = element.get("geometry") or ([element] if "lat" in element else [])
        if not points:
            continue
        lat = sum(p["lat"] for p in points) / len(points)
        lon = sum(p["lon"] for p in points) / len(points)
        x = int((lon - box.west) / (box.east - box.west) * width)
        y = int((box.north - lat) / (box.north - box.south) * height)
        if 0 <= x < width and 0 <= y < height:
            return x, y
    return None


def stage_disaster(
    grid: Grid, rng: random.Random, spec: DisasterSpec, hospital: Tile | None = None
) -> dict[str, Any]:
    """Place the hospital, start, collapses, victims and fires; return map-config entries.

    Raises:
        ValueError: If no layout with every victim reachable is found.
    """
    height, width = len(grid), len(grid[0])
    passable = sorted(_largest_region(grid))
    if len(passable) < width * height * spec.min_passable_share:
        raise ValueError("too little open street in this area for a mission; try another place")
    home = _nearest(passable, hospital or (1, 1))
    start = _nearest([t for t in passable if _distance(t, home) >= 2], home)
    for _ in range(spec.attempts):
        staged = [row[:] for row in grid]
        collapses = _collapse(staged, rng, spec.collapses, home)
        victims = _spread(staged, rng, spec, home, collapses)
        fires = _fires(staged, rng, spec.fires, victims)
        if victims and _all_reachable(staged, start, victims, fires, home):
            return _entries(staged, home, start, victims, fires)
    raise ValueError("could not stage a disaster with every victim reachable; try another seed")


# ----------------------------------------------------------------------
# Rasterising
# ----------------------------------------------------------------------


def _polyline(grid: Grid, points: Sequence[tuple[float, float]]) -> None:
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0)) * 3))
        previous: Tile | None = None
        for i in range(steps + 1):
            tile = (int(x0 + (x1 - x0) * i / steps), int(y0 + (y1 - y0) * i / steps))
            if previous and tile[0] != previous[0] and tile[1] != previous[1]:
                _paint(grid, (tile[0], previous[1]), ROAD)  # keep diagonals 4-connected
            _paint(grid, tile, ROAD)
            previous = tile


def _fill(
    grid: Grid,
    polygon: Sequence[tuple[float, float]],
    glyph: str,
    only_on: set[str],
    sparse: bool = False,
) -> None:
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    for y in range(max(0, int(min(ys))), min(len(grid), int(max(ys)) + 1)):
        for x in range(max(0, int(min(xs))), min(len(grid[0]), int(max(xs)) + 1)):
            if sparse and (x + y) % 3:
                continue
            if grid[y][x] in only_on and _inside(x + 0.5, y + 0.5, polygon):
                grid[y][x] = glyph


def _inside(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Ray casting point-in-polygon."""
    inside = False
    for (x0, y0), (x1, y1) in zip(polygon, [*polygon[1:], polygon[0]], strict=True):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _paint(grid: Grid, tile: Tile, glyph: str) -> None:
    x, y = tile
    if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
        grid[y][x] = glyph


# ----------------------------------------------------------------------
# Staging
# ----------------------------------------------------------------------


def _collapse(grid: Grid, rng: random.Random, count: int, home: Tile) -> list[Tile]:
    """Turn building tiles beside the street into collapsed buildings with rubble."""
    candidates = [
        (x, y)
        for y, row in enumerate(grid)
        for x, cell in enumerate(row)
        if cell == BUILDING and _distance((x, y), home) > 5 and _touches(grid, (x, y), _PASSABLE)
    ]
    chosen = rng.sample(candidates, min(count, len(candidates)))
    for x, y in chosen:
        grid[y][x] = COLLAPSED
        behind = [(x + dx, y + dy) for dx, dy in _STEPS if _at(grid, (x + dx, y + dy)) == BUILDING]
        if behind:
            _paint(grid, rng.choice(behind), RUBBLE)
    return chosen


def _spread(
    grid: Grid, rng: random.Random, spec: DisasterSpec, home: Tile, collapses: list[Tile]
) -> list[Tile]:
    """Victims beside damaged or standing buildings, far from home and from each other."""
    region = _largest_region(grid)
    beside = [t for t in region if _touches(grid, t, {BUILDING, COLLAPSED, RUBBLE})]
    beside = [t for t in beside if _distance(t, home) >= spec.min_victim_distance]
    if spec.max_victim_distance is not None:
        beside = [t for t in beside if _distance(t, home) <= spec.max_victim_distance]
    victims: list[Tile] = []
    near_collapse = [t for t in beside if any(_distance(t, c) <= 1 for c in collapses)]
    if near_collapse:
        victims.append(rng.choice(near_collapse))
    while len(victims) < spec.victims and beside:
        pool = rng.sample(beside, min(40, len(beside)))
        best = max(pool, key=lambda t: min([_distance(t, v) for v in victims] or [99]))
        if victims and min(_distance(best, v) for v in victims) < 4:
            break
        victims.append(best)
    return victims


def _fires(grid: Grid, rng: random.Random, count: int, victims: list[Tile]) -> list[Tile]:
    """Fires in buildings two or three tiles from a victim: threatening, not sealing."""
    fires: list[Tile] = []
    for victim in rng.sample(victims, min(count, len(victims))):
        near = [
            (x, y)
            for y, row in enumerate(grid)
            for x, cell in enumerate(row)
            if cell == BUILDING and 2 <= _distance((x, y), victim) <= 3
        ]
        if near:
            fires.append(rng.choice(near))
    return fires


def _all_reachable(
    grid: Grid, start: Tile, victims: list[Tile], fires: list[Tile], home: Tile
) -> bool:
    """Whether every victim and the hospital can be reached with fires burning at radius 1."""
    burning = {(fx + dx, fy + dy) for fx, fy in fires for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    reach = _flood(grid, start, blocked=burning)
    return all(v in reach for v in victims) and home in reach


def _entries(
    grid: Grid, home: Tile, start: Tile, victims: list[Tile], fires: list[Tile]
) -> dict[str, Any]:
    return {
        "grid": ["".join(row) for row in grid],
        "safe_zone": {"position": list(home), "radius": 2, "capacity": 4},
        "vehicle_start": list(start),
        "victims": [
            {"id": f"victim_{i:02d}", "position": list(v)} for i, v in enumerate(victims, 1)
        ],
        "fires": [
            {"id": f"fire_{i:02d}", "position": list(f), "intensity": 0.6, "radius": 1}
            for i, f in enumerate(fires, 1)
        ],
    }


# ----------------------------------------------------------------------
# Grid helpers
# ----------------------------------------------------------------------


def _largest_region(grid: Grid) -> set[Tile]:
    seen: set[Tile] = set()
    best: set[Tile] = set()
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell in _PASSABLE and (x, y) not in seen:
                region = _flood(grid, (x, y))
                seen |= region
                best = max(best, region, key=len)
    return best


def _flood(grid: Grid, origin: Tile, blocked: set[Tile] | None = None) -> set[Tile]:
    blocked = blocked or set()
    reached, queue = {origin}, deque([origin])
    while queue:
        x, y = queue.popleft()
        for dx, dy in _STEPS:
            tile = (x + dx, y + dy)
            if tile not in reached and _at(grid, tile) in _PASSABLE and tile not in blocked:
                reached.add(tile)
                queue.append(tile)
    return reached


def _at(grid: Grid, tile: Tile) -> str | None:
    x, y = tile
    return grid[y][x] if 0 <= y < len(grid) and 0 <= x < len(grid[0]) else None


def _touches(grid: Grid, tile: Tile, glyphs: set[str] | frozenset[str]) -> bool:
    return any(_at(grid, (tile[0] + dx, tile[1] + dy)) in glyphs for dx, dy in _STEPS)


def _nearest(tiles: Sequence[Tile], target: Tile) -> Tile:
    return min(tiles, key=lambda t: (_distance(t, target), t))


def _distance(a: Tile, b: Tile) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
