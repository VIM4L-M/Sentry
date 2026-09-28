"""Cars and pedestrians sharing the streets with the rescue vehicle (Phase 9).

Two pieces:

* :class:`TrafficProcess` — an :class:`IWorldProcess` that spawns road
  users on the first tick and moves them every tick after. Cars drive the
  road tiles, going straight and sometimes turning at junctions; pedestrians
  walk the pavements and now and then step out to cross. Every agent gives
  way to the rescue vehicle's own tile, the way traffic pulls over for an
  ambulance, and a car blocked for a while turns or backs away so no two
  road users can deadlock. It never reports a :class:`WorldChange`: road
  users are not part of the command center's map, so they never trigger a
  replan.
* :class:`TrafficAwarePhysics` — wraps the engine's physics grid so an
  agent's tile is solid. Driving into a car or a person is a collision,
  exactly like driving into debris.

Deterministic given the seed, like the hazards.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.schema import TrafficConfig
from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import Heading, TerrainType
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.domain.traffic import AgentKind, TrafficAgent
from sentry_ai.interfaces.world import IOccupancyGridSource, IWorldProcess, WorldChange

logger = get_logger(__name__)

#: Terrain each kind of road user may stand on.
_CAR_GROUND = frozenset({TerrainType.ROAD})
_WALK_GROUND = frozenset({TerrainType.OPEN_GROUND, TerrainType.ROAD})

#: A car blocked for this many attempts will turn or back away.
_PATIENCE = 3

#: Chance a car at a junction turns rather than going straight.
_TURN_CHANCE = 0.3

#: Tiles kept clear around the vehicle's start when spawning.
_SPAWN_CLEARANCE = 3

_HEADINGS = (Heading.NORTH, Heading.EAST, Heading.SOUTH, Heading.WEST)

Tile = tuple[int, int]


class TrafficProcess(IWorldProcess):
    """Spawns and moves the city's cars and pedestrians."""

    def __init__(self, config: TrafficConfig) -> None:
        """Create the process; agents are spawned on the first :meth:`advance`."""
        self._config = config
        self._rng = random.Random(config.seed)
        self._clock = 0
        self._spawned = False
        self._occupied: dict[Tile, TrafficAgent] = {}

    @property
    def clock(self) -> int:
        """Ticks advanced so far; agents' ``moved_at`` counts in these."""
        return self._clock

    @property
    def config(self) -> TrafficConfig:
        """The traffic settings this process runs with."""
        return self._config

    def step_ticks(self, agent: TrafficAgent) -> int:
        """How many ticks one move of ``agent`` spans, for drawing it gliding."""
        if agent.kind is AgentKind.CAR:
            return self._config.car_step_ticks
        return self._config.pedestrian_step_ticks

    def occupants(self) -> dict[Tile, AgentKind]:
        """Every tile a road user stands on, and what stands there."""
        return {tile: agent.kind for tile, agent in self._occupied.items()}

    def advance(self, city_map: CityMap, delta_seconds: float) -> WorldChange:
        """Move every agent whose turn it is. Never changes the planner's map."""
        if not self._config.enabled:
            return WorldChange.none()
        if not self._spawned:
            self._spawn(city_map)
        self._clock += 1
        vehicle = city_map.vehicle.position.as_tuple()
        for agent in city_map.traffic:
            if self._clock - agent.moved_at >= self.step_ticks(agent):
                self._move(city_map, agent, vehicle)
        return WorldChange.none()

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    def _spawn(self, city_map: CityMap) -> None:
        self._spawned = True
        start = city_map.vehicle.position
        roads = [
            p for p in _tiles(city_map, _CAR_GROUND) if p.distance_to(start) > _SPAWN_CLEARANCE
        ]
        pavement = [
            p
            for p in _tiles(city_map, frozenset({TerrainType.OPEN_GROUND}))
            if p.distance_to(start) > _SPAWN_CLEARANCE and _beside_road(city_map, p)
        ] or roads
        self._rng.shuffle(roads)
        self._rng.shuffle(pavement)
        for index, position in enumerate(roads[: self._config.cars]):
            self._add(city_map, f"car_{index:03d}", AgentKind.CAR, position)
        taken = {p.as_tuple() for p in roads[: self._config.cars]}
        walkers = [p for p in pavement if p.as_tuple() not in taken]
        for index, position in enumerate(walkers[: self._config.pedestrians]):
            self._add(city_map, f"person_{index:03d}", AgentKind.PEDESTRIAN, position)
        logger.info(
            "Traffic: %d cars, %d pedestrians",
            sum(a.kind is AgentKind.CAR for a in city_map.traffic),
            sum(a.kind is AgentKind.PEDESTRIAN for a in city_map.traffic),
        )

    def _add(self, city_map: CityMap, agent_id: str, kind: AgentKind, position: Position) -> None:
        agent = TrafficAgent(
            agent_id=agent_id,
            kind=kind,
            position=position,
            heading=self._rng.choice(_HEADINGS),
            previous=position,
            moved_at=-self._rng.randrange(8),  # stagger, so they do not all step together
        )
        city_map.traffic.append(agent)
        self._occupied[position.as_tuple()] = agent

    # ------------------------------------------------------------------
    # Moving
    # ------------------------------------------------------------------

    def _move(self, city_map: CityMap, agent: TrafficAgent, vehicle: Tile) -> None:
        for heading in self._choices(city_map, agent):
            dx, dy = heading.delta
            target = (agent.position.x + dx, agent.position.y + dy)
            if target == vehicle or target in self._occupied:
                continue
            if not self._may_enter(city_map, agent, target):
                continue
            del self._occupied[agent.position.as_tuple()]
            agent.previous = agent.position
            agent.position = Position(*target)
            agent.heading = heading
            agent.moved_at = self._clock
            agent.waiting = 0
            self._occupied[target] = agent
            return
        agent.waiting += 1
        agent.previous = agent.position

    def _choices(self, city_map: CityMap, agent: TrafficAgent) -> list[Heading]:
        """Headings to try, best first."""
        ahead = agent.heading
        sides = [ahead.turn_left(), ahead.turn_right()]
        self._rng.shuffle(sides)
        back = ahead.turn_left().turn_left()
        if agent.kind is AgentKind.PEDESTRIAN:
            if self._rng.random() < 0.25:
                return [*sides, ahead, back]
            return [ahead, *sides, back]
        if agent.waiting >= _PATIENCE:
            return [*sides, back]
        junction = sum(
            _is(city_map, agent.position.x + h.delta[0], agent.position.y + h.delta[1], _CAR_GROUND)
            for h in _HEADINGS
        )
        if junction >= 3 and self._rng.random() < _TURN_CHANCE:
            return [*sides, ahead]
        return [ahead, *sides] if agent.waiting == 0 else [ahead]

    def _may_enter(self, city_map: CityMap, agent: TrafficAgent, target: Tile) -> bool:
        x, y = target
        if agent.kind is AgentKind.CAR:
            return _is(city_map, x, y, _CAR_GROUND)
        if not _is(city_map, x, y, _WALK_GROUND):
            return False
        here_road = city_map.tile_at(agent.position) is TerrainType.ROAD
        there_road = city_map.tile_at(Position(x, y)) is TerrainType.ROAD
        if there_road and not here_road:
            return self._rng.random() < self._config.crossing_chance
        return True


class TrafficAwarePhysics(IOccupancyGridSource):
    """The engine's physics grid with every road user's tile made solid."""

    def __init__(self, inner: IOccupancyGridSource, traffic: TrafficProcess) -> None:
        """Wrap ``inner``, which supplies everything but the road users."""
        self._inner = inner
        self._traffic = traffic

    def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
        """A copy of the inner grid with each agent's tile marked as debris."""
        grid = OccupancyGrid(cells=self._inner.grid_for(city_map, vehicle_position).cells.copy())
        for x, y in self._traffic.occupants():
            grid.cells[y, x] = OccupancyCode.DEBRIS
        return grid


def _tiles(city_map: CityMap, ground: frozenset[TerrainType]) -> list[Position]:
    return list(_iter_tiles(city_map, ground))


def _iter_tiles(city_map: CityMap, ground: frozenset[TerrainType]) -> Iterator[Position]:
    for y in range(city_map.height):
        for x in range(city_map.width):
            position = Position(x, y)
            if city_map.tile_at(position) in ground:
                yield position


def _is(city_map: CityMap, x: int, y: int, ground: frozenset[TerrainType]) -> bool:
    if not (0 <= x < city_map.width and 0 <= y < city_map.height):
        return False
    return city_map.tile_at(Position(x, y)) in ground


def _beside_road(city_map: CityMap, position: Position) -> bool:
    return any(
        _is(city_map, position.x + h.delta[0], position.y + h.delta[1], _CAR_GROUND)
        for h in _HEADINGS
    )
