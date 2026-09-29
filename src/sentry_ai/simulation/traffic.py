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

#: An agent blocked this many attempts right beside the rescue vehicle gives
#: way (see ``TrafficProcess._give_way``).
_GIVE_WAY = 6

#: How far from the vehicle an agent that left the street rejoins traffic.
_REJOIN_DISTANCE = 15

#: Chance a car at a junction turns rather than going straight.
_TURN_CHANCE = 0.3

#: Tiles kept clear around the vehicle's start when spawning.
_SPAWN_CLEARANCE = 3

_HEADINGS = (Heading.NORTH, Heading.EAST, Heading.SOUTH, Heading.WEST)

Tile = tuple[int, int]


class TrafficProcess(IWorldProcess):
    """Spawns and moves the city's road users."""

    def __init__(self, config: TrafficConfig) -> None:
        """Create the process; agents are spawned on the first :meth:`advance`."""
        self._config = config
        self._rng = random.Random(config.seed)
        self._clock = 0
        self._spawned = False
        self._occupied: dict[Tile, TrafficAgent] = {}
        self._roads: list[Position] = []

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
        return self.step_ticks_by_kind()[agent.kind]

    def step_ticks_by_kind(self) -> dict[AgentKind, int]:
        """Ticks per move for every kind of road user."""
        config = self._config
        return {
            AgentKind.CAR: config.car_step_ticks,
            AgentKind.AUTO_RICKSHAW: config.auto_step_ticks,
            AgentKind.TWO_WHEELER: config.two_wheeler_step_ticks,
            AgentKind.PEDESTRIAN: config.pedestrian_step_ticks,
            AgentKind.COW: config.cow_step_ticks,
        }

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
        self._roads = _tiles(city_map, _CAR_GROUND)
        roads = [p for p in self._roads if p.distance_to(start) > _SPAWN_CLEARANCE]
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
        self._spawn_indian_traffic(city_map, roads[self._config.cars :], walkers)
        counts = {kind: 0 for kind in AgentKind}
        for agent in city_map.traffic:
            counts[agent.kind] += 1
        logger.info(
            "Traffic: %s", ", ".join(f"{n} {kind.value}" for kind, n in counts.items() if n)
        )

    def _spawn_indian_traffic(
        self, city_map: CityMap, free_roads: list[Position], walkers: list[Position]
    ) -> None:
        """Autorickshaws, two-wheelers and cattle, on tiles no one else took.

        Drawn from the already-shuffled lists after the cars and pedestrians,
        so a config without them spawns exactly what it did before.
        """
        taken = {agent.position.as_tuple() for agent in city_map.traffic}
        roads = [p for p in free_roads if p.as_tuple() not in taken]
        config = self._config
        for kind, count, prefix in (
            (AgentKind.AUTO_RICKSHAW, config.autos, "auto"),
            (AgentKind.TWO_WHEELER, config.two_wheelers, "bike"),
        ):
            for index, position in enumerate(roads[:count]):
                self._add(city_map, f"{prefix}_{index:03d}", kind, position)
            roads = roads[count:]
        grazing = roads + [p for p in walkers if p.as_tuple() not in self._occupied]
        for index, position in enumerate(grazing[: config.cows]):
            if position.as_tuple() not in self._occupied:
                self._add(city_map, f"cow_{index:03d}", AgentKind.COW, position)

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
        if agent.waiting >= _GIVE_WAY and agent.position.distance_to(Position(*vehicle)) <= 2:
            self._give_way(city_map, agent, vehicle)

    def _give_way(self, city_map: CityMap, agent: TrafficAgent, vehicle: Tile) -> None:
        """Clear the rescue vehicle's way when this agent is stuck beside it.

        The emergency brake rightly refuses to drive into a road user, so a
        car boxed in on a one-lane street in front of the vehicle, with
        nowhere to turn, held the vehicle there for the rest of the mission.
        Real traffic mounts the kerb or pulls into a driveway for an
        ambulance: the agent steps onto any free neighbouring pavement or
        road tile; failing that, it leaves the street and rejoins the
        traffic far away.
        """
        for heading in _HEADINGS:
            dx, dy = heading.delta
            target = (agent.position.x + dx, agent.position.y + dy)
            if target == vehicle or target in self._occupied:
                continue
            if _is(city_map, target[0], target[1], _WALK_GROUND):
                self._relocate(agent, Position(*target), heading, glide=True)
                return
        far = [
            p
            for p in self._roads
            if p.as_tuple() not in self._occupied
            and p.distance_to(Position(*vehicle)) > _REJOIN_DISTANCE
        ]
        if far:
            self._relocate(agent, self._rng.choice(far), agent.heading, glide=False)

    def _relocate(
        self, agent: TrafficAgent, target: Position, heading: Heading, glide: bool
    ) -> None:
        del self._occupied[agent.position.as_tuple()]
        agent.previous = agent.position if glide else target
        agent.position = target
        agent.heading = heading
        agent.moved_at = self._clock
        agent.waiting = 0
        self._occupied[target.as_tuple()] = agent

    def _choices(self, city_map: CityMap, agent: TrafficAgent) -> list[Heading]:
        """Headings to try, best first."""
        ahead = agent.heading
        sides = [ahead.turn_left(), ahead.turn_right()]
        self._rng.shuffle(sides)
        back = ahead.turn_left().turn_left()
        if not agent.kind.is_vehicle:
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
        if agent.kind.is_vehicle:
            return _is(city_map, x, y, _CAR_GROUND)
        if not _is(city_map, x, y, _WALK_GROUND):
            return False
        if agent.kind is AgentKind.COW:
            return True  # cattle wander onto the road with no regard for the kerb
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
