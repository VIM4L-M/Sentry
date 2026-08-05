"""Hazards that change the city while the vehicle is driving through it.

Two processes, deliberately given different jobs so their effects can be
reasoned about separately:

* :class:`FireSpreadProcess` — fire grows, creeps through fuel, and burns
  itself out. It only takes hold on flammable terrain (buildings, trees,
  rubble), never on roads, so it threatens *victims* and raises the cost of
  routes near it without ever severing the road network outright.
* :class:`DebrisCollapseProcess` — a weakened facade drops into the street
  beside it. This is the specification's dynamic obstacle: something that
  appears mid-mission on a tile the planner already believed was clear, and
  the thing that actually forces replanning.
* :class:`VictimRiskProcess` — trapped victims lose health while they wait,
  faster beside a fire. This is what turns "which victim next?" into a real
  decision instead of a lookup of the nearest one.

Both are seeded and deterministic. Given the same config and the same tick
sequence a mission produces byte-identical hazards, which is what makes
Phase 6's training runs comparable and today's bug reports reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.schema import (
    DebrisCollapseConfig,
    FireSpreadConfig,
    HazardConfig,
    VictimRiskConfig,
)
from sentry_ai.domain.entities import FireSource, Obstacle, Position, Victim
from sentry_ai.domain.enums import TerrainType, VictimStatus
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import tiles_within
from sentry_ai.interfaces.world import IWorldProcess, WorldChange

logger = get_logger(__name__)

#: Terrain that will carry a fire: a standing building full of contents, or
#: a tree. Everything else is deliberately absent.
#:
#: Roads and open ground, because a fire that crossed tarmac would sever the
#: road network and turn every mission into a coin flip.
#:
#: Rubble and collapsed buildings, because masonry does not burn — and
#: because letting it burn had a consequence worth spelling out. A trapped
#: victim's own tile survives being set alight (the ``VICTIM`` code outranks
#: ``FIRE`` on the grid) but the tiles *around* them do not, so a fire
#: taking hold in a rubble field walls the victim off completely. That is
#: how victim_02, who lies under rubble beside more rubble, became
#: unreachable through no decision the vehicle made.
_FLAMMABLE_TERRAIN = frozenset({TerrainType.BUILDING, TerrainType.TREE})

#: Intensity a newly-ignited fire starts at — high enough to survive its
#: first burn-out check, low enough that it starts as a single tile.
_IGNITION_INTENSITY = 0.2

_NEIGHBOUR_OFFSETS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def orthogonal_neighbours(position: Position) -> Iterator[Position]:
    """The four orthogonally adjacent tiles, skipping negative coordinates.

    ``Position`` rejects negative coordinates, so the filter has to happen
    on raw integers before construction — the same rule the planner and the
    vehicle controller follow at the map edge.
    """
    for dx, dy in _NEIGHBOUR_OFFSETS:
        x, y = position.x + dx, position.y + dy
        if x >= 0 and y >= 0:
            yield Position(x, y)


class _PeriodicProcess(IWorldProcess):
    """Base for hazards that act on a fixed cadence rather than every tick.

    Accumulates simulated time and fires :meth:`step` once per interval, so
    a hazard's pace is set by ``interval_seconds`` and not by the tick rate.
    """

    def __init__(self, interval_seconds: float, enabled: bool) -> None:
        self._interval_seconds = interval_seconds
        self._enabled = enabled
        self._elapsed_seconds = 0.0

    def advance(self, city_map: CityMap, delta_seconds: float) -> WorldChange:
        """Accumulate time and run one :meth:`step` per elapsed interval."""
        if not self._enabled:
            return WorldChange.none()
        self._elapsed_seconds += delta_seconds
        if self._elapsed_seconds < self._interval_seconds:
            return WorldChange.none()
        self._elapsed_seconds -= self._interval_seconds
        return self.step(city_map)

    def step(self, city_map: CityMap) -> WorldChange:
        """Perform one cadence step. Overridden by concrete hazards."""
        raise NotImplementedError


class FireSpreadProcess(_PeriodicProcess):
    """A cellular automaton over the city's fires.

    Each step every fire grows toward full intensity, then burns out and
    disappears. A fire's footprint scales with its intensity, and a fire may
    ignite one flammable tile just outside that footprint per step.
    """

    def __init__(self, config: FireSpreadConfig, rng: random.Random) -> None:
        """Create the process.

        Args:
            config: Growth, spread, and burn-out rates.
            rng: Seeded generator. Owned by the caller so several processes
                can be given independent, reproducible streams.
        """
        super().__init__(config.interval_seconds, config.enabled)
        self._config = config
        self._rng = rng
        self._burning_out: set[str] = set()
        self._ignitions = 0

    def step(self, city_map: CityMap) -> WorldChange:
        """Grow and burn out existing fires, then try to spread them."""
        changed = self._evolve_existing(city_map) | self._spread(city_map)
        if not changed:
            return WorldChange.none()
        return WorldChange(
            changed_tiles=frozenset(changed),
            description=f"fire changed {len(changed)} tile(s)",
        )

    # ------------------------------------------------------------------
    # Growth and burn-out
    # ------------------------------------------------------------------

    def _evolve_existing(self, city_map: CityMap) -> set[Position]:
        """Advance every fire's lifecycle, dropping the ones that died."""
        changed: set[Position] = set()
        survivors: list[FireSource] = []

        for fire in city_map.fires:
            before = self._footprint(city_map, fire)
            intensity = self._next_intensity(fire)
            if intensity <= 0.0:
                self._burning_out.discard(fire.fire_id)
                changed |= before
                logger.info("Fire %s burned out", fire.fire_id)
                continue
            fire.intensity = intensity
            fire.radius = self._radius_for(intensity)
            survivors.append(fire)
            changed |= before ^ self._footprint(city_map, fire)

        city_map.fires = survivors
        return changed

    def _next_intensity(self, fire: FireSource) -> float:
        """This fire's intensity after one step of its lifecycle.

        A fire grows until it peaks at ``1.0``, then decays. Returning a
        non-positive value means the fire is out.
        """
        if fire.fire_id in self._burning_out:
            return round(fire.intensity - self._config.burnout_per_step, 4)
        grown = fire.intensity + self._config.growth_per_step
        if grown >= 1.0:
            self._burning_out.add(fire.fire_id)
            return 1.0
        return round(grown, 4)

    def _radius_for(self, intensity: float) -> int:
        """Footprint radius for an intensity, from 1 tile up to ``max_radius``."""
        span = self._config.max_radius - 1
        return 1 + int(round(intensity * span))

    @staticmethod
    def _footprint(city_map: CityMap, fire: FireSource) -> set[Position]:
        """Every in-bounds tile this fire currently covers."""
        return set(tiles_within(fire.position, fire.radius, city_map.width, city_map.height))

    # ------------------------------------------------------------------
    # Spread
    # ------------------------------------------------------------------

    def _spread(self, city_map: CityMap) -> set[Position]:
        """Give every fire one chance to ignite a tile beside it."""
        changed: set[Position] = set()
        for fire in list(city_map.fires):
            if len(city_map.fires) >= self._config.max_active_fires:
                break
            threshold = self._config.ignition_chance * fire.intensity
            for target in self._ignitable_frontier(city_map, fire):
                if self._rng.random() < threshold:
                    changed |= self._ignite(city_map, target)
                    break
        return changed

    def _ignitable_frontier(self, city_map: CityMap, fire: FireSource) -> list[Position]:
        """Flammable tiles immediately outside ``fire``'s footprint.

        Sorted row-major so the sequence of random draws — and therefore the
        whole mission — is reproducible.
        """
        footprint = self._footprint(city_map, fire)
        already_burning = {source.position for source in city_map.fires}
        frontier = {
            neighbour
            for tile in footprint
            for neighbour in orthogonal_neighbours(tile)
            if neighbour not in footprint and neighbour not in already_burning
        }
        ignitable = (tile for tile in frontier if self._is_ignitable(city_map, tile))
        return sorted(ignitable, key=lambda tile: (tile.y, tile.x))

    @staticmethod
    def _is_ignitable(city_map: CityMap, position: Position) -> bool:
        """Whether fire can take hold here: in bounds, fuelled, not the hospital."""
        return (
            city_map.in_bounds(position)
            and not city_map.safe_zone.contains(position)
            and city_map.tile_at(position) in _FLAMMABLE_TERRAIN
        )

    def _ignite(self, city_map: CityMap, position: Position) -> set[Position]:
        """Start a new fire at ``position`` and return the tiles it covers."""
        self._ignitions += 1
        fire = FireSource(
            fire_id=f"fire_spread_{self._ignitions:03d}",
            position=position,
            intensity=_IGNITION_INTENSITY,
            radius=1,
        )
        city_map.fires.append(fire)
        logger.info("Fire spread to %s as %s", position.as_tuple(), fire.fire_id)
        return self._footprint(city_map, fire)


class DebrisCollapseProcess(_PeriodicProcess):
    """Drops debris from a weakened building into the street beside it.

    Collapse sites are always clear tiles adjacent to a standing building —
    debris has to fall off something — and never the vehicle's tile, a
    trapped victim's tile, or the hospital, none of which would be a
    recoverable mission state.
    """

    def __init__(self, config: DebrisCollapseConfig, rng: random.Random) -> None:
        """Create the process.

        Args:
            config: Cadence, probability, and per-mission collapse budget.
            rng: Seeded generator, owned by the caller.
        """
        super().__init__(config.interval_seconds, config.enabled)
        self._config = config
        self._rng = rng
        self._collapses = 0

    def step(self, city_map: CityMap) -> WorldChange:
        """Possibly drop debris onto one eligible tile."""
        if self._collapses >= self._config.max_collapses:
            return WorldChange.none()
        if self._rng.random() >= self._config.collapse_chance:
            return WorldChange.none()

        candidates = self._candidates(city_map)
        if not candidates:
            return WorldChange.none()
        return self._collapse_onto(city_map, self._rng.choice(candidates))

    def _collapse_onto(self, city_map: CityMap, position: Position) -> WorldChange:
        """Turn ``position`` into impassable debris and report the change."""
        self._collapses += 1
        city_map.terrain[position] = TerrainType.COLLAPSED_BUILDING
        city_map.obstacles.append(
            Obstacle(
                obstacle_id=f"collapse_{self._collapses:03d}",
                position=position,
                kind=TerrainType.COLLAPSED_BUILDING,
            )
        )
        logger.warning("Building collapsed onto %s", position.as_tuple())
        return WorldChange(
            changed_tiles=frozenset({position}),
            description=f"collapse at {position.as_tuple()}",
        )

    def _candidates(self, city_map: CityMap) -> list[Position]:
        """Every tile debris may legally fall on, in row-major order."""
        occupied = {city_map.vehicle.position, city_map.safe_zone.position}
        occupied |= {
            victim.position
            for victim in city_map.victims
            if victim.status is VictimStatus.TRAPPED
        }
        return [
            position
            for position in self._streets_beside_buildings(city_map)
            if position not in occupied and not city_map.safe_zone.contains(position)
        ]

    @staticmethod
    def _streets_beside_buildings(city_map: CityMap) -> list[Position]:
        """Clear tiles that have at least one standing building next to them."""
        tiles: list[Position] = []
        for y in range(city_map.height):
            for x in range(city_map.width):
                position = Position(x, y)
                if city_map.tile_at(position).blocks_movement:
                    continue
                if any(
                    city_map.in_bounds(neighbour)
                    and city_map.tile_at(neighbour) is TerrainType.BUILDING
                    for neighbour in orthogonal_neighbours(position)
                ):
                    tiles.append(position)
        return tiles


class VictimRiskProcess(_PeriodicProcess):
    """Drains trapped victims' health, faster the closer they are to fire.

    This is what makes the command center's objective ordering a real
    decision. Without it, waiting costs a victim nothing, "go to the nearest
    one" is always optimal, and there is no reason for a planner — learned
    or otherwise — to ever weigh one victim against another.

    Deliberately has no random generator: deterioration is a consequence of
    where a victim is, not of luck. Two identical missions must kill exactly
    the same people.
    """

    def __init__(self, config: VictimRiskConfig) -> None:
        """Create the process.

        Args:
            config: Drain rates and how far a fire's effect reaches.
        """
        super().__init__(config.interval_seconds, config.enabled)
        self._config = config

    def step(self, city_map: CityMap) -> WorldChange:
        """Age every trapped victim, and report the ones who did not make it.

        A lost victim's tile is reported as changed so the command center
        replans: a route being driven toward someone who has just died is a
        route worth abandoning immediately.
        """
        lost: set[Position] = set()
        for victim in city_map.victims:
            if not victim.status.is_rescuable:
                continue
            victim.health = max(0, victim.health - self._drain_for(victim, city_map))
            if victim.health == 0:
                victim.status = VictimStatus.LOST
                lost.add(victim.position)
                logger.warning("Lost %s at %s", victim.victim_id, victim.position.as_tuple())

        if not lost:
            return WorldChange.none()
        return WorldChange(
            changed_tiles=frozenset(lost),
            description=f"lost {len(lost)} victim(s)",
        )

    def _drain_for(self, victim: Victim, city_map: CityMap) -> int:
        """Health this victim loses this step, given the nearest fire."""
        return self._config.base_drain + round(
            self._config.fire_drain * self._fire_exposure(victim.position, city_map)
        )

    def _fire_exposure(self, position: Position, city_map: CityMap) -> float:
        """Closeness to the nearest fire, ``1.0`` at its seat and ``0.0`` beyond reach.

        Measured to the fire *source* rather than its drawn footprint, so a
        victim is endangered by heat and smoke before the flames arrive.
        """
        if not city_map.fires:
            return 0.0
        nearest = min(position.distance_to(fire.position) for fire in city_map.fires)
        if nearest >= self._config.fire_radius:
            return 0.0
        return 1.0 - nearest / self._config.fire_radius


def build_world_processes(config: HazardConfig) -> list[IWorldProcess]:
    """Every hazard process a mission runs, in the order the engine drives them.

    Each process gets its own generator, seeded from ``config.seed``, rather
    than sharing one. Sharing would couple them: disabling fire would shift
    every debris draw and silently change an otherwise identical mission.
    """
    seeds = random.Random(config.seed)
    return [
        FireSpreadProcess(config.fire, random.Random(seeds.randrange(2**32))),
        DebrisCollapseProcess(config.debris, random.Random(seeds.randrange(2**32))),
        # Victim risk runs last so it sees the fires this tick produced, and
        # takes no generator at all — deterioration is a consequence of
        # where a victim is, not of luck.
        VictimRiskProcess(config.victims),
    ]
