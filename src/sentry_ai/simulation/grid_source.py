"""The simulator handing the command center its own truth (Phase 2 behaviour).

This is the baseline half of the swap Phase 3 introduces. It wraps what
:class:`~sentry_ai.simulation.mission.MissionController` used to call
directly, so that the call site becomes an injected dependency without the
default behaviour changing at all.

Keeping it is not sentiment. A detector-driven grid is only meaningful next
to the grid it *should* have produced, so this class is simultaneously the
Phase 2 default, the answer key
:mod:`sentry_ai.perception.grid_metrics` scores against, and the control in
any experiment asking whether perception cost the mission anything.
"""

from __future__ import annotations

from collections import deque

from sentry_ai.domain.entities import Position
from sentry_ai.domain.map import CityMap
from sentry_ai.domain.occupancy import OccupancyGrid
from sentry_ai.interfaces.world import IOccupancyGridSource


class GroundTruthGridSource(IOccupancyGridSource):
    """Builds the belief map straight from the simulated world.

    Perfect perception by construction: no camera, no model, no missed
    victim. A mission running on this is measuring its navigation and
    triage logic and nothing else.
    """

    def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
        """Project the whole world, then stamp the vehicle onto it."""
        grid = OccupancyGrid.from_city_map(city_map)
        grid.mark_vehicle(vehicle_position)
        return grid


class LaggedGridSource(IOccupancyGridSource):
    """A belief map that trails reality: the command center's view, delayed.

    In a real disaster the central map is assembled from reports that take
    time to arrive, while the vehicle's own camera sees the present. This
    source models exactly that gap. It hands back the grid its inner source
    produced ``delay`` refreshes ago, so a building that collapsed a moment
    ago is not on the map yet — but it is in front of the vehicle.

    That gap is what Phase 7's decision fusion exists for: the onboard
    camera can see what the command center has not heard about. With the
    ground-truth source and no lag, there is nothing for it to catch.

    The delay is counted in refreshes, not seconds. The mission refreshes
    once per tick, and once more on a tick when a hazard changes the city,
    so ``delay`` refreshes is ``delay`` ticks or slightly fewer.
    """

    def __init__(self, inner: IOccupancyGridSource, delay: int) -> None:
        """Wrap ``inner`` with a lag of ``delay`` refreshes.

        Raises:
            ValueError: If ``delay`` is negative. Zero is allowed and is
                exactly ``inner``.
        """
        if delay < 0:
            raise ValueError(f"delay must be non-negative, got {delay}")
        self._inner = inner
        self._delay = delay
        self._history: deque[OccupancyGrid] = deque(maxlen=delay + 1)

    def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
        """The grid from ``delay`` refreshes ago — or the oldest one held, early on."""
        self._history.append(self._inner.grid_for(city_map, vehicle_position))
        return self._history[0]


class ThrottledGridSource(IOccupancyGridSource):
    """Re-perceives the city only every ``every`` refreshes; reuses the map in between.

    The camera-built map is by far the most expensive thing a tick does:
    four frames through the denoiser and the detector. The mission refreshes
    it every tick — ten times a simulated second — while hazards change the
    city every three seconds. Refreshing every fourth tick keeps the map at
    most 0.4 s old, small against the three-second changes it tracks, and
    cuts the perception cost by four. The vehicle's *own* camera is not
    throttled; decision fusion needs the present.

    ``every=1`` is the inner source, unchanged.
    """

    def __init__(self, inner: IOccupancyGridSource, every: int) -> None:
        """Wrap ``inner``, re-asking it once per ``every`` refreshes.

        Raises:
            ValueError: If ``every`` is not positive.
        """
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")
        self._inner = inner
        self._every = every
        self._calls = 0
        self._cached: OccupancyGrid | None = None

    def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
        """A fresh map on the first call and every ``every``-th after; the cached one otherwise."""
        if self._cached is None or self._calls % self._every == 0:
            self._cached = self._inner.grid_for(city_map, vehicle_position)
        self._calls += 1
        return self._cached
