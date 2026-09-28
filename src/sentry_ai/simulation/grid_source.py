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
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
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
    """Hands the command center a map that is a fixed number of refreshes old.

    Models the delay between the world changing and the command center
    hearing about it — CCTV processing, radio, a busy operator. The Phase 7
    scenario depends on it: with a perfectly fresh map the A* planner
    already routes around every hazard, the local controller is never
    asked to drive into one, and a fusion network would have nothing to
    catch. A lag is what opens the gap between *belief* and *truth* that
    the vehicle's own camera can close.

    The mission refreshes its grid once per tick (plus once more on a tick
    where a hazard fires), so ``lag_refreshes`` is close to a lag in ticks.
    A lag of ``0`` is exactly the wrapped source.

    The lag can be changed while a mission runs (:attr:`lag`) — the
    mission-control screen's ``L`` key — up to the ``max_lag`` it was built
    with, which bounds the history kept.
    """

    def __init__(
        self, inner: IOccupancyGridSource, lag_refreshes: int, max_lag: int | None = None
    ) -> None:
        """Wrap ``inner`` so it answers ``lag_refreshes`` calls late.

        Args:
            inner: The source whose answers are delayed.
            lag_refreshes: The starting lag.
            max_lag: The largest lag :attr:`lag` may later be set to.
                Defaults to ``lag_refreshes``.

        Raises:
            ValueError: If a lag is negative or exceeds ``max_lag``.
        """
        self._max_lag = lag_refreshes if max_lag is None else max_lag
        self._inner = inner
        self._history: deque[OccupancyGrid] = deque(maxlen=self._max_lag + 1)
        self._lag = 0
        self.lag = lag_refreshes
        self._reports: dict[Position, tuple[OccupancyCode, int]] = {}

    @property
    def lag(self) -> int:
        """How many refreshes behind the world the answers currently are."""
        return self._lag

    @lag.setter
    def lag(self, value: int) -> None:
        if not 0 <= value <= self._max_lag:
            raise ValueError(f"lag must be within 0-{self._max_lag}, got {value}")
        self._lag = value

    def report(self, position: Position, code: OccupancyCode) -> None:
        """Overlay a fresh local observation — the vehicle's own camera — on the stale map.

        The report stands for as many refreshes as the map lags, by which
        time the lagged map has caught up with the world and the report is
        redundant. It is how onboard sensing reaches the command center
        faster than the CCTV pipeline: the planner routes around a collapse
        the vehicle has just seen instead of waiting for the map.
        """
        self._reports[position] = (code, self._lag + 1)

    @property
    def max_lag(self) -> int:
        """The largest lag this source can be set to."""
        return self._max_lag

    def grid_for(self, city_map: CityMap, vehicle_position: Position) -> OccupancyGrid:
        """The grid from ``lag_refreshes`` calls ago, re-stamped with the vehicle.

        Until enough history exists the oldest grid seen is returned, so the
        first ticks of a mission are merely stale rather than empty. The
        vehicle is always stamped where it is *now* — the command center
        knows where its own vehicle is; what it lags on is the city.
        """
        self._history.append(self._inner.grid_for(city_map, vehicle_position))
        cells = self._history[max(0, len(self._history) - 1 - self._lag)].as_array()
        cells[cells == OccupancyCode.VEHICLE] = OccupancyCode.ROAD
        grid = OccupancyGrid(cells)
        self._apply_reports(grid)
        grid.mark_vehicle(vehicle_position)
        return grid

    def _apply_reports(self, grid: OccupancyGrid) -> None:
        """Stamp live reports onto ``grid`` and age them by one refresh."""
        for position, (code, remaining) in list(self._reports.items()):
            if grid.in_bounds(position):
                grid.mark(position, code)
            if remaining <= 1:
                del self._reports[position]
            else:
                self._reports[position] = (code, remaining - 1)
