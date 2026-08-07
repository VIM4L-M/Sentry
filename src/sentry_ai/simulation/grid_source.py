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
