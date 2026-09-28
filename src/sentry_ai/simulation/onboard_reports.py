"""The vehicle telling the command center what its own camera just saw (Phase 7-8).

When the command center's map lags the world, the vehicle's onboard camera
is the freshest sensor in the system. Fusion uses it to stop the vehicle
driving into a hazard the map has not heard of. Stopping alone is not
enough, though: the route still runs through that tile, so the vehicle
would wait until the map caught up, sometimes while a fire spread onto it.

:class:`OnboardHazardReporter` closes that loop. When fusion overrules the
DQN because the camera sees debris or fire directly ahead, the tile is
reported to the lagging map — overlaid on the stale belief until the map
catches up — and the route is dropped, so the command center replans around
the hazard on the next tick. Real vehicles do the same: local perception
patches the global map faster than the global pipeline can.
"""

from __future__ import annotations

from sentry_ai.domain.entities import Position
from sentry_ai.domain.enums import EntityKind
from sentry_ai.domain.occupancy import OccupancyCode
from sentry_ai.interfaces.decision import Region, SceneEvidence
from sentry_ai.interfaces.navigation import LocalObservation
from sentry_ai.simulation.grid_source import LaggedGridSource
from sentry_ai.simulation.mission import MissionController

#: What each reportable detector class becomes on the map.
_REPORTED: dict[EntityKind, OccupancyCode] = {
    EntityKind.OBSTACLE: OccupancyCode.DEBRIS,
    EntityKind.FIRE: OccupancyCode.FIRE,
}


class OnboardHazardReporter:
    """Turns a fusion override into a map report and a replan."""

    def __init__(
        self, lag: LaggedGridSource, mission: MissionController, threshold: float = 0.5
    ) -> None:
        """Wire the reporter.

        Args:
            lag: The command center's lagging map source, which takes the report.
            mission: Told to drop its route so it replans on the patched map.
            threshold: Minimum camera confidence before a hazard is reported.
        """
        self._lag = lag
        self._mission = mission
        self._threshold = threshold
        self.reports = 0

    def __call__(self, evidence: SceneEvidence, observation: LocalObservation) -> None:
        """Report whatever hazard the camera sees on the tile ahead, if any."""
        dx, dy = observation.heading.delta
        x, y = observation.position.x + dx, observation.position.y + dy
        if x < 0 or y < 0:
            return
        for kind, code in _REPORTED.items():
            if evidence.at(kind, Region.AHEAD) >= self._threshold:
                self._lag.report(Position(x, y), code)
                self._mission.invalidate_route()
                self.reports += 1
                return
