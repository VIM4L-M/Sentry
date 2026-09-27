"""Records what the vehicle did, tick by tick, across many missions (Unit III data).

A trajectory is the vehicle's :class:`~sentry_ai.interfaces.sequence.VehicleState`
at mission start and after every tick, until the mission ends. Variety
comes from re-seeding the hazards: every seed is a different disaster, so
the vehicle replans, detours and backs out of different streets.

**Split by mission, as for the detector.** Consecutive windows of one
trajectory overlap by all but one tick. Splitting windows at random would
put near-copies on both sides and measure memorisation; a validation
mission here contributes no window to training.

**Starts are randomised, or validation measures memorisation.** Every
mission on the shipped map starts on the same tile and drives to the same
victims, so the planner produces nearly the same route whatever the hazard
seed. Measured with a fixed start, 99.4% of validation windows were exact
copies of a training window — a model scoring 0.96 there had learned the
routes, not the behaviour. Randomised starts
(:class:`~sentry_ai.training.missions.MissionFactory`) make each mission's
route genuinely its own.

Stored as JSON: small (a few hundred kilobytes for hundreds of missions),
diffable, and readable without this codebase.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.common.logging_config import get_logger
from sentry_ai.domain.entities import Position, Vehicle
from sentry_ai.interfaces.sequence import VehicleState
from sentry_ai.sequence.behaviour import heading_to_degrees
from sentry_ai.simulation.factory import MissionSource

logger = get_logger(__name__)

#: Bumped when the file layout changes.
TRAJECTORY_FORMAT = 1


@dataclass(frozen=True)
class Trajectory:
    """One mission's vehicle states, oldest first.

    Attributes:
        seed: The hazard seed that produced the mission.
        outcome: The mission's final phase, e.g. ``"completed"``.
        states: The state at mission start, then after every tick.
    """

    seed: int
    outcome: str
    states: tuple[VehicleState, ...]


@dataclass(frozen=True)
class TrajectorySet:
    """Trajectories recorded on one map, with the map's size.

    The size travels with the data because positions are only meaningful
    relative to it, and the model scales them by it.
    """

    grid_width: int
    grid_height: int
    trajectories: tuple[Trajectory, ...]

    @property
    def total_states(self) -> int:
        """States across every trajectory."""
        return sum(len(trajectory.states) for trajectory in self.trajectories)

    def save(self, path: Path) -> None:
        """Write the set as JSON, creating parent directories."""
        document = {
            "format": TRAJECTORY_FORMAT,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "trajectories": [
                {
                    "seed": trajectory.seed,
                    "outcome": trajectory.outcome,
                    "states": [_state_row(state) for state in trajectory.states],
                }
                for trajectory in self.trajectories
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TrajectorySet:
        """Read a set written by :meth:`save`.

        Raises:
            AssetNotFoundError: If the file does not exist.
            ValueError: If it is from an incompatible format.
        """
        if not path.is_file():
            raise AssetNotFoundError(
                f"Trajectory file not found: {path}. Run scripts/record_trajectories.py first."
            )
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("format") != TRAJECTORY_FORMAT:
            raise ValueError(
                f"Unsupported trajectory format {document.get('format')!r} in {path}; "
                f"expected {TRAJECTORY_FORMAT}. Re-record with scripts/record_trajectories.py."
            )
        return cls(
            grid_width=int(document["grid_width"]),
            grid_height=int(document["grid_height"]),
            trajectories=tuple(
                Trajectory(
                    seed=int(entry["seed"]),
                    outcome=str(entry["outcome"]),
                    states=tuple(_state_from_row(row) for row in entry["states"]),
                )
                for entry in document["trajectories"]
            ),
        )


class TrajectoryRecorder:
    """Runs missions and records the vehicle's state after every tick."""

    def __init__(self, mission_source: MissionSource, max_ticks: int = 5000) -> None:
        """Create a recorder.

        Args:
            mission_source: Builds a fresh, unstarted mission for a hazard
                seed — the same factory ``build_dataset.py`` uses.
            max_ticks: Hard stop per mission, so a stuck mission cannot
                record forever.
        """
        self._mission_source = mission_source
        self._max_ticks = max_ticks

    def record(self, seed: int) -> Trajectory:
        """Run the mission for ``seed`` to its end and return what the vehicle did."""
        mission = self._mission_source(seed)
        vehicle = mission.city_map.vehicle
        states = [vehicle_state(vehicle)]
        for _ in range(self._max_ticks):
            if mission.engine.tick() is None:
                break
            states.append(vehicle_state(vehicle))
        return Trajectory(seed=seed, outcome=mission.controller.phase.value, states=tuple(states))

    def record_set(self, seeds: Iterable[int], grid_width: int, grid_height: int) -> TrajectorySet:
        """Record one trajectory per seed."""
        trajectories = []
        for seed in seeds:
            trajectory = self.record(seed)
            logger.debug("seed %d: %d states, %s", seed, len(trajectory.states), trajectory.outcome)
            trajectories.append(trajectory)
        return TrajectorySet(grid_width, grid_height, tuple(trajectories))


def vehicle_state(vehicle: Vehicle) -> VehicleState:
    """The sequence model's view of the vehicle at this instant."""
    return VehicleState(
        position=vehicle.position,
        battery_percent=vehicle.battery_percent,
        heading_degrees=heading_to_degrees(vehicle.heading),
    )


def _state_row(state: VehicleState) -> list[float]:
    return [
        state.position.x,
        state.position.y,
        round(state.battery_percent, 4),
        state.heading_degrees,
    ]


def _state_from_row(row: list[float]) -> VehicleState:
    return VehicleState(
        position=Position(int(row[0]), int(row[1])),
        battery_percent=float(row[2]),
        heading_degrees=float(row[3]),
    )
