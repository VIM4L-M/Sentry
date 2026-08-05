"""Applies a :class:`LocalAction` to the rescue vehicle's physical state.

This is the only code allowed to move the vehicle, spend its battery, or
damage it. Keeping that in one place means the Phase 6 RL environment and
the interactive keyboard mode share identical physics — a policy cannot
learn against rules the live simulation does not enforce.

Nothing here decides *which* action to take; that is the local
controller's job (:class:`~sentry_ai.interfaces.navigation.ILocalController`).
"""

from __future__ import annotations

from dataclasses import dataclass

from sentry_ai.config.schema import VehicleConfig
from sentry_ai.domain.entities import Position, Vehicle
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.interfaces.navigation import LocalAction


@dataclass(frozen=True)
class MoveOutcome:
    """What actually happened when an action was applied.

    Returned rather than logged-and-forgotten because Phase 6's reward
    function is defined in terms of these facts (did we advance, did we
    hit something, what did it cost), and the HUD reports them live.

    Attributes:
        action: The action that was applied.
        moved: Whether the vehicle's tile changed.
        collided: Whether a move was refused because the target tile was
            impassable or off-map.
        battery_spent: Battery percentage consumed this tick.
        damage_taken: Health percentage lost this tick, from collision
            and/or standing in fire.
    """

    action: LocalAction
    moved: bool
    collided: bool
    battery_spent: float
    damage_taken: float


class VehicleController:
    """Mutates a :class:`Vehicle` according to the configured physics."""

    def __init__(self, config: VehicleConfig) -> None:
        """Create a controller.

        Args:
            config: Battery drain and damage model, injected so the RL
                environment can train against modified physics without
                editing this class.
        """
        self._config = config

    def apply(self, vehicle: Vehicle, action: LocalAction, grid: OccupancyGrid) -> MoveOutcome:
        """Apply ``action`` to ``vehicle``, mutating it in place.

        ``grid`` supplies collision and fire information — the vehicle
        reacts to the command center's *belief* about the world, exactly
        as it will once that belief comes from cameras instead of ground
        truth.
        """
        battery = self._config.battery
        if action is LocalAction.TURN_LEFT:
            vehicle.heading = vehicle.heading.turn_left()
            return self._settle(vehicle, action, False, False, battery.drain_per_turn, grid)
        if action is LocalAction.TURN_RIGHT:
            vehicle.heading = vehicle.heading.turn_right()
            return self._settle(vehicle, action, False, False, battery.drain_per_turn, grid)
        if action is LocalAction.STOP:
            return self._settle(vehicle, action, False, False, battery.drain_per_idle_tick, grid)
        return self._attempt_step(vehicle, action, grid)

    def _attempt_step(
        self, vehicle: Vehicle, action: LocalAction, grid: OccupancyGrid
    ) -> MoveOutcome:
        """Move one tile along (or against) the current heading, if legal."""
        dx, dy = vehicle.heading.delta
        if action is LocalAction.REVERSE:
            dx, dy = -dx, -dy

        target_x, target_y = vehicle.position.x + dx, vehicle.position.y + dy
        battery = self._config.battery
        if target_x < 0 or target_y < 0:
            return self._settle(vehicle, action, False, True, battery.drain_per_idle_tick, grid)

        target = Position(target_x, target_y)
        if not grid.is_traversable(target):
            return self._settle(vehicle, action, False, True, battery.drain_per_idle_tick, grid)

        vehicle.position = target
        return self._settle(vehicle, action, True, False, battery.drain_per_move, grid)

    def _settle(
        self,
        vehicle: Vehicle,
        action: LocalAction,
        moved: bool,
        collided: bool,
        battery_spent: float,
        grid: OccupancyGrid,
    ) -> MoveOutcome:
        """Charge battery and damage for the tick, clamped to legal ranges."""
        damage = self._config.collision_damage_percent if collided else 0.0
        standing_in_fire = (
            grid.in_bounds(vehicle.position)
            and grid.code_at(vehicle.position) is OccupancyCode.FIRE
        )
        if standing_in_fire:
            damage += self._config.fire_damage_per_tick

        vehicle.battery_percent = max(0.0, vehicle.battery_percent - battery_spent)
        vehicle.health_percent = max(0.0, vehicle.health_percent - damage)
        return MoveOutcome(
            action=action,
            moved=moved,
            collided=collided,
            battery_spent=battery_spent,
            damage_taken=damage,
        )
