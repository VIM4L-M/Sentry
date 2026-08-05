"""The live mission window: renders the city while the engine runs it.

Drives :class:`~sentry_ai.simulation.engine.SimulationEngine` on a fixed
timestep accumulator, so simulated time advances at the configured tick
rate no matter what frame rate the window achieves. A slow machine drops
frames; it does not slow the mission down or change its outcome.

Controls: ``Escape`` quit, ``Space`` pause, ``Tab`` switch between
autonomous and manual driving, arrow keys / WASD to drive in manual mode.
"""

from __future__ import annotations

import pygame

from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.schema import RenderConfig
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.navigation import ILocalController, LocalDecision, LocalObservation
from sentry_ai.rendering.hud import HudRenderer
from sentry_ai.rendering.keyboard import KeyboardController
from sentry_ai.rendering.map_renderer import MapRenderer
from sentry_ai.rendering.theme import Theme
from sentry_ai.simulation.engine import SimulationEngine

logger = get_logger(__name__)

#: Never process more than this many simulation ticks in one frame. Without
#: it, a long stall (a breakpoint, a paused window) would be "caught up" in
#: a single burst that skips right past the mission.
_MAX_TICKS_PER_FRAME = 5


class ModeSwitchController(ILocalController):
    """Delegates to either an autonomous or a manual controller.

    Composition rather than a flag inside either controller: each stays
    unaware that the other exists, and a third mode (say, a replay driver)
    slots in without touching them.
    """

    def __init__(self, autonomous: ILocalController, manual: ILocalController) -> None:
        """Wrap the two controllers, starting in autonomous mode."""
        self._autonomous = autonomous
        self._manual = manual
        self._manual_mode = False

    @property
    def manual_mode(self) -> bool:
        """Whether a human is currently driving."""
        return self._manual_mode

    def toggle(self) -> None:
        """Switch between autonomous and manual driving."""
        self._manual_mode = not self._manual_mode
        logger.info("Control mode: %s", "manual" if self._manual_mode else "autonomous")

    def decide(self, observation: LocalObservation) -> LocalDecision:
        """Delegate to whichever controller is currently in charge."""
        active = self._manual if self._manual_mode else self._autonomous
        return active.decide(observation)


class SimulationApp:
    """Opens a window and runs a mission in it until it ends or is closed."""

    def __init__(
        self,
        city_map: CityMap,
        engine: SimulationEngine,
        render_config: RenderConfig,
        theme: Theme,
        mode_switch: ModeSwitchController | None = None,
    ) -> None:
        """Create the app.

        Args:
            city_map: The world being simulated, read for rendering.
            engine: The already-composed engine to drive.
            render_config: Window title, tile size, frame-rate cap.
            theme: Palette for the map and the HUD.
            mode_switch: The controller the engine was built with, when it
                supports manual override. ``None`` disables the Tab key.
        """
        self._city_map = city_map
        self._engine = engine
        self._render_config = render_config
        self._theme = theme
        self._mode_switch = mode_switch
        self._paused = False
        self._accumulated_seconds = 0.0

    def run(self) -> None:
        """Open the window and block until the user closes it."""
        pygame.init()
        try:
            hud = HudRenderer(self._theme)
            renderer = MapRenderer(theme=self._theme, tile_size_px=self._render_config.tile_size_px)
            surface = self._create_surface(hud.height_px)
            clock = pygame.time.Clock()

            running = True
            while running:
                running = self._handle_events()
                self._advance(clock.get_time() / 1000.0)
                renderer.draw(surface, self._city_map, route=self._engine.mission.route)
                hud.draw(surface, self._engine.mission, self._city_map.vehicle)
                pygame.display.flip()
                clock.tick(self._render_config.target_fps)
        finally:
            pygame.quit()
            logger.info("Mission window closed")

    def _create_surface(self, hud_height_px: int) -> pygame.Surface:
        """Size the window to the map plus the HUD strip."""
        tile = self._render_config.tile_size_px
        surface = pygame.display.set_mode(
            (self._city_map.width * tile, self._city_map.height * tile + hud_height_px)
        )
        pygame.display.set_caption(self._render_config.window_title)
        return surface

    def _handle_events(self) -> bool:
        """Drain the event queue. Returns whether the app should keep running."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_SPACE:
                self._paused = not self._paused
            elif event.key == pygame.K_TAB and self._mode_switch is not None:
                self._mode_switch.toggle()
        return True

    def _advance(self, frame_seconds: float) -> None:
        """Run however many whole simulation ticks this frame has earned."""
        if self._paused or self._engine.is_done:
            return

        seconds_per_tick = self._engine.seconds_per_tick
        self._accumulated_seconds += frame_seconds
        for _ in range(_MAX_TICKS_PER_FRAME):
            if self._accumulated_seconds < seconds_per_tick:
                return
            self._engine.tick()
            self._accumulated_seconds -= seconds_per_tick
            if self._engine.is_done:
                self._accumulated_seconds = 0.0
                return


def build_mode_switch(autonomous: ILocalController) -> ModeSwitchController:
    """Pair ``autonomous`` with a keyboard controller for manual override."""
    return ModeSwitchController(autonomous=autonomous, manual=KeyboardController())
