"""The live mission window: renders the city while the engine runs it.

Drives :class:`~sentry_ai.simulation.engine.SimulationEngine` on a fixed
timestep accumulator, so simulated time advances at the configured tick
rate no matter what frame rate the window achieves. A slow machine drops
frames; it does not slow the mission down or change its outcome.

The window is laid out so a viewer can follow a whole mission without
narration: the world on the left, what the cameras see on the right, and
along the bottom the mission's state, its counters, and a running log of
what just happened.

Controls
    ``Escape`` quit, ``Space`` pause, ``R`` restart the mission, ``Tab``
    autonomous/manual driving, ``G`` toggle the occupancy-grid debug view,
    ``C`` toggle the camera panel, arrow keys / WASD to drive in manual mode.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pygame

from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.schema import RenderConfig
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.navigation import ILocalController, LocalDecision, LocalObservation
from sentry_ai.rendering.camera_panel import CameraPanelRenderer
from sentry_ai.rendering.grid_overlay import GridOverlayRenderer
from sentry_ai.rendering.hud import HudRenderer
from sentry_ai.rendering.keyboard import KeyboardController
from sentry_ai.rendering.map_renderer import MapRenderer
from sentry_ai.rendering.theme import Theme
from sentry_ai.sensors.camera import CameraView
from sentry_ai.sensors.frame import CameraFrame
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.simulation.engine import SimulationEngine

logger = get_logger(__name__)

#: Never process more than this many simulation ticks in one frame. Without
#: it, a long stall (a breakpoint, a paused window) would be "caught up" in
#: a single burst that skips right past the mission.
_MAX_TICKS_PER_FRAME = 5

#: How often the camera panel re-captures, in simulation ticks. Capturing
#: five frames every rendered frame is pure waste — the world cannot change
#: faster than the tick rate.
_CAMERA_REFRESH_TICKS = 3


@dataclass(frozen=True)
class MissionScene:
    """One composed mission: a world, an engine driving it, and its driver.

    Exists so the window can start a *fresh* mission without knowing how one
    is built. Composing a mission means reading configs and wiring adapters,
    which is a composition root's job — a renderer that knew how to do it
    would be reaching several layers down past its own.
    """

    city_map: CityMap
    engine: SimulationEngine
    mode_switch: ModeSwitchController | None = None


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
        sensor_rig: SensorRig | None = None,
        restart: Callable[[], MissionScene] | None = None,
    ) -> None:
        """Create the app.

        Args:
            city_map: The world being simulated, read for rendering.
            engine: The already-composed engine to drive.
            render_config: Window title, tile size, frame-rate cap.
            theme: Palette for the map and the HUD.
            mode_switch: The controller the engine was built with, when it
                supports manual override. ``None`` disables the Tab key.
            sensor_rig: Cameras to display alongside the world. ``None``
                omits the panel entirely and narrows the window to match —
                the rig is genuinely optional, not a required dependency of
                watching a mission.
            restart: Builds a fresh mission on demand, bound to the ``R``
                key. ``None`` disables it. Rebuilding from config rather
                than rewinding state is deliberate: a mission mutates the
                city, the victims, and the hazard generators, and restoring
                all of that correctly is far more error-prone than simply
                composing a new one.
        """
        self._city_map = city_map
        self._engine = engine
        self._render_config = render_config
        self._theme = theme
        self._mode_switch = mode_switch
        self._sensor_rig = sensor_rig
        self._restart = restart
        self._paused = False
        self._show_grid = False
        self._show_cameras = sensor_rig is not None
        self._accumulated_seconds = 0.0
        self._camera_frames: list[CameraFrame] = []
        self._ticks_since_capture = _CAMERA_REFRESH_TICKS

    @property
    def engine(self) -> SimulationEngine:
        """The engine currently being driven.

        Not necessarily the one passed in: restarting swaps it, and a caller
        reporting the outcome afterwards wants the mission that actually
        just ran.
        """
        return self._engine

    def run(self) -> None:
        """Open the window and block until the user closes it."""
        pygame.init()
        try:
            self._run_loop(*self._build_renderers())
        finally:
            pygame.quit()
            logger.info("Mission window closed")

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def _build_renderers(
        self,
    ) -> tuple[HudRenderer, MapRenderer, GridOverlayRenderer, CameraPanelRenderer]:
        """Create every renderer the window uses."""
        tile = self._render_config.tile_size_px
        return (
            HudRenderer(self._theme),
            MapRenderer(theme=self._theme, tile_size_px=tile),
            GridOverlayRenderer(theme=self._theme, tile_size_px=tile),
            CameraPanelRenderer(self._theme),
        )

    def _run_loop(
        self,
        hud: HudRenderer,
        map_renderer: MapRenderer,
        grid_renderer: GridOverlayRenderer,
        camera_panel: CameraPanelRenderer,
    ) -> None:
        """The frame loop, split out so ``run`` stays a thin try/finally."""
        surface = self._create_surface(hud.height_px, camera_panel.width_px)
        clock = pygame.time.Clock()

        running = True
        while running:
            frame_seconds = clock.get_time() / 1000.0
            running = self._handle_events()
            self._advance(frame_seconds)
            map_renderer.advance_animation(frame_seconds)

            self._draw_world(surface, map_renderer, grid_renderer)
            self._draw_cameras(surface, camera_panel)
            hud.draw(surface, self._engine.mission, self._city_map.vehicle)
            pygame.display.flip()
            clock.tick(self._render_config.target_fps)

    def _create_surface(self, hud_height_px: int, panel_width_px: int) -> pygame.Surface:
        """Size the window to the map, the camera strip, and the HUD."""
        tile = self._render_config.tile_size_px
        width = self._city_map.width * tile + (panel_width_px if self._sensor_rig else 0)
        surface = pygame.display.set_mode(
            (width, self._city_map.height * tile + hud_height_px)
        )
        pygame.display.set_caption(self._render_config.window_title)
        return surface

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_world(
        self,
        surface: pygame.Surface,
        map_renderer: MapRenderer,
        grid_renderer: GridOverlayRenderer,
    ) -> None:
        """Draw either the world or the command center's belief about it."""
        if self._show_grid:
            grid_renderer.draw(surface, self._engine.mission.grid)
            return

        mission = self._engine.mission
        map_renderer.draw(
            surface,
            self._city_map,
            route=mission.route,
            previous_route=mission.previous_route,
            camera_views=self._visible_camera_views(),
        )

    def _visible_camera_views(self) -> tuple[CameraView, ...]:
        """The CCTV footprints to outline on the map, if any are shown."""
        if self._sensor_rig is None or not self._show_cameras:
            return ()
        return self._sensor_rig.cctv_views

    def _draw_cameras(self, surface: pygame.Surface, panel: CameraPanelRenderer) -> None:
        """Refresh and draw the camera strip, if it is enabled."""
        if self._sensor_rig is None or not self._show_cameras:
            return
        if self._ticks_since_capture >= _CAMERA_REFRESH_TICKS:
            self._camera_frames = self._sensor_rig.capture_all(self._city_map)
            self._ticks_since_capture = 0

        tile = self._render_config.tile_size_px
        panel.draw(
            surface,
            self._camera_frames,
            left=self._city_map.width * tile,
            top=0,
            height=self._city_map.height * tile,
        )

    # ------------------------------------------------------------------
    # Input and time
    # ------------------------------------------------------------------

    def _handle_events(self) -> bool:
        """Drain the event queue. Returns whether the app should keep running."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                return False
            self._handle_keypress(event.key)
        return True

    def _handle_keypress(self, key: int) -> None:
        """Apply one non-quitting key press."""
        if key == pygame.K_SPACE:
            self._paused = not self._paused
        elif key == pygame.K_TAB and self._mode_switch is not None:
            self._mode_switch.toggle()
        elif key == pygame.K_g:
            self._show_grid = not self._show_grid
            logger.info("View: %s", "occupancy grid" if self._show_grid else "world")
        elif key == pygame.K_c and self._sensor_rig is not None:
            self._show_cameras = not self._show_cameras
        elif key == pygame.K_r and self._restart is not None:
            self._restart_mission()

    def _restart_mission(self) -> None:
        """Swap in a freshly composed mission, keeping the view settings.

        Which view you were looking at is a preference about the window, not
        state belonging to the mission, so ``G`` and ``C`` survive a restart
        while everything the mission owns does not.
        """
        assert self._restart is not None
        scene = self._restart()
        self._city_map = scene.city_map
        self._engine = scene.engine
        self._mode_switch = scene.mode_switch
        self._paused = False
        self._accumulated_seconds = 0.0
        self._camera_frames = []
        self._ticks_since_capture = _CAMERA_REFRESH_TICKS
        logger.info("Mission restarted")

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
            self._ticks_since_capture += 1
            self._accumulated_seconds -= seconds_per_tick
            if self._engine.is_done:
                self._accumulated_seconds = 0.0
                return


def build_mode_switch(autonomous: ILocalController) -> ModeSwitchController:
    """Pair ``autonomous`` with a keyboard controller for manual override."""
    return ModeSwitchController(autonomous=autonomous, manual=KeyboardController())
