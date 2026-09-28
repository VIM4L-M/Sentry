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

    With the full AI stack composed (Phase 8), a mission-control strip
    appears on the right and the ablation keys switch models live:
    ``1`` onboard camera + YOLO, ``2`` LSTM, ``3`` fusion MLP, ``4``
    denoiser, ``5`` DQN driver (off = waypoint follower), ``L`` map lag.
    ``V`` switches the world between the top-down and the 3D (isometric) view.
    ``[`` and ``]`` halve and double the simulation speed (0.25x to 2x); in
    the top-down view the vehicle glides from tile to tile at that speed.
    ``F`` switches to the drive view: the camera follows the vehicle and the
    map turns with it, Tesla style. City-sized maps open in it, with four
    surround cameras beside it. ``6`` switches the emergency brake for cars
    and pedestrians, on maps with traffic.
    ``S`` switches the top-down view to aerial imagery, on maps that have it.
    ``P`` shows or hides the street-view column (real photos), on maps that have it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pygame

from sentry_ai.common.logging_config import get_logger
from sentry_ai.config.schema import RenderConfig
from sentry_ai.decision.emergency_brake import EmergencyBrake
from sentry_ai.decision.fused_controller import FusedLocalController
from sentry_ai.domain.map import CityMap
from sentry_ai.interfaces.navigation import ILocalController, LocalDecision, LocalObservation
from sentry_ai.perception.scene_evidence import OnboardEvidenceSource
from sentry_ai.rendering.ai_panel import AiPanelRenderer, MissionControlState
from sentry_ai.rendering.camera_panel import CameraPanelRenderer
from sentry_ai.rendering.drive_view import DriveStatus, DriveViewRenderer, TrafficView
from sentry_ai.rendering.grid_overlay import GridOverlayRenderer
from sentry_ai.rendering.hud import HudRenderer
from sentry_ai.rendering.iso_renderer import IsoRenderer
from sentry_ai.rendering.keyboard import KeyboardController
from sentry_ai.rendering.map_renderer import MapRenderer
from sentry_ai.rendering.motion import VehicleGlide, VehiclePose
from sentry_ai.rendering.street_view import StreetPhotoLibrary, StreetViewPanel
from sentry_ai.rendering.surround_cameras import SurroundCameraPanel
from sentry_ai.rendering.theme import Theme
from sentry_ai.sensors.camera import CameraView
from sentry_ai.sensors.frame import CameraFrame
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.simulation.engine import SimulationEngine
from sentry_ai.simulation.grid_source import LaggedGridSource
from sentry_ai.simulation.traffic import TrafficProcess

logger = get_logger(__name__)

#: Never process more than this many simulation ticks in one frame. Without
#: it, a long stall (a breakpoint, a paused window) would be "caught up" in
#: a single burst that skips right past the mission.
_MAX_TICKS_PER_FRAME = 5

#: How often the camera panel re-captures, in simulation ticks. Capturing
#: five frames every rendered frame is pure waste — the world cannot change
#: faster than the tick rate.
_CAMERA_REFRESH_TICKS = 3

#: Bounds of the simulation speed set with ``[`` and ``]``. Speed changes only
#: how fast ticks happen in wall-clock time, never what a tick does.
MIN_SPEED, MAX_SPEED = 0.25, 2.0

#: Maps with more tiles than this are city-sized: the window keeps a fixed
#: world area, opens in the drive view, and shrinks the whole-map views to fit.
_LARGE_MAP_TILES = 2_000
_LARGE_WORLD_PX = (900, 640)


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
    brain: FusedLocalController | None = None
    camera: OnboardEvidenceSource | None = None
    lag: LaggedGridSource | None = None
    traffic: TrafficProcess | None = None
    brake: EmergencyBrake | None = None


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
        scene: MissionScene | None = None,
        satellite: pygame.Surface | None = None,
        street: StreetPhotoLibrary | None = None,
        speed: float = 1.0,
        metres_per_tile: float | None = None,
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
            scene: The composed mission, when it carries the Phase 8 AI
                stack (``brain``, ``camera``, ``lag``). Its presence adds
                the mission-control strip and the ablation keys.
            satellite: Aerial imagery of the map (``fetch_satellite.py``);
                enables the ``S`` key, and starts shown.
            street: Street photos for the map (``fetch_street_photos.py``);
                adds the street-view column and the ``P`` key.
            speed: Simulation speed to start at, between :data:`MIN_SPEED`
                and :data:`MAX_SPEED`; below 1 the mission plays in slow
                motion so a viewer can follow each move.
            metres_per_tile: Real size of a tile on an imported map, for the
                drive view's distances. ``None`` shows distances in tiles.
        """
        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise ValueError(f"speed must be in [{MIN_SPEED}, {MAX_SPEED}], got {speed}")
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
        self._scene = scene
        self._iso = IsoRenderer(theme)
        self._show_3d = False
        self._glide = VehicleGlide()
        self._frame_seconds = 0.0
        self._time_scale = speed
        self._drive = DriveViewRenderer()
        self._surround: SurroundCameraPanel | None = None
        self._glide_pose = VehiclePose(0.0, 0.0, 0.0)
        self._show_drive = self._is_large
        self._metres_per_tile = metres_per_tile
        self._satellite = satellite
        self._show_satellite = satellite is not None
        self._street = street
        self._street_panel: StreetViewPanel | None = None
        self._show_street = street is not None

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
    ) -> tuple[HudRenderer, MapRenderer, GridOverlayRenderer, CameraPanelRenderer, AiPanelRenderer]:
        """Create every renderer the window uses."""
        tile = self._map_tile_px
        return (
            HudRenderer(self._theme),
            MapRenderer(theme=self._theme, tile_size_px=tile),
            GridOverlayRenderer(theme=self._theme, tile_size_px=tile),
            CameraPanelRenderer(self._theme),
            AiPanelRenderer(self._theme),
        )

    def _run_loop(
        self,
        hud: HudRenderer,
        map_renderer: MapRenderer,
        grid_renderer: GridOverlayRenderer,
        camera_panel: CameraPanelRenderer,
        ai_panel: AiPanelRenderer,
    ) -> None:
        """The frame loop, split out so ``run`` stays a thin try/finally."""
        ai_width = ai_panel.width_px if self._has_brain else 0
        if self._street is not None:
            self._street_panel = StreetViewPanel(self._theme)
            ai_width += self._street_panel.width_px
        self._surround = SurroundCameraPanel(self._theme, self._drive)
        surface = self._create_surface(hud.height_px, camera_panel.width_px, ai_width)
        clock = pygame.time.Clock()

        running = True
        while running:
            frame_seconds = clock.get_time() / 1000.0
            running = self._handle_events()
            self._advance(frame_seconds)
            map_renderer.advance_animation(frame_seconds)
            self._iso.advance_animation(frame_seconds)
            self._drive.advance_animation(frame_seconds)
            self._frame_seconds = frame_seconds

            self._draw_world(surface, map_renderer, grid_renderer)
            self._draw_cameras(surface, camera_panel)
            self._draw_mission_control(surface, ai_panel, camera_panel.width_px)
            self._draw_street_view(surface, camera_panel.width_px, ai_panel.width_px)
            hud.draw(surface, self._engine.mission, self._city_map.vehicle)
            pygame.display.flip()
            clock.tick(self._render_config.target_fps)

    def _create_surface(
        self, hud_height_px: int, panel_width_px: int, ai_width_px: int = 0
    ) -> pygame.Surface:
        """Size the window to the map, the camera strip, the AI strip, and the HUD."""
        world_width, world_height = self._world_px
        width = world_width + (panel_width_px if self._sensor_rig else 0) + ai_width_px
        surface = pygame.display.set_mode((width, world_height + hud_height_px))
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
        area = pygame.Rect(0, 0, *self._world_px)
        step_seconds = self._engine.seconds_per_tick / self._time_scale
        pose = self._glide.update(self._city_map.vehicle, self._frame_seconds, step_seconds)
        self._glide_pose = pose
        if self._show_drive:
            self._draw_drive_view(surface, area, pose)
            return
        if self._show_3d:
            self._iso.draw(surface, self._city_map, area, route=mission.route)
            return
        pygame.draw.rect(surface, self._theme.background.as_tuple(), area)
        map_renderer.draw(
            surface,
            self._city_map,
            route=mission.route,
            previous_route=mission.previous_route,
            camera_views=self._visible_camera_views(),
            background=self._satellite if self._show_satellite else None,
            vehicle_pose=pose,
        )

    def _draw_drive_view(
        self, surface: pygame.Surface, area: pygame.Rect, pose: VehiclePose
    ) -> None:
        """The follow-camera view, with what the onboard model detected this tick."""
        mission = self._engine.mission
        camera = self._scene.camera if self._scene is not None else None
        goal = mission.route.goal
        victim = next(
            (
                v.victim_id
                for v in self._city_map.victims
                if goal is not None and v.position == goal
            ),
            None,
        )
        status = DriveStatus(
            phase=mission.phase.value.replace("_", " ").upper(),
            goal=f"to {victim}" if victim else "to the hospital" if goal else "holding",
            metres_per_tile=self._metres_per_tile,
        )
        self._drive.draw(
            surface,
            area,
            self._city_map,
            pose,
            route=mission.route,
            sightings=camera.last_sightings if camera is not None else (),
            background=self._satellite if self._show_satellite else None,
            status=status,
            traffic=self._traffic_view(),
        )

    def _traffic_view(self) -> TrafficView | None:
        """The road users to draw, placed part way through the tick in progress."""
        traffic = self._scene.traffic if self._scene is not None else None
        if traffic is None:
            return None
        brake = self._scene.brake if self._scene is not None else None
        fraction = min(1.0, self._accumulated_seconds / self._engine.seconds_per_tick)
        return TrafficView(
            agents=self._city_map.traffic,
            clock=traffic.clock + fraction,
            car_step_ticks=traffic.config.car_step_ticks,
            pedestrian_step_ticks=traffic.config.pedestrian_step_ticks,
            brake=brake.last if brake is not None else None,
        )

    @property
    def _is_large(self) -> bool:
        """A city-sized map: too big to show whole at the configured tile size."""
        return self._city_map.width * self._city_map.height > _LARGE_MAP_TILES

    @property
    def _map_tile_px(self) -> int:
        """Tile size for the whole-map views: as configured, or shrunk to fit a large map."""
        if not self._is_large:
            return self._render_config.tile_size_px
        width, height = _LARGE_WORLD_PX
        return max(1, min(width // self._city_map.width, height // self._city_map.height))

    @property
    def _world_px(self) -> tuple[int, int]:
        """Pixel size of the world area, left of the panels."""
        if self._is_large:
            return _LARGE_WORLD_PX
        tile = self._render_config.tile_size_px
        return self._city_map.width * tile, self._city_map.height * tile

    def _visible_camera_views(self) -> tuple[CameraView, ...]:
        """The CCTV footprints to outline on the map, if any are shown."""
        if self._sensor_rig is None or not self._show_cameras:
            return ()
        return self._sensor_rig.cctv_views

    def _draw_cameras(self, surface: pygame.Surface, panel: CameraPanelRenderer) -> None:
        """Refresh and draw the camera strip, if it is enabled.

        In the drive view on a city-sized map the fixed CCTV cameras cover
        one corner of the city, so the column shows the vehicle's four
        surround cameras instead.
        """
        if self._sensor_rig is None or not self._show_cameras:
            return
        world_width, world_height = self._world_px
        if self._show_drive and self._is_large and self._surround is not None:
            self._surround.draw(
                surface,
                world_width,
                world_height,
                self._city_map,
                self._glide_pose,
                self._traffic_view(),
                self._satellite if self._show_satellite else None,
            )
            return
        if self._ticks_since_capture >= _CAMERA_REFRESH_TICKS:
            self._camera_frames = self._sensor_rig.capture_all(self._city_map)
            self._ticks_since_capture = 0
        panel.draw(surface, self._camera_frames, left=world_width, top=0, height=world_height)

    @property
    def _has_brain(self) -> bool:
        return self._scene is not None and self._scene.brain is not None

    def _draw_mission_control(
        self, surface: pygame.Surface, panel: AiPanelRenderer, camera_width_px: int
    ) -> None:
        """Draw the AI strip right of the camera panel, when the AI stack is composed."""
        if self._scene is None or self._scene.brain is None:
            return
        world_width, world_height = self._world_px
        left = world_width + (camera_width_px if self._sensor_rig else 0)
        state = MissionControlState(
            brain=self._scene.brain,
            camera=self._scene.camera,
            lag=self._scene.lag,
            seconds_per_tick=self._engine.seconds_per_tick,
            collisions=self._engine.stats.collisions,
            brake=self._scene.brake,
        )
        panel.draw(surface, state, left, world_height)

    def _draw_street_view(
        self, surface: pygame.Surface, camera_width_px: int, ai_width_px: int
    ) -> None:
        """The real-photo column at the far right, when the map has street photos."""
        if self._street is None or self._street_panel is None:
            return
        world_width, height = self._world_px
        left = world_width + (camera_width_px if self._sensor_rig else 0)
        left += ai_width_px if self._has_brain else 0
        if not self._show_street:
            pygame.draw.rect(
                surface,
                self._theme.hud.panel.as_tuple(),
                pygame.Rect(left, 0, self._street_panel.width_px, height),
            )
            return
        onboard = self._camera_frames[-1] if self._camera_frames else None
        self._street_panel.draw(
            surface, self._street, self._city_map.vehicle, onboard, left, height
        )

    def _handle_ablation_key(self, key: int) -> bool:
        """Flip one model on or off. Returns whether ``key`` was an ablation key."""
        if self._scene is None or self._scene.brain is None:
            return False
        switches, camera, lag = self._scene.brain.switches, self._scene.camera, self._scene.lag
        if key == pygame.K_1:
            switches.camera = not switches.camera
        elif key == pygame.K_2:
            switches.behaviour = not switches.behaviour
        elif key == pygame.K_3:
            switches.fusion = not switches.fusion
        elif key == pygame.K_4 and camera is not None and camera.has_denoiser:
            camera.use_denoiser = not camera.use_denoiser
        elif key == pygame.K_5:
            switches.policy = not switches.policy
        elif key == pygame.K_6 and self._scene.brake is not None:
            self._scene.brake.enabled = not self._scene.brake.enabled
        elif key == pygame.K_l and lag is not None:
            lag.lag = 0 if lag.lag else lag.max_lag
        else:
            return False
        logger.info("Ablation switches: %s, lag %s", switches, lag.lag if lag else "n/a")
        return True

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
        if self._handle_ablation_key(key):
            return
        if key == pygame.K_SPACE:
            self._paused = not self._paused
        elif key == pygame.K_TAB and self._mode_switch is not None:
            self._mode_switch.toggle()
        elif key == pygame.K_g:
            self._show_grid = not self._show_grid
            logger.info("View: %s", "occupancy grid" if self._show_grid else "world")
        elif key == pygame.K_p and self._street is not None:
            self._show_street = not self._show_street
        elif key == pygame.K_s and self._satellite is not None:
            self._show_satellite = not self._show_satellite
            logger.info("View: %s", "satellite" if self._show_satellite else "drawn map")
        elif key == pygame.K_f:
            self._show_drive = not self._show_drive
            logger.info("View: %s", "drive" if self._show_drive else "whole map")
        elif key == pygame.K_v:
            self._show_3d = not self._show_3d
            logger.info("View: %s", "3D" if self._show_3d else "top-down")
        elif key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            step = 0.5 if key == pygame.K_LEFTBRACKET else 2.0
            self._time_scale = min(MAX_SPEED, max(MIN_SPEED, self._time_scale * step))
            logger.info("Simulation speed: %gx", self._time_scale)
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
        self._scene = self._carry_switches(scene)
        self._paused = False
        self._accumulated_seconds = 0.0
        self._camera_frames = []
        self._ticks_since_capture = _CAMERA_REFRESH_TICKS
        self._glide.reset()
        logger.info("Mission restarted")

    def _carry_switches(self, scene: MissionScene) -> MissionScene:
        """Keep the ablation settings across a restart, like the view settings."""
        old = self._scene
        if old is not None and old.brain is not None and scene.brain is not None:
            scene.brain.switches = old.brain.switches
        if old is not None and old.camera is not None and scene.camera is not None:
            scene.camera.use_denoiser = old.camera.use_denoiser
        if old is not None and old.lag is not None and scene.lag is not None:
            scene.lag.lag = min(old.lag.lag, scene.lag.max_lag)
        if old is not None and old.brake is not None and scene.brake is not None:
            scene.brake.enabled = old.brake.enabled
        return scene

    def _advance(self, frame_seconds: float) -> None:
        """Run however many whole simulation ticks this frame has earned."""
        if self._paused or self._engine.is_done:
            return

        seconds_per_tick = self._engine.seconds_per_tick
        self._accumulated_seconds += frame_seconds * self._time_scale
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
