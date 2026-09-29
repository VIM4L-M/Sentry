"""The mission-control panel: what every model is thinking, live (Phases 7-9).

A strip beside the camera panel that shows, every tick, each signal the
vehicle's decision is built from and the decision itself:

* the ablation switches — which models are live, and the key that flips each;
* the DQN's Q-value per action, its pick highlighted;
* the LSTM's predicted behaviour and confidence;
* what the onboard camera sees around the vehicle, as a class x region grid;
* the fusion network's final action, with an ``OVERRIDE`` flag when it
  overrules the DQN, and running counts of overrides and collisions.

It reads a :class:`~sentry_ai.decision.fused_controller.FusedLocalController`
and never drives anything: every number shown was computed for the decision
the vehicle actually made.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from sentry_ai.common.color import Color
from sentry_ai.decision.emergency_brake import EmergencyBrake
from sentry_ai.decision.fused_controller import FusedLocalController, FusionTrace
from sentry_ai.domain.enums import EntityKind
from sentry_ai.interfaces.decision import EVIDENCE_KINDS, EVIDENCE_REGIONS
from sentry_ai.interfaces.navigation import LOCAL_ACTION_ORDER
from sentry_ai.perception.scene_evidence import OnboardEvidenceSource
from sentry_ai.rendering.theme import Theme
from sentry_ai.simulation.grid_source import LaggedGridSource

#: Short labels for the action bars.
_ACTION_LABELS = {
    "move_forward": "forward",
    "reverse": "reverse",
    "turn_left": "left",
    "turn_right": "right",
    "stop": "stop",
}

#: Row labels for the camera grid, in ``EVIDENCE_KINDS`` order.
_KIND_LABELS = {EntityKind.VICTIM: "victim", EntityKind.FIRE: "fire", EntityKind.OBSTACLE: "debris"}

#: Column labels for the camera grid, in ``EVIDENCE_REGIONS`` order.
_REGION_LABELS = ("ahd", "far", "lft", "rgt", "bhd", "near")


@dataclass(frozen=True)
class AiPanelLayout:
    """Pixel geometry of the mission-control strip."""

    width_px: int = 300
    padding_px: int = 10
    line_px: int = 17
    bar_height_px: int = 11
    font_size_px: int = 14


@dataclass
class MissionControlState:
    """Everything the panel shows, gathered by the window each frame.

    Attributes:
        brain: The fused controller whose last decision is displayed.
        camera: The onboard evidence source, for its on/off and denoiser switches.
        lag: The lagging map source, when the scenario is enabled.
        seconds_per_tick: To show the lag in simulated seconds.
        collisions: The mission's collision count so far.
    """

    brain: FusedLocalController
    camera: OnboardEvidenceSource | None
    lag: LaggedGridSource | None
    seconds_per_tick: float
    collisions: int
    brake: EmergencyBrake | None = None


class AiPanelRenderer:
    """Draws :class:`MissionControlState` as a vertical strip."""

    def __init__(self, theme: Theme, layout: AiPanelLayout | None = None) -> None:
        """Create the renderer. ``pygame.font`` must already be initialised."""
        self._theme = theme
        self._layout = layout or AiPanelLayout()
        self._font = pygame.font.SysFont(
            "consolas,dejavusansmono,monospace", self._layout.font_size_px
        )
        self._bold = pygame.font.SysFont(
            "consolas,dejavusansmono,monospace", self._layout.font_size_px + 2, bold=True
        )

    @property
    def width_px(self) -> int:
        """How much horizontal space this panel needs."""
        return self._layout.width_px

    def draw(
        self, surface: pygame.Surface, state: MissionControlState, left: int, height: int
    ) -> None:
        """Draw the whole strip with its left edge at ``left``."""
        layout = self._layout
        pygame.draw.rect(
            surface, self._theme.hud.panel.as_tuple(), pygame.Rect(left, 0, layout.width_px, height)
        )
        pygame.draw.line(surface, self._theme.hud.accent.as_tuple(), (left, 0), (left, height), 2)
        x, y = left + layout.padding_px, layout.padding_px
        y = self._title(surface, "AI MISSION CONTROL", x, y)
        y = self._switches(surface, state, x, y)
        trace = state.brain.last
        if trace is None:
            self._text(surface, "waiting for the first tick...", x, y + layout.line_px)
            return
        y = self._policy(surface, trace, x, y)
        y = self._behaviour(surface, trace, x, y)
        y = self._camera(surface, trace, x, y)
        self._decision(surface, state, trace, x, y)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _switches(self, surface: pygame.Surface, state: MissionControlState, x: int, y: int) -> int:
        switches = state.brain.switches
        camera = state.camera
        rows: list[tuple[str, str, bool | None]] = [
            ("1", "camera + YOLO", switches.camera),
            ("2", "LSTM behaviour", switches.behaviour),
            ("3", "fusion MLP", switches.fusion if state.brain.has_fusion else None),
            ("4", "denoiser", camera.use_denoiser if camera and camera.has_denoiser else None),
            ("5", "DQN driver", switches.policy if state.brain.has_fallback else None),
        ]
        if state.brake is not None:
            rows.append(("6", "emergency brake", state.brake.enabled))
        if state.lag is not None:
            seconds = state.lag.max_lag * state.seconds_per_tick
            rows.append(("L", f"map lag {seconds:.1f}s", state.lag.lag > 0))
        for key, name, live in rows:
            status, color = self._status(live)
            self._text(surface, f"[{key}] {name}", x, y)
            self._text(
                surface,
                status,
                x + self._layout.width_px - 2 * self._layout.padding_px - 36,
                y,
                color,
            )
            y += self._layout.line_px
        return y + 4

    def _policy(self, surface: pygame.Surface, trace: FusionTrace, x: int, y: int) -> int:
        y = self._title(surface, "DQN  Q-values", x, y)
        q = [trace.local_decision.q_values.get(action, 0.0) for action in LOCAL_ACTION_ORDER]
        low, high = min(q), max(q)
        for action, value in zip(LOCAL_ACTION_ORDER, q, strict=True):
            share = (value - low) / (high - low) if high > low else 1.0
            chosen = action is trace.local_decision.action
            y = self._bar(
                surface, _ACTION_LABELS[action.value], share, f"{value:+.2f}", x, y, chosen
            )
        return y + 4

    def _behaviour(self, surface: pygame.Surface, trace: FusionTrace, x: int, y: int) -> int:
        y = self._title(surface, "LSTM  behaviour", x, y)
        signal = trace.behaviour
        name = signal.predicted_class.value if signal.confidence > 0 else "off"
        return (
            self._bar(surface, name, signal.confidence, f"{signal.confidence:.0%}", x, y, True) + 4
        )

    def _camera(self, surface: pygame.Surface, trace: FusionTrace, x: int, y: int) -> int:
        y = self._title(surface, "CAMERA  sees (ego)", x, y)
        cell = 30
        label_width = 56
        for column, label in enumerate(_REGION_LABELS):
            self._text(surface, label, x + label_width + column * cell, y)
        y += self._layout.line_px
        for kind in EVIDENCE_KINDS:
            self._text(surface, _KIND_LABELS[kind], x, y)
            for column, region in enumerate(EVIDENCE_REGIONS):
                value = trace.evidence.at(kind, region)
                rect = pygame.Rect(
                    x + label_width + column * cell, y, cell - 4, self._layout.line_px - 3
                )
                pygame.draw.rect(surface, self._heat(value).as_tuple(), rect)
            y += self._layout.line_px
        return y + 4

    def _decision(
        self,
        surface: pygame.Surface,
        state: MissionControlState,
        trace: FusionTrace,
        x: int,
        y: int,
    ) -> None:
        y = self._title(surface, "FUSION  decision", x, y)
        final = trace.final
        y = self._bar(
            surface,
            _ACTION_LABELS[final.action.value],
            final.rationale_score,
            f"{final.rationale_score:.0%}",
            x,
            y,
            True,
        )
        if trace.overridden:
            banner = self._bold.render(
                "OVERRIDE - camera wins", True, self._theme.hud.warning.as_tuple()
            )
            surface.blit(banner, (x, y + 2))
        y += self._layout.line_px + 6
        self._text(surface, f"overrides {state.brain.overrides}", x, y)
        self._text(surface, f"collisions {state.collisions}", x + 140, y, self._theme.hud.warning)
        if state.brake is not None:
            y += self._layout.line_px
            self._road_users(surface, state.brake, x, y)

    def _road_users(self, surface: pygame.Surface, brake: EmergencyBrake, x: int, y: int) -> None:
        """Brake interventions, and road users hit while the brake was off."""
        stops = sum(brake.brakes.values())
        hits = brake.hits
        self._text(surface, f"brakes {stops}", x, y)
        vehicles = sum(n for kind, n in hits.items() if kind.is_vehicle)
        others = sum(n for kind, n in hits.items() if not kind.is_vehicle)
        struck = f"hits {vehicles + others} ({others} on foot)"
        color = self._theme.hud.warning if sum(hits.values()) else None
        self._text(surface, struck, x + 100, y, color)

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def _title(self, surface: pygame.Surface, text: str, x: int, y: int) -> int:
        surface.blit(self._bold.render(text, True, self._theme.hud.accent.as_tuple()), (x, y))
        return y + self._layout.line_px + 3

    def _bar(
        self,
        surface: pygame.Surface,
        label: str,
        share: float,
        value: str,
        x: int,
        y: int,
        strong: bool,
    ) -> int:
        layout = self._layout
        self._text(surface, label, x, y)
        track = pygame.Rect(
            x + 70, y + 3, layout.width_px - 2 * layout.padding_px - 130, layout.bar_height_px
        )
        pygame.draw.rect(surface, self._track().as_tuple(), track)
        filled = track.copy()
        filled.width = max(1, int(track.width * max(0.0, min(1.0, share))))
        color = self._theme.hud.accent if strong else self._muted()
        pygame.draw.rect(surface, color.as_tuple(), filled)
        self._text(surface, value, track.right + 6, y)
        return y + layout.line_px

    def _status(self, live: bool | None) -> tuple[str, Color]:
        if live is None:
            return "n/a", self._muted()
        return ("ON", self._theme.hud.accent) if live else ("OFF", self._theme.hud.warning)

    def _heat(self, value: float) -> Color:
        """The empty-track colour for nothing seen, the warning colour for certain."""
        return self._track().blended_with(self._theme.hud.warning, max(0.0, min(1.0, value)))

    def _track(self) -> Color:
        """Unfilled bar background, derived from the HUD palette."""
        return self._theme.hud.panel.blended_with(self._theme.hud.text, 0.15)

    def _muted(self) -> Color:
        """De-emphasised text and bars, derived from the HUD palette."""
        return self._theme.hud.text.blended_with(self._theme.hud.panel, 0.5)

    def _text(
        self, surface: pygame.Surface, text: str, x: int, y: int, color: Color | None = None
    ) -> None:
        rendered = self._font.render(text, True, (color or self._theme.hud.text).as_tuple())
        surface.blit(rendered, (x, y))
