"""A window that always fits the screen it opens on.

The mission layouts are drawn at a fixed design size (the command center is
1600x900). A laptop with Windows display scaling at 125% or 150% would
otherwise stretch that past the edges of the screen. Two things prevent it:

* the process declares itself DPI-aware on Windows, so a 1600x900 window is
  1600x900 real pixels instead of being enlarged by the display scale;
* if the design size still does not fit the desktop, everything is drawn on
  an off-screen canvas of the design size and scaled down onto a smaller
  window each frame, keeping the aspect ratio.

Every renderer keeps drawing at design coordinates; only :func:`present`
knows about the real window.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pygame

from sentry_ai.common.logging_config import get_logger

logger = get_logger(__name__)

#: Share of the desktop the window may use, leaving room for the taskbar and title bar.
_MAX_SHARE_W, _MAX_SHARE_H = 0.98, 0.9


def make_dpi_aware() -> None:
    """Ask Windows not to enlarge the window by the display scale. No-op elsewhere."""
    if sys.platform != "win32":
        return
    try:
        import ctypes  # noqa: PLC0415

        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError) as exc:  # pragma: no cover - platform dependent
        logger.warning("Could not make the window DPI-aware: %s", exc)


@dataclass
class FittedWindow:
    """The real window, and the design-size canvas everything is drawn on."""

    display: pygame.Surface
    canvas: pygame.Surface

    @property
    def scaled(self) -> bool:
        """Whether the canvas is shrunk to fit the window."""
        return self.display.get_size() != self.canvas.get_size()

    def present(self) -> None:
        """Copy the canvas to the window (scaled if needed) and flip."""
        if self.scaled:
            pygame.transform.smoothscale(self.canvas, self.display.get_size(), self.display)
        pygame.display.flip()


def open_window(size: tuple[int, int], title: str) -> FittedWindow:
    """Open a window for a ``size`` design, shrunk to fit the desktop if needed."""
    desktop = _desktop_size()
    fit = size
    if desktop is not None:
        limit_w, limit_h = desktop[0] * _MAX_SHARE_W, desktop[1] * _MAX_SHARE_H
        scale = min(1.0, limit_w / size[0], limit_h / size[1])
        fit = (max(1, int(size[0] * scale)), max(1, int(size[1] * scale)))
    display = pygame.display.set_mode(fit)
    pygame.display.set_caption(title)
    canvas = display if fit == size else pygame.Surface(size)
    if fit != size:
        logger.info("Window %sx%s scaled to %sx%s to fit the screen", *size, *fit)
    return FittedWindow(display=display, canvas=canvas)


def _desktop_size() -> tuple[int, int] | None:
    sizes = pygame.display.get_desktop_sizes() if pygame.display.get_init() else []
    return tuple(sizes[0]) if sizes else None  # type: ignore[return-value]
