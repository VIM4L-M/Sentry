"""An RGB color value object, shared by everything that paints.

Lives in ``common`` rather than in ``rendering`` because two unrelated
adapters need it: the Pygame operator display and the synthetic camera
rasterizer in :mod:`sentry_ai.sensors`. Neither should have to import the
other to name a color.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentry_ai.common.exceptions import ConfigValidationError


@dataclass(frozen=True)
class Color:
    """An RGB color in the 0-255 range."""

    r: int
    g: int
    b: int

    def __post_init__(self) -> None:
        for name, channel in (("r", self.r), ("g", self.g), ("b", self.b)):
            if not 0 <= channel <= 255:
                raise ConfigValidationError(f"Color channel '{name}' must be 0-255, got {channel}")

    def as_tuple(self) -> tuple[int, int, int]:
        """This color as a plain ``(r, g, b)`` tuple."""
        return (self.r, self.g, self.b)

    def blended_with(self, other: Color, weight: float) -> Color:
        """Linear blend toward ``other``; ``weight`` 0 keeps self, 1 gives other."""
        ratio = min(1.0, max(0.0, weight))
        return Color(
            r=round(self.r + (other.r - self.r) * ratio),
            g=round(self.g + (other.g - self.g) * ratio),
            b=round(self.b + (other.b - self.b) * ratio),
        )


def color_from_entry(entry: Any, *, label: str) -> Color:
    """Parse a ``[r, g, b]`` YAML sequence into a :class:`Color`.

    Shared by every palette loader so a malformed color reports the same
    way whether it came from the operator display's theme or a camera's.

    Raises:
        ConfigValidationError: If the entry is missing or is not a
            3-element sequence.
    """
    if entry is None:
        raise ConfigValidationError(f"palette is missing a color for '{label}'")
    try:
        r, g, b = entry
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(
            f"palette entry '{label}' must be a 3-element [r, g, b], got {entry!r}"
        ) from exc
    return Color(r=int(r), g=int(g), b=int(b))
