"""Color palette for the renderer, loaded from ``configs/render.yaml``.

No color is ever hardcoded in drawing code — every terrain type and every
entity kind resolves through a :class:`Theme` loaded at startup, so the
whole city's look can be re-skinned by editing one YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentry_ai.common.exceptions import ConfigValidationError
from sentry_ai.common.types import PathLike
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.enums import EntityKind, TerrainType


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
        return (self.r, self.g, self.b)


@dataclass(frozen=True)
class Theme:
    """A complete color mapping for every terrain type and entity kind."""

    background: Color
    terrain_colors: dict[TerrainType, Color]
    entity_colors: dict[EntityKind, Color]

    @classmethod
    def from_config(cls, loader: ConfigLoader, relative_path: PathLike) -> Theme:
        """Load a palette YAML file and validate it covers every enum member.

        Raises:
            ConfigValidationError: If any :class:`TerrainType` or
                :class:`EntityKind` member has no color assigned.
        """
        data = loader.load_yaml(relative_path)

        background = _color_from_entry(data.get("background"), label="background")

        terrain_section = data.get("terrain", {})
        terrain_colors = {
            terrain_type: _color_from_entry(
                terrain_section.get(terrain_type.value), label=f"terrain.{terrain_type.value}"
            )
            for terrain_type in TerrainType
        }

        entity_section = data.get("entities", {})
        entity_colors = {
            entity_kind: _color_from_entry(
                entity_section.get(entity_kind.value), label=f"entities.{entity_kind.value}"
            )
            for entity_kind in EntityKind
        }

        return cls(
            background=background, terrain_colors=terrain_colors, entity_colors=entity_colors
        )


def _color_from_entry(entry: Any, *, label: str) -> Color:
    if entry is None:
        raise ConfigValidationError(f"render palette is missing a color for '{label}'")
    try:
        r, g, b = entry
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(
            f"render palette entry '{label}' must be a 3-element [r, g, b], got {entry!r}"
        ) from exc
    return Color(r=int(r), g=int(g), b=int(b))
