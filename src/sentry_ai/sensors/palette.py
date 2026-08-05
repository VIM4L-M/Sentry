"""The colors a synthetic camera paints with.

Kept separate from :class:`~sentry_ai.rendering.theme.Theme` on purpose.
The operator display is a schematic the *human* reads, and it is free to
change without consequence. This palette defines what the *detector* is
trained on: restyling it invalidates every model trained against the old
appearance, so the two must be able to move independently.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentry_ai.common.color import Color, color_from_entry
from sentry_ai.common.exceptions import ConfigValidationError
from sentry_ai.common.types import PathLike
from sentry_ai.config.loader import ConfigLoader
from sentry_ai.domain.enums import TerrainType

#: Ceiling on per-pixel texture jitter. Beyond this the noise starts
#: swamping the class colors it is meant to add texture to.
_MAX_JITTER = 60


@dataclass(frozen=True)
class SensorPalette:
    """Colors and texture strength for the camera rasterizer.

    Attributes:
        terrain: Base color per terrain type.
        void: Fill for tiles outside the map, which the onboard camera sees
            whenever it is clamped against a map edge.
        fire_core: Color at the centre of a fire.
        fire_edge: Color at the outer edge of a fire's footprint, blended
            toward ``fire_core`` by distance.
        victim: Color of a trapped victim's marker.
        debris: Color of rubble and collapsed structures.
        vehicle: Color of the rescue vehicle.
        texture_jitter: Peak per-pixel brightness variation. Zero produces
            flat color blocks, which a detector learns to separate almost
            trivially and which teaches it nothing useful.
    """

    terrain: dict[TerrainType, Color]
    void: Color
    fire_core: Color
    fire_edge: Color
    victim: Color
    debris: Color
    vehicle: Color
    texture_jitter: int

    def __post_init__(self) -> None:
        if not 0 <= self.texture_jitter <= _MAX_JITTER:
            raise ConfigValidationError(
                f"sensors.palette.texture_jitter must be within 0-{_MAX_JITTER}, "
                f"got {self.texture_jitter}"
            )
        missing = [terrain.value for terrain in TerrainType if terrain not in self.terrain]
        if missing:
            raise ConfigValidationError(
                f"sensors.palette.terrain is missing: {', '.join(sorted(missing))}"
            )

    @classmethod
    def from_config(cls, loader: ConfigLoader, relative_path: PathLike) -> SensorPalette:
        """Load the ``palette`` section of a sensors config file.

        Raises:
            AssetNotFoundError: If the file does not exist.
            ConfigValidationError: If a color is missing or malformed, or a
                terrain type has no entry.
        """
        data = loader.load_yaml(relative_path)
        palette = data.get("palette")
        if not isinstance(palette, dict):
            raise ConfigValidationError(
                f"sensors config '{relative_path}' must contain a 'palette' mapping"
            )

        terrain_section = palette.get("terrain", {})
        terrain = {
            terrain_type: color_from_entry(
                terrain_section.get(terrain_type.value),
                label=f"sensors.palette.terrain.{terrain_type.value}",
            )
            for terrain_type in TerrainType
        }
        return cls(
            terrain=terrain,
            void=color_from_entry(palette.get("void"), label="sensors.palette.void"),
            fire_core=color_from_entry(palette.get("fire_core"), label="sensors.palette.fire_core"),
            fire_edge=color_from_entry(palette.get("fire_edge"), label="sensors.palette.fire_edge"),
            victim=color_from_entry(palette.get("victim"), label="sensors.palette.victim"),
            debris=color_from_entry(palette.get("debris"), label="sensors.palette.debris"),
            vehicle=color_from_entry(palette.get("vehicle"), label="sensors.palette.vehicle"),
            texture_jitter=int(palette.get("texture_jitter", 10)),
        )
