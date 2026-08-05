"""Shared type aliases used across layers.

Kept deliberately tiny in Phase 1 — grows as later phases introduce frame
buffers, observation tensors, and action types that multiple packages need
to reference without creating import cycles.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

#: A grid coordinate in map/tile space, distinct from pixel space.
GridCoordinate: TypeAlias = tuple[int, int]

#: Anything accepted where a filesystem path is expected.
PathLike: TypeAlias = str | Path
