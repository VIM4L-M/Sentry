# Phase 1 — Foundation & Core Architecture: design notes

**Status:** Complete. See PROJECT.md §10 for scope, §11 for the milestone this phase
satisfies (M1 — Walking Skeleton).

This note records decisions made *during* implementation that weren't (or couldn't be)
fully specified in PROJECT.md up front, per CLAUDE.md's "never skip documentation" /
"always update README/architecture when it changes" rules.

## Deviations from the original PROJECT.md draft

### Map format: ASCII grid instead of flat coordinate lists

PROJECT.md §13's original example showed the map's terrain as flat per-category
coordinate lists (`roads: [[2,0],[2,1],...]`). Once actually authoring a "proper
city, not a grid world" (per the project's simulation requirements) this proved
unwieldy: a 30x20 city with six building blocks needs on the order of a hundred
coordinate pairs, which is error-prone to hand-write and unreadable in review.

`CityMap.from_config` (`domain/map.py`) instead reads a `grid` of `height` strings,
each `width` characters, resolved through a small legend
(`DEFAULT_TERRAIN_LEGEND`, optionally extended per-map via `terrain_legend`). This is
the standard tile-map authoring format used by most 2D game/sim tooling, is trivial to
visually verify against the rendered output, and is easy to hand-edit or generate
programmatically later (e.g. for Phase 6's map-randomization stretch goal). PROJECT.md
§13 has been updated to match; this is the authoritative implementation regardless.

`SAFE_ZONE` deliberately has **no** grid glyph — it carries extra metadata (radius,
capacity) that a single character can't express, so it stays an explicit top-level
`safe_zone:` key, same as `vehicle_start`.

### `pygame` → `pygame-ce`

The assigned dev machine runs Python 3.14. Vanilla `pygame` ships no prebuilt wheel for
cp314 and fails to build from source (missing SDL dev headers, no root access to
install them). `pygame-ce`, the actively maintained community fork, ships a cp314
wheel and is import-compatible (`import pygame` unchanged) — verified by running the
full Phase 1 renderer against it. No code depends on anything `pygame-ce` doesn't
provide. Documented in `requirements.txt` and PROJECT.md's Environment Notes.

### Terrain vs. Obstacle: two views of the same tiles, on purpose

`CityMap.terrain` is a `dict[Position, TerrainType]` covering the full backdrop (what
can be walked on). `CityMap.obstacles` is a separate `list[Obstacle]` generated
*from* the same grid, for exactly the terrain types that represent discrete, trackable
things (`RUBBLE`, `COLLAPSED_BUILDING`, `TREE`, `BLOCKED_ROAD`). This looks like
duplication but isn't: `terrain` answers "can the vehicle be here" (a Phase 2 movement
question), while `obstacles` gives each of those tiles an identity
(`obstacle_id`) — needed from Phase 3 onward when YOLO detections need something to
label and later phases need something to reference across frames. Plain backdrop
(`ROAD`, `BUILDING`, `OPEN_GROUND`, `SAFE_ZONE`) never gets an `Obstacle` entity: it's
not something the vehicle "detects" as a discrete object, and Unit II's syllabus target
is specifically victims/fire/**obstacles**, not "buildings in general."

## Explicitly deferred to later phases

Confirmed out of scope here, per PROJECT.md §10, and not attempted:

- Any vehicle movement, collision resolution, or input handling (Phase 2)
- Battery drain, mission timer, mission-objective tracking (Phase 2)
- Fire/smoke spread simulation — `EntityKind.SMOKE` is defined but has no domain
  entity or map field yet; smoke is a *dynamic byproduct* of fire spread, which is
  simulation logic, not static map data (Phase 2)
- HUD rendering (Phase 2)
- Any PyTorch/YOLO/LSTM/DQN/MLP code — `interfaces/` defines the five ports these
  will implement, with zero logic behind them (Phases 3-7)

## What Phase 1 proves

`pytest` is green (90 tests, 96% coverage on the modules this phase owns).
`python scripts/run_preview.py` loads `configs/app.yaml`, builds the validated
30x20 default disaster city (4 victims, 2 fires, 17 obstacles, 92 building tiles, 113
road tiles), and renders it in a Pygame(-ce) window — the full
config → domain → rendering path the rest of the project builds on.
