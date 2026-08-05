"""Pure domain layer: entities, enums, and the ``CityMap`` aggregate.

Nothing here imports Pygame, PyTorch, YAML, or any I/O library — the
domain layer is plain Python + dataclasses, so it is trivially unit
testable and reusable from the simulation engine, the RL environment, and
the renderer alike. See PROJECT.md §2 "Architectural Style".
"""
