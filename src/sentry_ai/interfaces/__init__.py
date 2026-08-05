"""AI module ports (contracts) — implemented by adapters in later phases.

Every class here is an ``ABC`` with abstract methods only: no model logic,
no inference, no imports of PyTorch/YOLO/SB3. ``simulation/`` and other
application code depend on these interfaces, never on a concrete adapter
directly — see PROJECT.md §2 "Architectural Style" (dependency inversion)
and §12 "API Contracts Between Modules".
"""
