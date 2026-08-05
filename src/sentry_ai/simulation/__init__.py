"""Application layer: the tick-based world and the mission that runs in it.

This package owns *orchestration*, not intelligence. It advances simulated
time, applies actions to the world, and decides when a mission succeeds or
fails. Every intelligent decision it needs is pulled from a port in
:mod:`sentry_ai.interfaces` and injected at a composition root, so the
Phase 2 deterministic stand-ins and the Phase 6 trained models are
interchangeable without touching a line in here.
"""
