"""Phase 8: every trained model wired into one autonomous mission.

This package is the reusable half of a composition root. It knows which
concrete adapters exist and how to connect them — the camera-built map
(Phases 3-4), the fused local controller (Phases 5-7) — so that the live
window, the headless evaluation, and the Phase 9 dashboard all run exactly
the same stack. It holds no model logic of its own; everything here is
wiring, timing, and record-keeping.
"""
