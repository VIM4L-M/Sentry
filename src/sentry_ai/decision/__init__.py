"""Learned decision-making: the DQN local controller (Phase 6) and fusion (Phase 7).

:mod:`~sentry_ai.decision.dqn_controller` fills
:class:`~sentry_ai.interfaces.navigation.ILocalController` with a policy trained
by reinforcement learning. It decides one tick at a time — forward, reverse,
turn, stop — along the route the A* planner chose. It never plans a route
itself: that split is ADR 0002.
"""
