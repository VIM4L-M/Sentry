"""Global route planning — the command center's classical (non-learned) tier.

Holds the concrete adapters for :class:`~sentry_ai.interfaces.navigation.
IRoutePlanner`. Deliberately model-free: a city-scale route must be exact,
explainable, and replannable in milliseconds, which is what a shortest-path
search gives and a learned policy does not. Reinforcement learning owns the
*local* tier instead — see ``docs/adr/0002-two-tier-navigation-and-command-
center.md``.
"""
