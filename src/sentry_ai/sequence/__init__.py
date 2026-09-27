"""The sequence model: near-term vehicle behaviour from a window of past states (Unit III).

Phase 5 fills :class:`~sentry_ai.interfaces.sequence.IMotionPredictor` with an
LSTM. Two things live here:

* :mod:`~sentry_ai.sequence.behaviour` — what the four behaviour classes
  *mean*, as a rule over two vehicle states. The labels the LSTM trains on
  and the baseline it has to beat both come from this one rule, so they
  cannot drift apart.
* :mod:`~sentry_ai.sequence.lstm_predictor` — the network and the adapter.

Like ``perception/``, nothing here reaches into the simulation: a predictor
is handed a list of :class:`~sentry_ai.interfaces.sequence.VehicleState` and
returns a :class:`~sentry_ai.interfaces.sequence.BehaviourSignal`.
"""
