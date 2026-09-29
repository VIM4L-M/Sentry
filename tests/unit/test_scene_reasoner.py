"""Unit tests for camera-scene driving decisions (sentry_ai.decision.scene_reasoner)."""

from __future__ import annotations

from sentry_ai.decision.scene_reasoner import (
    Action,
    DecisionHold,
    SeenObject,
    decide,
    signal_colour,
)

FAR_LEFT = (0.05, 0.4, 0.2, 0.55)
FAR_AHEAD = (0.45, 0.4, 0.55, 0.55)
NEAR_AHEAD = (0.3, 0.3, 0.7, 0.95)


def test_an_empty_road_means_proceed() -> None:
    assert decide([]).action is Action.PROCEED


def test_distant_traffic_means_follow_not_stop() -> None:
    cars = [SeenObject("car", 0.8, FAR_LEFT) for _ in range(4)]
    decision = decide(cars)
    assert decision.action is Action.SLOW
    assert "follow" in decision.headline


def test_a_close_queue_means_wait() -> None:
    queue = [SeenObject("car", 0.8, (0.4, 0.5, 0.6, 0.8)) for _ in range(4)]
    decision = decide(queue)
    assert decision.action is Action.WAIT
    assert "wait" in decision.headline


def test_green_signal_moves_with_the_traffic() -> None:
    queue = [SeenObject("car", 0.8, (0.4, 0.5, 0.6, 0.8)) for _ in range(6)]
    green = SeenObject("traffic light", 0.7, FAR_AHEAD, "green")
    decision = decide([*queue, green])
    assert decision.action is Action.CAUTION
    assert "move with the traffic" in decision.headline


def test_red_signal_with_traffic_means_stop() -> None:
    cars = [SeenObject("car", 0.8, FAR_LEFT) for _ in range(4)]
    red = SeenObject("traffic light", 0.7, FAR_AHEAD, "red")
    assert decide([*cars, red]).action is Action.STOP


def test_the_confident_signal_outvotes_a_faint_far_one() -> None:
    near_green = SeenObject("traffic light", 0.72, FAR_AHEAD, "green")
    faint_red = SeenObject("traffic light", 0.1, FAR_LEFT, "red")
    assert signal_colour([near_green, faint_red]) == "green"
    assert signal_colour([SeenObject("traffic light", 0.5, FAR_AHEAD)]) == "unknown"
    assert signal_colour([]) is None


def test_a_person_in_front_means_stop() -> None:
    cars = [SeenObject("car", 0.8, FAR_LEFT) for _ in range(4)]
    decision = decide([*cars, SeenObject("person", 0.9, FAR_AHEAD)])
    assert decision.action is Action.STOP
    assert "Person" in decision.headline


def test_a_pedestrian_at_the_side_means_slow() -> None:
    assert decide([SeenObject("person", 0.9, FAR_LEFT)]).action is Action.SLOW


def test_a_close_vehicle_ahead_means_brake() -> None:
    decision = decide([SeenObject("truck", 0.7, NEAR_AHEAD)])
    assert decision.action is Action.STOP
    assert "Auto/Truck" in decision.headline


def test_signal_colour_sets_the_action() -> None:
    red = SeenObject("traffic light", 0.7, FAR_AHEAD, "red")
    green = SeenObject("traffic light", 0.7, FAR_AHEAD, "green")
    assert decide([red]).action is Action.STOP
    assert decide([green]).action is Action.CAUTION


def test_a_cow_means_stop() -> None:
    assert decide([SeenObject("cow", 0.6, FAR_LEFT)]).action is Action.STOP


def test_the_hold_keeps_a_decision_then_lets_it_go() -> None:
    hold = DecisionHold(hold_s=1.0)
    stop = decide([SeenObject("cow", 0.6, FAR_LEFT)])
    clear = decide([])
    assert hold.update(stop, 0.0) is stop
    assert hold.update(clear, 0.5) is stop
    assert hold.update(clear, 1.2) is clear
    assert hold.update(stop, 1.3) is stop
