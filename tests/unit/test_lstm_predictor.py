"""Unit tests for sentry_ai.sequence.lstm_predictor.

The contract, not the quality: features mean what the module says they
mean, short histories are padded rather than refused, the adapter honours
``IMotionPredictor``, and a checkpoint rebuilds the network — map scale and
all — that wrote it. How well it predicts is a training result, scored by
``scripts/train_lstm.py`` against the baselines.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the LSTM needs torch (requirements-ml.txt)")

from sentry_ai.common.exceptions import AssetNotFoundError  # noqa: E402
from sentry_ai.domain.entities import Position  # noqa: E402
from sentry_ai.interfaces.sequence import (  # noqa: E402
    BehaviourClass,
    BehaviourSignal,
    IMotionPredictor,
    VehicleState,
)
from sentry_ai.sequence.lstm_predictor import (  # noqa: E402
    CHECKPOINT_FORMAT,
    FEATURES_PER_STATE,
    BehaviourLstmNet,
    LstmArchitecture,
    LstmMotionPredictor,
    encode_states,
    pad_history,
)

_ARCH = LstmArchitecture(window=6, grid_width=30, grid_height=20, hidden_size=8, num_layers=1)


def _state(x: int, y: int, heading: float = 90.0, battery: float = 50.0) -> VehicleState:
    return VehicleState(Position(x, y), battery_percent=battery, heading_degrees=heading)


def _network(architecture: LstmArchitecture = _ARCH) -> BehaviourLstmNet:
    torch.manual_seed(0)
    return BehaviourLstmNet(architecture)


class TestFeatures:
    def test_one_row_of_seven_per_state(self) -> None:
        assert encode_states([_state(0, 0)] * 3, 30, 20).shape == (3, FEATURES_PER_STATE)

    def test_position_is_scaled_to_the_map(self) -> None:
        row = encode_states([_state(29, 19)], 30, 20)[0]
        assert row[0] == pytest.approx(1.0) and row[1] == pytest.approx(1.0)

    def test_heading_is_a_point_on_a_circle(self) -> None:
        """350 and 10 degrees must be close; as raw degrees they are 340 apart."""
        a = encode_states([_state(0, 0, heading=350.0)], 30, 20)[0][2:4]
        b = encode_states([_state(0, 0, heading=10.0)], 30, 20)[0][2:4]
        assert abs(a[0] - b[0]) < 0.4 and abs(a[1] - b[1]) < 0.05

    def test_battery_is_a_fraction(self) -> None:
        assert encode_states([_state(0, 0, battery=25.0)], 30, 20)[0][4] == pytest.approx(0.25)

    def test_the_step_is_the_move_just_made(self) -> None:
        rows = encode_states([_state(4, 4), _state(5, 4), _state(5, 3)], 30, 20)
        assert rows[0][5:].tolist() == [0.0, 0.0]
        assert rows[1][5:].tolist() == [1.0, 0.0]
        assert rows[2][5:].tolist() == [0.0, -1.0]

    def test_a_jump_is_clipped(self) -> None:
        rows = encode_states([_state(0, 0), _state(5, 0)], 30, 20)
        assert rows[1][5] == 1.0


class TestPadding:
    def test_a_long_history_keeps_only_the_latest(self) -> None:
        history = [_state(x, 0) for x in range(10)]
        assert [s.position.x for s in pad_history(history, 4)] == [6, 7, 8, 9]

    def test_a_short_history_repeats_its_oldest_state(self) -> None:
        padded = pad_history([_state(1, 0), _state(2, 0)], 4)
        assert [s.position.x for s in padded] == [1, 1, 1, 2]

    def test_padding_reads_as_standing_still(self) -> None:
        rows = encode_states(pad_history([_state(1, 0), _state(2, 0)], 4), 30, 20)
        assert rows[:3, 5:].sum() == 0.0

    def test_no_history_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            pad_history([], 4)


class TestNetwork:
    @pytest.mark.parametrize("layers", [1, 2])
    def test_one_logit_per_behaviour(self, layers: int) -> None:
        architecture = LstmArchitecture(
            window=6, grid_width=30, grid_height=20, hidden_size=8, num_layers=layers
        )
        logits = _network(architecture)(torch.zeros(5, 6, FEATURES_PER_STATE))
        assert logits.shape == (5, 4)

    @pytest.mark.parametrize("field", ["window", "grid_width", "hidden_size", "num_layers"])
    def test_degenerate_architectures_are_rejected(self, field: str) -> None:
        values = {"window": 6, "grid_width": 30, "grid_height": 20, field: 0}
        with pytest.raises(ValueError):
            LstmArchitecture(**values)  # type: ignore[arg-type]


class TestPredictor:
    def test_it_is_an_imotionpredictor(self) -> None:
        assert isinstance(LstmMotionPredictor(_network()), IMotionPredictor)

    def test_it_returns_a_valid_signal(self) -> None:
        signal = LstmMotionPredictor(_network()).predict([_state(x, 2) for x in range(8)])
        assert isinstance(signal, BehaviourSignal)
        assert 0.0 <= signal.confidence <= 1.0

    def test_probabilities_cover_every_class_and_sum_to_one(self) -> None:
        probabilities = LstmMotionPredictor(_network()).probabilities([_state(0, 0)])
        assert set(probabilities) == set(BehaviourClass)
        assert sum(probabilities.values()) == pytest.approx(1.0)

    def test_the_signal_is_the_most_likely_class(self) -> None:
        predictor = LstmMotionPredictor(_network())
        history = [_state(x, 2) for x in range(3)]
        probabilities = predictor.probabilities(history)
        signal = predictor.predict(history)
        assert signal.confidence == pytest.approx(max(probabilities.values()))
        assert probabilities[signal.predicted_class] == pytest.approx(signal.confidence)

    def test_a_single_state_is_enough(self) -> None:
        """The first tick of a mission must not crash the pipeline."""
        LstmMotionPredictor(_network()).predict([_state(0, 0)])

    def test_the_network_is_left_in_evaluation_mode(self) -> None:
        network = _network()
        network.train()
        LstmMotionPredictor(network)
        assert not network.training


class TestCheckpoint:
    def test_a_checkpoint_predicts_identically(self, tmp_path: Path) -> None:
        network = _network()
        path = tmp_path / "best.pt"
        torch.save(network.to_checkpoint({"epoch": 2}), path)

        history = [_state(x, 5) for x in range(6)]
        restored = LstmMotionPredictor.from_checkpoint(path).probabilities(history)
        assert restored == pytest.approx(LstmMotionPredictor(network).probabilities(history))

    def test_the_map_scale_travels_with_the_weights(self) -> None:
        architecture = LstmArchitecture(window=5, grid_width=41, grid_height=17, hidden_size=4)
        restored = BehaviourLstmNet.from_checkpoint(_network(architecture).to_checkpoint())
        assert restored.architecture == architecture

    def test_an_unknown_format_fails_clearly(self) -> None:
        checkpoint = _network().to_checkpoint()
        checkpoint["format"] = CHECKPOINT_FORMAT + 1
        with pytest.raises(ValueError, match="checkpoint format"):
            BehaviourLstmNet.from_checkpoint(checkpoint)

    def test_missing_weights_fail_before_anything_loads(self, tmp_path: Path) -> None:
        with pytest.raises(AssetNotFoundError):
            LstmMotionPredictor.from_checkpoint(tmp_path / "absent.pt")
