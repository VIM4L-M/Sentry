"""Unit tests for sentry_ai.perception.autoencoder.

What is tested is the contract, not the quality: shapes survive every frame
size the cameras produce, the adapter honours the ``IDenoiser`` port exactly
(dtype, shape, no mutation), and a checkpoint rebuilds the very network that
wrote it. How *well* it denoises is a training result, measured by
``scripts/evaluate_denoiser.py`` — PROJECT.md §16 keeps those two apart.

Untrained networks are used throughout, which is fine: every assertion here
holds for any weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the autoencoder needs torch (requirements-ml.txt)")

from sentry_ai.common.exceptions import AssetNotFoundError  # noqa: E402
from sentry_ai.interfaces.perception import IDenoiser  # noqa: E402
from sentry_ai.perception.autoencoder import (  # noqa: E402
    CHECKPOINT_FORMAT,
    AutoencoderArchitecture,
    ConvDenoisingAutoencoder,
    DenoisingAutoencoderNet,
    to_image,
    to_tensor,
)

_SMALL = AutoencoderArchitecture(base_channels=4, depth=2)


def _network(architecture: AutoencoderArchitecture = _SMALL) -> DenoisingAutoencoderNet:
    torch.manual_seed(0)
    return DenoisingAutoencoderNet(architecture)


def _frame(height: int = 160, width: int = 256, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


class TestArchitecture:
    def test_channels_double_at_every_level(self) -> None:
        assert AutoencoderArchitecture(base_channels=8, depth=3).widths == (8, 16, 32, 64)

    @pytest.mark.parametrize("overrides", [{"base_channels": 0}, {"depth": 0}])
    def test_degenerate_shapes_are_rejected(self, overrides: dict[str, int]) -> None:
        with pytest.raises(ValueError):
            AutoencoderArchitecture(**overrides)

    def test_skip_connections_cost_parameters(self) -> None:
        """The switch really changes the network, or the ablation measures nothing."""
        with_skips = _network(AutoencoderArchitecture(base_channels=4, depth=2))
        without = _network(
            AutoencoderArchitecture(base_channels=4, depth=2, skip_connections=False)
        )
        assert with_skips.parameter_count() > without.parameter_count()


class TestNetwork:
    @pytest.mark.parametrize(
        ("height", "width"),
        [(160, 256), (144, 144), (128, 128), (37, 51)],
        ids=["cctv", "onboard", "crop", "odd"],
    )
    @pytest.mark.parametrize("skips", [True, False], ids=["skips", "bottleneck"])
    def test_output_matches_input_shape(self, height: int, width: int, skips: bool) -> None:
        architecture = AutoencoderArchitecture(base_channels=4, depth=3, skip_connections=skips)
        network = _network(architecture)
        images = torch.rand(2, 3, height, width)
        assert network(images).shape == images.shape

    def test_output_is_always_a_valid_image(self) -> None:
        with torch.no_grad():
            output = _network()(torch.rand(1, 3, 32, 32) * 10 - 5)
        assert float(output.min()) >= 0.0
        assert float(output.max()) <= 1.0


class TestAdapter:
    def test_it_is_an_idenoiser(self) -> None:
        assert isinstance(ConvDenoisingAutoencoder(_network()), IDenoiser)

    @pytest.mark.parametrize(("height", "width"), [(160, 256), (144, 144), (21, 33)])
    def test_shape_and_dtype_are_preserved(self, height: int, width: int) -> None:
        cleaned = ConvDenoisingAutoencoder(_network()).denoise(_frame(height, width))
        assert cleaned.shape == (height, width, 3)
        assert cleaned.dtype == np.uint8

    def test_the_input_frame_is_not_mutated(self) -> None:
        frame = _frame()
        original = frame.copy()
        ConvDenoisingAutoencoder(_network()).denoise(frame)
        assert np.array_equal(frame, original)

    def test_a_non_contiguous_view_is_accepted(self) -> None:
        """Pipeline frames can be slices or flips; the port does not promise contiguity."""
        frame = _frame()[:, ::-1]
        assert ConvDenoisingAutoencoder(_network()).denoise(frame).shape == frame.shape

    @pytest.mark.parametrize(
        "bad",
        [
            np.zeros((16, 16), dtype=np.uint8),
            np.zeros((16, 16, 4), dtype=np.uint8),
            np.zeros((16, 16, 3), dtype=np.float32),
        ],
        ids=["greyscale", "rgba", "float"],
    )
    def test_malformed_frames_are_rejected(self, bad: np.ndarray) -> None:
        with pytest.raises(ValueError):
            ConvDenoisingAutoencoder(_network()).denoise(bad)

    def test_it_is_deterministic(self) -> None:
        denoiser = ConvDenoisingAutoencoder(_network())
        frame = _frame()
        assert np.array_equal(denoiser.denoise(frame), denoiser.denoise(frame))

    def test_the_network_is_left_in_evaluation_mode(self) -> None:
        network = _network()
        network.train()
        ConvDenoisingAutoencoder(network)
        assert not network.training


class TestCheckpoint:
    def test_a_checkpoint_rebuilds_the_same_network(self, tmp_path: Path) -> None:
        architecture = AutoencoderArchitecture(base_channels=4, depth=2, skip_connections=False)
        network = _network(architecture)
        path = tmp_path / "best.pt"
        torch.save(network.to_checkpoint({"epoch": 3}), path)

        restored = ConvDenoisingAutoencoder.from_checkpoint(path)
        frame = _frame(48, 64)
        assert np.array_equal(
            restored.denoise(frame), ConvDenoisingAutoencoder(network).denoise(frame)
        )

    def test_the_architecture_travels_with_the_weights(self) -> None:
        architecture = AutoencoderArchitecture(base_channels=4, depth=3, skip_connections=False)
        checkpoint = _network(architecture).to_checkpoint()
        assert DenoisingAutoencoderNet.from_checkpoint(checkpoint).architecture == architecture

    def test_an_unknown_format_fails_clearly(self) -> None:
        checkpoint = _network().to_checkpoint()
        checkpoint["format"] = CHECKPOINT_FORMAT + 1
        with pytest.raises(ValueError, match="checkpoint format"):
            DenoisingAutoencoderNet.from_checkpoint(checkpoint)

    def test_missing_weights_fail_before_anything_loads(self, tmp_path: Path) -> None:
        with pytest.raises(AssetNotFoundError):
            ConvDenoisingAutoencoder.from_checkpoint(tmp_path / "absent.pt")


class TestConversions:
    def test_a_frame_round_trips_exactly(self) -> None:
        frame = _frame(20, 30)
        assert np.array_equal(to_image(to_tensor(frame)), frame)

    def test_tensors_are_channels_first_and_scaled(self) -> None:
        tensor = to_tensor(np.full((4, 6, 3), 255, dtype=np.uint8))
        assert tensor.shape == (3, 4, 6)
        assert float(tensor.max()) == pytest.approx(1.0)

    def test_out_of_range_values_are_clamped_not_wrapped(self) -> None:
        image = to_image(torch.tensor([[[1.7]], [[-0.4]], [[0.5]]]))
        assert image.tolist() == [[[255, 0, 128]]]
