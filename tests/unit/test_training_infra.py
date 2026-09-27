"""Unit tests for the shared training infrastructure: seed, metrics, checkpoint.

These three modules outlive Phase 4 — the LSTM, DQN and fusion trainers
reuse them — so their contracts are pinned down on their own rather than
only through the autoencoder that first needed them.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.training.metrics import CsvMetricLogger
from sentry_ai.training.seed import resolve_device, seed_everything


class TestSeeding:
    def test_seeding_twice_replays_every_generator(self) -> None:
        seed_everything(11)
        first = (random.random(), float(np.random.rand()))
        seed_everything(11)
        assert (random.random(), float(np.random.rand())) == first

    def test_torch_is_seeded_too_when_installed(self) -> None:
        torch = pytest.importorskip("torch")
        seed_everything(11)
        first = torch.rand(3)
        seed_everything(11)
        assert torch.equal(torch.rand(3), first)

    def test_a_negative_seed_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            seed_everything(-1)

    @pytest.mark.parametrize("device", ["cpu", "cuda", "0"])
    def test_an_explicit_device_is_passed_through(self, device: str) -> None:
        assert resolve_device(device) == device

    def test_auto_resolves_to_something_real(self) -> None:
        pytest.importorskip("torch")
        assert resolve_device("auto") in {"cpu", "cuda"}


class TestCsvMetricLogger:
    def _rows(self, path: Path) -> list[list[str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.reader(handle))

    def test_the_header_is_written_once(self, tmp_path: Path) -> None:
        log = CsvMetricLogger(tmp_path / "run" / "metrics.csv")
        log.log({"epoch": 1, "loss": 0.5})
        log.log({"epoch": 2, "loss": 0.25})
        assert self._rows(log.path) == [["epoch", "loss"], ["1", "0.5"], ["2", "0.25"]]

    def test_floats_are_kept_short(self, tmp_path: Path) -> None:
        log = CsvMetricLogger(tmp_path / "metrics.csv")
        log.log({"loss": 1 / 3})
        assert self._rows(log.path)[1] == ["0.333333"]

    def test_a_new_log_replaces_an_old_one(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.csv"
        CsvMetricLogger(path).log({"loss": 1.0})
        CsvMetricLogger(path).log({"loss": 2.0})
        assert self._rows(path) == [["loss"], ["2"]]

    def test_changing_columns_mid_run_is_an_error(self, tmp_path: Path) -> None:
        log = CsvMetricLogger(tmp_path / "metrics.csv")
        log.log({"loss": 1.0})
        with pytest.raises(ValueError, match="columns changed"):
            log.log({"loss": 1.0, "psnr": 30.0})

    def test_an_empty_row_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            CsvMetricLogger(tmp_path / "metrics.csv").log({})


@dataclass(frozen=True)
class _Config:
    learning_rate: float = 0.001
    dataset_dir: Path = Path("data/synthetic")


class TestCheckpoint:
    @pytest.fixture(autouse=True)
    def _needs_torch(self) -> None:
        pytest.importorskip("torch")

    def test_weights_and_metadata_are_written_together(self, tmp_path: Path) -> None:
        from sentry_ai.training.checkpoint import load_metadata, save_checkpoint

        path = tmp_path / "run" / "best.pt"
        metadata_path = save_checkpoint(path, {"value": 1}, _Config(), {"psnr": 30.5})

        assert path.is_file()
        assert metadata_path == tmp_path / "run" / "best.json"
        metadata = load_metadata(path)
        assert metadata["weights"] == "best.pt"
        assert metadata["metrics"] == {"psnr": 30.5}
        assert metadata["config"]["dataset_dir"] == "data/synthetic"
        assert metadata["torch_version"]

    def test_the_payload_round_trips(self, tmp_path: Path) -> None:
        import torch

        from sentry_ai.training.checkpoint import save_checkpoint

        path = tmp_path / "best.pt"
        save_checkpoint(path, {"weights": torch.ones(2)}, _Config())
        assert torch.equal(torch.load(path, weights_only=True)["weights"], torch.ones(2))

    def test_missing_metadata_fails_clearly(self, tmp_path: Path) -> None:
        from sentry_ai.training.checkpoint import load_metadata

        with pytest.raises(AssetNotFoundError):
            load_metadata(tmp_path / "best.pt")

    def test_the_fingerprint_is_stable_and_sensitive(self) -> None:
        from sentry_ai.training.checkpoint import config_fingerprint

        config = _Config()
        assert config_fingerprint(config) == config_fingerprint(_Config())
        assert config_fingerprint(config) != config_fingerprint(replace(config, learning_rate=0.01))

    def test_paths_are_recorded_the_same_on_every_os(self) -> None:
        """A Windows path must fingerprint like its POSIX twin, not with backslashes."""
        from pathlib import PurePosixPath, PureWindowsPath

        from sentry_ai.training.checkpoint import config_fingerprint

        windows = _Config(dataset_dir=PureWindowsPath(r"data\synthetic"))  # type: ignore[arg-type]
        posix = _Config(dataset_dir=PurePosixPath("data/synthetic"))  # type: ignore[arg-type]
        assert config_fingerprint(windows) == config_fingerprint(posix)

    def test_the_fingerprint_needs_a_dataclass_instance(self) -> None:
        from sentry_ai.training.checkpoint import config_fingerprint

        with pytest.raises(TypeError):
            config_fingerprint({"learning_rate": 0.001})
        with pytest.raises(TypeError):
            config_fingerprint(_Config)
