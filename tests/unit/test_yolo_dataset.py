"""Unit tests for sentry_ai.training.dataset.

The property these guard hardest is the split. A frame-level split would
put near-duplicates in both train and val and inflate mAP into
meaninglessness, and nothing about the resulting dataset would *look*
wrong — so it needs a test rather than care.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from sentry_ai.config.schema import DegradationConfig, SimulationConfig, VehicleConfig
from sentry_ai.domain.map import CityMap
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.frame import YOLO_CLASSES
from sentry_ai.sensors.palette import SensorPalette
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.simulation.factory import Mission, build_mission

pytest.importorskip("cv2", reason="the dataset writer needs opencv (requirements-ml.txt)")

from sentry_ai.training.dataset import (  # noqa: E402 - after the skip guard
    SPLITS,
    CaptureOptions,
    DatasetLayout,
    DatasetStats,
    YoloDatasetBuilder,
)


@pytest.fixture
def rig(project_root: Path) -> SensorRig:
    from sentry_ai.config.loader import ConfigLoader

    loader = ConfigLoader(project_root=project_root)
    return SensorRig.from_config(
        loader.load_sensor_config("configs/sensors.yaml"),
        SensorPalette.from_config(loader, "configs/sensors.yaml"),
    )


@pytest.fixture
def map_data(project_root: Path) -> dict[str, Any]:
    from sentry_ai.config.loader import ConfigLoader

    return ConfigLoader(project_root=project_root).load_yaml("configs/maps/city_default.yaml")


@pytest.fixture
def builder(rig: SensorRig, tmp_path: Path) -> YoloDatasetBuilder:
    return YoloDatasetBuilder(
        rig=rig,
        degrader=FrameDegrader(DegradationConfig(), np.random.default_rng(0)),
        layout=DatasetLayout(root=tmp_path / "ds"),
    )


@pytest.fixture
def mission_source(map_data: dict[str, Any]):  # type: ignore[no-untyped-def]
    def build(seed: int) -> Mission:
        return build_mission(
            city_map=CityMap.from_config(map_data),
            simulation_config=SimulationConfig(),
            vehicle_config=VehicleConfig(),
            hazard_seed=seed,
        )

    return build


_TINY = CaptureOptions(train_missions=2, val_missions=1, capture_every=60)


class TestLayout:
    def test_paths_follow_the_ultralytics_convention(self, tmp_path: Path) -> None:
        layout = DatasetLayout(root=tmp_path)
        assert layout.images("train") == tmp_path / "images" / "train"
        assert layout.labels("val") == tmp_path / "labels" / "val"
        assert layout.data_yaml == tmp_path / "data.yaml"

    def test_it_only_claims_ownership_of_its_own_subtrees(self, tmp_path: Path) -> None:
        """Clearing must never touch whatever else lives in the output dir."""
        owned = {path.name for path in DatasetLayout(root=tmp_path).owned_directories()}
        assert owned == {"images", "labels", "clean"}


class TestSplitting:
    def test_missions_are_assigned_whole_to_a_split(self) -> None:
        plan = YoloDatasetBuilder._plan(CaptureOptions(train_missions=3, val_missions=2))
        assert [split for split, _ in plan] == ["train"] * 3 + ["val"] * 2

    def test_no_hazard_seed_appears_in_two_splits(self) -> None:
        """This is what stops near-duplicate frames straddling the split."""
        plan = YoloDatasetBuilder._plan(CaptureOptions(train_missions=5, val_missions=3))
        seeds = [seed for _, seed in plan]
        assert len(set(seeds)) == len(seeds)

    def test_seed_base_shifts_every_mission(self) -> None:
        plan = YoloDatasetBuilder._plan(
            CaptureOptions(train_missions=2, val_missions=1, seed_base=100)
        )
        assert [seed for _, seed in plan] == [100, 101, 102]


class TestBuilding:
    def test_it_writes_matching_images_and_labels(
        self, builder: YoloDatasetBuilder, mission_source: Any, tmp_path: Path
    ) -> None:
        builder.build(mission_source, _TINY)
        layout = DatasetLayout(root=tmp_path / "ds")
        for split in SPLITS:
            images = {path.stem for path in layout.images(split).glob("*.png")}
            labels = {path.stem for path in layout.labels(split).glob("*.txt")}
            assert images and images == labels

    def test_every_image_has_a_clean_counterpart(
        self, builder: YoloDatasetBuilder, mission_source: Any, tmp_path: Path
    ) -> None:
        """Unit IV needs the pair; a missing half is a silently broken dataset."""
        builder.build(mission_source, _TINY)
        layout = DatasetLayout(root=tmp_path / "ds")
        for split in SPLITS:
            degraded = {path.name for path in layout.images(split).glob("*.png")}
            clean = {path.name for path in layout.clean(split).glob("*.png")}
            assert degraded == clean

    def test_the_degraded_and_clean_frames_actually_differ(
        self, builder: YoloDatasetBuilder, mission_source: Any, tmp_path: Path
    ) -> None:
        import cv2

        builder.build(mission_source, _TINY)
        layout = DatasetLayout(root=tmp_path / "ds")
        name = next(iter(layout.images("train").glob("*.png"))).name
        degraded = cv2.imread(str(layout.images("train") / name))
        clean = cv2.imread(str(layout.clean("train") / name))
        assert not np.array_equal(degraded, clean)

    def test_filenames_carry_the_mission_seed(
        self, builder: YoloDatasetBuilder, mission_source: Any, tmp_path: Path
    ) -> None:
        """So a suspicious frame can be traced back to the disaster that made it."""
        builder.build(mission_source, _TINY)
        names = [p.name for p in (tmp_path / "ds" / "images" / "train").glob("*.png")]
        assert all(name.startswith("m0") for name in names)

    def test_data_yaml_describes_the_splits_and_classes(
        self, builder: YoloDatasetBuilder, mission_source: Any, tmp_path: Path
    ) -> None:
        builder.build(mission_source, _TINY)
        document = yaml.safe_load((tmp_path / "ds" / "data.yaml").read_text(encoding="utf-8"))
        assert document["train"] == "images/train"
        assert document["val"] == "images/val"
        assert document["names"] == {i: k.value for i, k in enumerate(YOLO_CLASSES)}

    def test_labels_are_valid_yolo_rows(
        self, builder: YoloDatasetBuilder, mission_source: Any, tmp_path: Path
    ) -> None:
        builder.build(mission_source, _TINY)
        rows = 0
        for path in (tmp_path / "ds" / "labels" / "train").glob("*.txt"):
            for line in path.read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                class_id, *values = line.split()
                assert 0 <= int(class_id) < len(YOLO_CLASSES)
                assert all(0.0 <= float(value) <= 1.0 for value in values)
                rows += 1
        assert rows > 0

    def test_rebuilding_replaces_rather_than_accumulates(
        self, builder: YoloDatasetBuilder, mission_source: Any, tmp_path: Path
    ) -> None:
        """A stale frame from a bigger previous build would poison training."""
        builder.build(mission_source, CaptureOptions(train_missions=2, val_missions=1,
                                                     capture_every=30))
        many = len(list((tmp_path / "ds" / "images" / "train").glob("*.png")))
        builder.build(mission_source, CaptureOptions(train_missions=1, val_missions=1,
                                                     capture_every=90))
        few = len(list((tmp_path / "ds" / "images" / "train").glob("*.png")))
        assert few < many

    def test_it_is_reproducible(
        self, rig: SensorRig, mission_source: Any, tmp_path: Path
    ) -> None:
        counts = []
        for run in range(2):
            builder = YoloDatasetBuilder(
                rig=rig,
                degrader=FrameDegrader(DegradationConfig(), np.random.default_rng(0)),
                layout=DatasetLayout(root=tmp_path / f"run{run}"),
            )
            stats = builder.build(mission_source, _TINY)
            counts.append((stats.total_frames, dict(stats.instances)))
        assert counts[0] == counts[1]


class TestStats:
    def test_it_counts_frames_per_split(self) -> None:
        stats = DatasetStats()
        stats.record("train", ["0 0.5 0.5 0.1 0.1"])
        stats.record("val", [])
        assert stats.frames["train"] == 1
        assert stats.frames["val"] == 1
        assert stats.total_frames == 2

    def test_empty_labels_count_as_background(self) -> None:
        """Background frames are useful, but too many skews the training set."""
        stats = DatasetStats()
        stats.record("train", [])
        assert stats.empty_frames == 1
        assert stats.instances == {}

    def test_it_counts_instances_by_class_name(self) -> None:
        stats = DatasetStats()
        stats.record("train", ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1", "1 0.3 0.3 0.1 0.1"])
        assert stats.instances["victim"] == 1
        assert stats.instances["fire"] == 2

    def test_display_rows_survive_an_empty_build(self) -> None:
        """Dividing by zero frames would crash the report, not the build."""
        assert DatasetStats().as_display_rows()
