"""Turns seeded rescue missions into a YOLO training dataset on disk.

The one decision here worth arguing about is **how the data is split**.

Consecutive frames from the same mission are nearly identical — the same
city, the same fires, the same camera, a second apart. Splitting those at
random between train and validation puts near-duplicates on both sides, and
the resulting mAP measures memorisation rather than detection. So the split
is by *mission*: validation missions are disasters the model has never seen
any frame of.

That is also why variety comes from re-seeding the hazards rather than from
capturing more frames per mission. A hundred frames of one fire teach less
than ten frames of ten different fires.
"""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from sentry_ai.common.logging_config import get_logger
from sentry_ai.domain.map import CityMap
from sentry_ai.sensors.degradation import FrameDegrader
from sentry_ai.sensors.frame import YOLO_CLASSES
from sentry_ai.sensors.rig import SensorRig
from sentry_ai.simulation.factory import Mission

logger = get_logger(__name__)

#: The splits written, in the order missions are assigned to them.
SPLITS: tuple[str, ...] = ("train", "val")

#: Builds an unstarted mission for a given hazard seed. Supplied by the
#: composition root so this module never reads a config file.
MissionSource = Callable[[int], Mission]


@dataclass(frozen=True)
class DatasetLayout:
    """Where each kind of file goes, in the layout Ultralytics expects.

    Attributes:
        root: Directory the dataset is written into.
    """

    root: Path

    def images(self, split: str) -> Path:
        """Degraded frames — what the detector is trained on."""
        return self.root / "images" / split

    def labels(self, split: str) -> Path:
        """YOLO label files, one per image, matched by stem."""
        return self.root / "labels" / split

    def clean(self, split: str) -> Path:
        """Uncorrupted frames, kept paired for the Unit IV autoencoder."""
        return self.root / "clean" / split

    @property
    def data_yaml(self) -> Path:
        """The dataset descriptor Ultralytics is pointed at."""
        return self.root / "data.yaml"

    def owned_directories(self) -> list[Path]:
        """Every directory this layout writes, and may therefore clear."""
        return [self.root / kind for kind in ("images", "labels", "clean")]


@dataclass
class CaptureOptions:
    """How much data to generate, and how densely.

    Attributes:
        train_missions: Disasters captured into the training split.
        val_missions: Disasters held out entirely for validation.
        capture_every: Capture one sample every N simulation ticks. Low
            values buy little — neighbouring ticks look almost identical.
        max_ticks: Tick budget per mission.
        seed_base: First hazard seed. Missions use consecutive seeds from
            here, so a build is reproducible and two builds with different
            bases share no disasters.
    """

    train_missions: int = 12
    val_missions: int = 4
    capture_every: int = 8
    max_ticks: int = 5000
    seed_base: int = 0


@dataclass
class DatasetStats:
    """What a build actually produced.

    Reported rather than assumed: synthetic data can look fine and be
    badly unbalanced, and the class counts here are the first place that
    shows up.
    """

    missions: int = 0
    frames: Counter[str] = field(default_factory=Counter)
    instances: Counter[str] = field(default_factory=Counter)
    empty_frames: int = 0

    @property
    def total_frames(self) -> int:
        """Frames written across every split."""
        return sum(self.frames.values())

    def record(self, split: str, label_lines: list[str]) -> None:
        """Account for one written frame and its labels."""
        self.frames[split] += 1
        if not label_lines:
            self.empty_frames += 1
            return
        for line in label_lines:
            self.instances[YOLO_CLASSES[int(line.split()[0])].value] += 1

    def as_display_rows(self) -> list[tuple[str, str]]:
        """Label/value pairs for direct printing, in presentation order."""
        rows = [
            ("Missions", str(self.missions)),
            ("Frames", str(self.total_frames)),
        ]
        rows += [(f"  {split}", str(self.frames[split])) for split in SPLITS]
        rows.append(("Background frames", f"{self.empty_frames} ({self._empty_share:.0%})"))
        rows += [
            (f"  {name}", str(self.instances[name]))
            for name in (kind.value for kind in YOLO_CLASSES)
        ]
        return rows

    @property
    def _empty_share(self) -> float:
        """Fraction of frames containing nothing to detect."""
        return self.empty_frames / self.total_frames if self.total_frames else 0.0


class YoloDatasetBuilder:
    """Runs missions and writes what the cameras saw as a YOLO dataset."""

    def __init__(
        self, rig: SensorRig, degrader: FrameDegrader, layout: DatasetLayout
    ) -> None:
        """Create a builder.

        Args:
            rig: The camera network to capture through.
            degrader: Applied to every frame to produce the detector's
                input. The clean original is written alongside it.
            layout: Where to write.
        """
        self._rig = rig
        self._degrader = degrader
        self._layout = layout

    def build(self, mission_source: MissionSource, options: CaptureOptions) -> DatasetStats:
        """Generate the whole dataset, replacing anything already there.

        Args:
            mission_source: Builds a fresh mission for a hazard seed.
            options: How many missions to run and how densely to sample.

        Returns:
            Counts of what was written, including the class balance.
        """
        self._clear()
        stats = DatasetStats()
        for split, seed in self._plan(options):
            logger.info("Capturing %s mission with hazard seed %d", split, seed)
            self._capture(mission_source(seed), split, seed, options, stats)
            stats.missions += 1
        self._write_data_yaml()
        return stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _plan(options: CaptureOptions) -> list[tuple[str, int]]:
        """Assign consecutive hazard seeds to splits, whole missions at a time."""
        counts = {"train": options.train_missions, "val": options.val_missions}
        plan: list[tuple[str, int]] = []
        seed = options.seed_base
        for split in SPLITS:
            for _ in range(counts[split]):
                plan.append((split, seed))
                seed += 1
        return plan

    def _capture(
        self,
        mission: Mission,
        split: str,
        seed: int,
        options: CaptureOptions,
        stats: DatasetStats,
    ) -> None:
        """Run one mission, sampling every ``capture_every`` ticks."""
        for tick in range(options.max_ticks):
            if tick % options.capture_every == 0:
                self._write_sample(mission.city_map, split, seed, tick, stats)
            if mission.engine.tick() is None:
                break

    def _write_sample(
        self, city_map: CityMap, split: str, seed: int, tick: int, stats: DatasetStats
    ) -> None:
        """Write every camera's degraded frame, clean frame, and labels."""
        for frame in self._rig.capture_all(city_map):
            stem = f"m{seed:03d}_{frame.camera_id}_{tick:05d}"
            lines = frame.to_yolo_lines()

            degraded = self._degrader.degrade(frame)
            _write_png(self._layout.images(split) / f"{stem}.png", degraded.pixels)
            _write_png(self._layout.clean(split) / f"{stem}.png", frame.pixels)
            _write_text(self._layout.labels(split) / f"{stem}.txt", "\n".join(lines))
            stats.record(split, lines)

    def _clear(self) -> None:
        """Remove previously generated files, but only the ones we own.

        A rebuild with fewer missions would otherwise leave stale frames
        behind and silently train on a mixture of two datasets. Only the
        ``images``/``labels``/``clean`` subtrees are removed — never the
        output directory itself, which may hold other things.
        """
        for directory in self._layout.owned_directories():
            if directory.exists():
                logger.info("Clearing %s", directory)
                shutil.rmtree(directory)

    def _write_data_yaml(self) -> None:
        """Write the descriptor Ultralytics reads to find the splits."""
        document = {
            "path": str(self._layout.root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {index: kind.value for index, kind in enumerate(YOLO_CLASSES)},
        }
        _write_text(self._layout.data_yaml, yaml.safe_dump(document, sort_keys=False).strip())


def _write_png(path: Path, pixels: NDArray[np.uint8]) -> None:
    """Save an ``(h, w, 3)`` RGB array as a PNG.

    OpenCV writes BGR, so the channels are reversed on the way out. Used
    rather than Pygame because a training package has no business importing
    a UI toolkit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), pixels[:, :, ::-1])


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text}\n" if text else "", encoding="utf-8")
