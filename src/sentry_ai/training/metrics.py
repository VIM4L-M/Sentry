"""A deliberately small metric log: one CSV row per epoch.

Kept dependency-free on purpose (PROJECT.md §14). A CSV opens in a
spreadsheet, diffs in a terminal, and plots in one line of pandas, which is
everything a single-developer project needs from experiment tracking.
TensorBoard or W&B can replace it later without any trainer changing,
because trainers only ever call :meth:`CsvMetricLogger.log`.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path


class CsvMetricLogger:
    """Appends metric rows to a CSV file, writing the header once.

    The columns are fixed by the first row. A later row with different keys
    is an error rather than a silently misaligned file: a column that
    appears halfway through a run is almost always a bug in the trainer.
    """

    def __init__(self, path: Path) -> None:
        """Start a fresh log at ``path``, replacing any previous one.

        Replaced rather than appended to: a re-run under the same name is a
        new experiment, and mixing its rows with the old one's would make
        both unreadable.
        """
        self._path = path
        self._columns: tuple[str, ...] | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)

    @property
    def path(self) -> Path:
        """Where the log is written."""
        return self._path

    def log(self, row: Mapping[str, float | int | str]) -> None:
        """Write one row, flushing immediately so a crashed run keeps its history.

        Raises:
            ValueError: If ``row`` is empty or its keys differ from the
                first row's.
        """
        if not row:
            raise ValueError("cannot log an empty metric row")
        columns = tuple(row)
        if self._columns is None:
            self._columns = columns
            self._append(columns)
        elif columns != self._columns:
            raise ValueError(
                f"metric columns changed mid-run: expected {self._columns}, got {columns}"
            )
        self._append(tuple(_format(row[column]) for column in columns))

    def _append(self, values: tuple[str, ...]) -> None:
        with self._path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(values)


def _format(value: float | int | str) -> str:
    """Floats to six significant figures — enough to compare, short enough to read."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
