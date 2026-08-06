"""Unit tests for the validation-result parsing in sentry_ai.training.yolo.

There is one genuinely subtle thing here worth pinning down. Ultralytics
reports ``p``/``r``/``ap50`` as arrays ordered by ``ap_class_index`` — only
the classes that actually appeared — while ``maps`` is indexed by class id
across *all* classes. Reading both the same way silently attributes one
class's numbers to another as soon as a class is absent from the split,
which on this dataset is entirely possible for victims.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from sentry_ai.training.yolo import ClassMetrics, EvaluationReport, _report_from


@dataclass
class _FakeBox:
    ap_class_index: list[int]
    p: list[float]
    r: list[float]
    ap50: list[float]
    maps: list[float]
    map50: float = 0.0
    map: float = 0.0


@dataclass
class _FakeResults:
    box: _FakeBox | None
    names: dict[int, str] = field(default_factory=dict)


_NAMES = {0: "victim", 1: "fire", 2: "obstacle"}


class TestParsingResults:
    def test_every_evaluated_class_is_reported(self) -> None:
        results = _FakeResults(
            box=_FakeBox(
                ap_class_index=[0, 1, 2],
                p=[0.1, 0.2, 0.3],
                r=[0.4, 0.5, 0.6],
                ap50=[0.7, 0.8, 0.9],
                maps=[0.11, 0.22, 0.33],
                map50=0.8,
                map=0.5,
            ),
            names=_NAMES,
        )
        report = _report_from(results)

        assert [entry.name for entry in report.per_class] == ["victim", "fire", "obstacle"]
        assert report.map50 == pytest.approx(0.8)
        assert report.map50_95 == pytest.approx(0.5)

    def test_metrics_land_on_the_right_class(self) -> None:
        results = _FakeResults(
            box=_FakeBox(
                ap_class_index=[0, 1, 2],
                p=[0.1, 0.2, 0.3],
                r=[0.4, 0.5, 0.6],
                ap50=[0.7, 0.8, 0.9],
                maps=[0.11, 0.22, 0.33],
            ),
            names=_NAMES,
        )
        victim = _report_from(results).metrics_for("victim")

        assert victim is not None
        assert victim.precision == pytest.approx(0.1)
        assert victim.recall == pytest.approx(0.4)
        assert victim.ap50 == pytest.approx(0.7)

    def test_a_class_missing_from_the_split_does_not_shift_the_others(self) -> None:
        """The indexing trap: p/r/ap50 are positional, maps is by class id."""
        results = _FakeResults(
            box=_FakeBox(
                ap_class_index=[0, 2],  # fire never appeared in validation
                p=[0.1, 0.3],
                r=[0.4, 0.6],
                ap50=[0.7, 0.9],
                maps=[0.11, 0.0, 0.33],  # still length-3, indexed by class id
            ),
            names=_NAMES,
        )
        report = _report_from(results)

        assert [entry.name for entry in report.per_class] == ["victim", "obstacle"]
        obstacle = report.metrics_for("obstacle")
        assert obstacle is not None
        assert obstacle.precision == pytest.approx(0.3)
        assert obstacle.ap50_95 == pytest.approx(0.33)
        assert report.metrics_for("fire") is None

    def test_an_unnamed_class_falls_back_to_its_index(self) -> None:
        results = _FakeResults(
            box=_FakeBox(ap_class_index=[7], p=[0.1], r=[0.2], ap50=[0.3], maps=[0.0] * 8),
            names={},
        )
        assert _report_from(results).per_class[0].name == "7"

    def test_a_result_without_a_box_yields_an_empty_report(self) -> None:
        report = _report_from(_FakeResults(box=None))
        assert report.per_class == ()
        assert report.map50 == 0.0

    def test_a_short_metric_array_does_not_raise(self) -> None:
        """Defensive: Ultralytics' shapes shift between releases."""
        results = _FakeResults(
            box=_FakeBox(ap_class_index=[0, 1], p=[0.1], r=[], ap50=[], maps=[]),
            names=_NAMES,
        )
        report = _report_from(results)
        assert report.per_class[1].precision == 0.0


class TestReport:
    def _report(self) -> EvaluationReport:
        return EvaluationReport(
            map50=0.5,
            map50_95=0.3,
            per_class=(
                ClassMetrics("victim", 0.8, 0.7, 0.75, 0.5),
                ClassMetrics("fire", 0.9, 0.95, 0.92, 0.7),
            ),
        )

    def test_lookup_by_name(self) -> None:
        found = self._report().metrics_for("fire")
        assert found is not None
        assert found.recall == pytest.approx(0.95)

    def test_lookup_of_an_absent_class_returns_none(self) -> None:
        assert self._report().metrics_for("obstacle") is None

    def test_display_rows_lead_with_the_overall_numbers(self) -> None:
        rows = self._report().as_display_rows()
        assert rows[0][0] == "mAP50"
        assert rows[1][0] == "mAP50-95"

    def test_display_rows_include_every_class(self) -> None:
        text = " ".join(label for label, _ in self._report().as_display_rows())
        assert "victim" in text
        assert "fire" in text

    def test_an_empty_report_still_renders(self) -> None:
        rows = EvaluationReport(map50=0.0, map50_95=0.0, per_class=()).as_display_rows()
        assert len(rows) == 2
