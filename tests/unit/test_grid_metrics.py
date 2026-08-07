"""Unit tests for sentry_ai.perception.grid_metrics (Phase 3.3).

These pin down the distinction the whole module exists to make: that cell
accuracy is nearly meaningless on a grid this sparse, and that the three
ways a belief can be wrong have very different consequences for a mission.

Grids here are built by hand rather than from a city, so each test states
exactly one disagreement and asserts what it costs.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentry_ai.common.exceptions import DomainValidationError
from sentry_ai.domain.entities import Position
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid
from sentry_ai.perception.grid_metrics import CodeScore, GridComparison


def _grid(width: int = 4, height: int = 3) -> OccupancyGrid:
    return OccupancyGrid.empty(width, height)


def _with(*marks: tuple[int, int, OccupancyCode]) -> OccupancyGrid:
    grid = _grid()
    for x, y, code in marks:
        grid.mark(Position(x, y), code)
    return grid


class TestCodeScore:
    def test_a_perfect_score(self) -> None:
        score = CodeScore(OccupancyCode.VICTIM, true_positives=3, false_positives=0,
                          false_negatives=0)
        assert (score.precision, score.recall, score.f1, score.iou) == (1.0, 1.0, 1.0, 1.0)

    def test_claiming_nothing_when_there_is_nothing_scores_perfect(self) -> None:
        """Otherwise a correct grid is punished for a class the city lacks."""
        score = CodeScore(OccupancyCode.FIRE, 0, 0, 0)
        assert score.precision == 1.0
        assert score.recall == 1.0
        assert score.iou == 1.0

    def test_missing_everything_scores_zero_recall(self) -> None:
        score = CodeScore(OccupancyCode.VICTIM, 0, 0, 4)
        assert score.recall == 0.0
        assert score.f1 == 0.0

    def test_precision_and_recall_are_counted_independently(self) -> None:
        score = CodeScore(OccupancyCode.DEBRIS, true_positives=6, false_positives=2,
                          false_negatives=3)
        assert score.precision == pytest.approx(6 / 8)
        assert score.recall == pytest.approx(6 / 9)
        assert score.iou == pytest.approx(6 / 11)

    def test_f1_sits_between_precision_and_recall(self) -> None:
        score = CodeScore(OccupancyCode.FIRE, 5, 5, 0)
        assert score.precision < score.f1 < score.recall


class TestAgreement:
    def test_identical_grids_agree_completely(self) -> None:
        grid = _with((1, 1, OccupancyCode.FIRE), (2, 2, OccupancyCode.VICTIM))
        result = GridComparison.between(grid, grid)
        assert result.cell_agreement == 1.0
        assert result.traversability_agreement == 1.0
        assert result.is_drivable
        assert result.finds_every_victim

    def test_cell_agreement_is_generous_on_a_sparse_grid(self) -> None:
        """The number the module docstring warns about, demonstrated.

        An empty belief misses the single fire entirely and still scores
        11/12. This is why traversability and the named failure sets exist.
        """
        truth = _with((1, 1, OccupancyCode.FIRE))
        result = GridComparison.between(truth, _grid())
        assert result.cell_agreement == pytest.approx(11 / 12)
        assert result.scores[OccupancyCode.FIRE].recall == 0.0

    def test_a_wrong_code_can_still_agree_on_traversability(self) -> None:
        """Fire called debris is a mislabel, not a navigation error."""
        truth = _with((1, 1, OccupancyCode.FIRE))
        belief = _with((1, 1, OccupancyCode.DEBRIS))
        result = GridComparison.between(truth, belief)
        assert result.cell_agreement < 1.0
        assert result.traversability_agreement == 1.0
        assert result.is_drivable

    def test_mismatched_shapes_are_rejected(self) -> None:
        with pytest.raises(DomainValidationError, match="cannot compare"):
            GridComparison.between(_grid(4, 3), _grid(5, 3))


class TestMissedHazards:
    def test_a_hazard_the_belief_missed_is_named(self) -> None:
        """Believed clear, actually burning — the vehicle drives into it."""
        truth = _with((2, 1, OccupancyCode.FIRE))
        result = GridComparison.between(truth, _grid())
        assert result.missed_hazards == frozenset({Position(2, 1)})
        assert not result.is_drivable

    def test_a_detected_hazard_is_not_a_miss(self) -> None:
        truth = _with((2, 1, OccupancyCode.FIRE))
        result = GridComparison.between(truth, _with((2, 1, OccupancyCode.FIRE)))
        assert result.missed_hazards == frozenset()
        assert result.is_drivable

    @pytest.mark.parametrize(
        "code", [OccupancyCode.FIRE, OccupancyCode.DEBRIS, OccupancyCode.BUILDING]
    )
    def test_every_impassable_code_counts_as_a_hazard(self, code: OccupancyCode) -> None:
        result = GridComparison.between(_with((0, 0, code)), _grid())
        assert result.missed_hazards == frozenset({Position(0, 0)})


class TestPhantomObstacles:
    def test_an_invented_obstacle_is_named(self) -> None:
        """Believed blocked, actually clear — a detour, or an unreachable victim."""
        result = GridComparison.between(_grid(), _with((3, 0, OccupancyCode.DEBRIS)))
        assert result.phantom_obstacles == frozenset({Position(3, 0)})
        assert result.is_drivable, "a phantom is safe, just expensive"

    def test_phantoms_and_misses_are_counted_separately(self) -> None:
        """They have opposite consequences and must never be averaged together."""
        truth = _with((0, 0, OccupancyCode.FIRE))
        belief = _with((3, 2, OccupancyCode.DEBRIS))
        result = GridComparison.between(truth, belief)
        assert result.missed_hazards == frozenset({Position(0, 0)})
        assert result.phantom_obstacles == frozenset({Position(3, 2)})


class TestVictims:
    def test_a_missed_victim_is_named(self) -> None:
        truth = _with((1, 2, OccupancyCode.VICTIM))
        result = GridComparison.between(truth, _grid())
        assert result.missed_victims == frozenset({Position(1, 2)})
        assert not result.finds_every_victim

    def test_a_victim_buried_under_debris_counts_as_missed(self) -> None:
        """The Phase 3.1 labelling bug, expressed as a grid metric.

        Calling a trapped victim ``DEBRIS`` is worse than not seeing them: it
        is an impassable code, so the planner routes around the person.
        """
        truth = _with((1, 2, OccupancyCode.VICTIM))
        belief = _with((1, 2, OccupancyCode.DEBRIS))
        result = GridComparison.between(truth, belief)
        assert result.missed_victims == frozenset({Position(1, 2)})
        assert result.phantom_obstacles == frozenset({Position(1, 2)})

    def test_an_invented_victim_is_named(self) -> None:
        result = GridComparison.between(_grid(), _with((0, 1, OccupancyCode.VICTIM)))
        assert result.phantom_victims == frozenset({Position(0, 1)})
        assert result.finds_every_victim, "inventing one is not the same as losing one"

    def test_a_victim_found_on_the_wrong_tile_is_both(self) -> None:
        truth = _with((1, 1, OccupancyCode.VICTIM))
        belief = _with((2, 1, OccupancyCode.VICTIM))
        result = GridComparison.between(truth, belief)
        assert result.missed_victims == frozenset({Position(1, 1)})
        assert result.phantom_victims == frozenset({Position(2, 1)})


class TestPerCodeScores:
    def test_every_code_is_scored_even_when_absent(self) -> None:
        result = GridComparison.between(_grid(), _grid())
        assert set(result.scores) == set(OccupancyCode)

    def test_counts_add_up_to_the_grid(self) -> None:
        truth = _with((0, 0, OccupancyCode.FIRE), (1, 0, OccupancyCode.VICTIM))
        belief = _with((0, 0, OccupancyCode.FIRE), (2, 0, OccupancyCode.VICTIM))
        result = GridComparison.between(truth, belief)

        victim = result.scores[OccupancyCode.VICTIM]
        assert (victim.true_positives, victim.false_positives, victim.false_negatives) == (0, 1, 1)
        assert result.scores[OccupancyCode.FIRE].true_positives == 1

    def test_total_cells_matches_the_grid_size(self) -> None:
        assert GridComparison.between(_grid(4, 3), _grid(4, 3)).total_cells == 12


class TestSummary:
    def test_the_summary_names_every_headline_number(self) -> None:
        truth = _with((1, 1, OccupancyCode.FIRE), (2, 2, OccupancyCode.VICTIM))
        text = GridComparison.between(truth, _grid()).summary()
        for expected in ("cell agreement", "traversability", "missed hazards", "missed victims"):
            assert expected in text
        assert "victim" in text and "fire" in text

    def test_the_summary_is_a_single_block_of_text(self) -> None:
        text = GridComparison.between(_grid(), _grid()).summary()
        assert isinstance(text, str)
        assert len(text.splitlines()) == len(OccupancyCode) + 8


class TestTraversabilityMask:
    def test_traversability_is_derived_from_the_code_contract(self) -> None:
        """Not a second copy of which codes are walls — the same one."""
        truth = OccupancyGrid(np.array([[code for code in OccupancyCode]], dtype=np.uint8))
        result = GridComparison.between(truth, truth)
        assert result.traversability_agreement == 1.0
        assert result.missed_hazards == frozenset()
