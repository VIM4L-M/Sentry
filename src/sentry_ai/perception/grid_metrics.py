"""Scores a detector-derived occupancy grid against ground truth (Phase 3.3).

:meth:`~sentry_ai.domain.occupancy.OccupancyGrid.from_city_map` stays in the
codebase for exactly one reason: it is the answer key. This module marks the
paper.

**Cell accuracy is the least interesting number here.** The grid is ~97%
road and building, so a belief that detected nothing at all would still
score in the nineties. What matters is what a *wrong* cell does to the
mission, and the three ways it can go wrong are not equivalent:

* A **missed hazard** — believed clear, actually fire or debris — drives the
  vehicle into it. This is the dangerous error.
* A **phantom obstacle** — believed blocked, actually clear — costs a detour,
  and if it seals the last corridor it makes a victim unreachable.
* A **missed victim** — never marked ``VICTIM`` — means nobody is ever
  dispatched. The person is not rescued, and no amount of accuracy
  elsewhere compensates.

So this module reports per-class precision/recall *and* names the specific
tiles behind each failure mode, because "which tiles" is what you need to
debug a mission that went wrong.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sentry_ai.common.exceptions import DomainValidationError
from sentry_ai.domain.entities import Position
from sentry_ai.domain.occupancy import OccupancyCode, OccupancyGrid


def _traversable_mask(cells: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Whether each cell may be routed through, vectorised over the array."""
    lookup = np.array([code.is_traversable for code in sorted(OccupancyCode)], dtype=bool)
    return lookup[cells]


def _positions(mask: NDArray[np.bool_]) -> frozenset[Position]:
    """Every ``(x, y)`` where ``mask`` is set."""
    ys, xs = np.nonzero(mask)
    return frozenset(Position(int(x), int(y)) for y, x in zip(ys, xs, strict=True))


@dataclass(frozen=True)
class CodeScore:
    """How well one occupancy code was recovered, counted in cells.

    Attributes:
        code: The code being scored.
        true_positives: Cells holding ``code`` in both grids.
        false_positives: Cells the belief claims hold ``code`` and truth does not.
        false_negatives: Cells truth holds ``code`` and the belief missed.
    """

    code: OccupancyCode
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Share of claimed cells that were right.

        A belief that claims this code nowhere, against a truth that holds it
        nowhere, scores ``1.0``: nothing was claimed, so nothing was claimed
        wrongly. Reporting ``0.0`` there would penalise a correct grid.
        """
        claimed = self.true_positives + self.false_positives
        if claimed == 0:
            return 1.0
        return self.true_positives / claimed

    @property
    def recall(self) -> float:
        """Share of actual cells that were found. Empty-vs-empty scores ``1.0``."""
        actual = self.true_positives + self.false_negatives
        if actual == 0:
            return 1.0
        return self.true_positives / actual

    @property
    def f1(self) -> float:
        """Harmonic mean of :attr:`precision` and :attr:`recall`."""
        total = self.precision + self.recall
        if total == 0.0:
            return 0.0
        return 2 * self.precision * self.recall / total

    @property
    def iou(self) -> float:
        """Intersection over union of the two cell sets."""
        union = self.true_positives + self.false_positives + self.false_negatives
        if union == 0:
            return 1.0
        return self.true_positives / union


@dataclass(frozen=True)
class GridComparison:
    """One belief grid, marked against ground truth.

    Attributes:
        scores: Per-code cell counts, keyed by :class:`OccupancyCode`.
        missed_hazards: Believed traversable, actually not. The vehicle would
            drive into these.
        phantom_obstacles: Believed blocked, actually clear. These cost
            detours and can make a victim unreachable.
        missed_victims: Held ``VICTIM`` in truth, not in the belief. Nobody
            is dispatched to these tiles.
        phantom_victims: Held ``VICTIM`` in the belief, not in truth. These
            waste a trip.
        total_cells: Size of the grids compared.
        matching_cells: Cells whose code is identical in both.
        traversability_matches: Cells the two grids agree are drivable — or
            agree are not. A cell can be wrong on code and right here, which
            is the distinction the planner actually cares about.
    """

    scores: Mapping[OccupancyCode, CodeScore]
    missed_hazards: frozenset[Position]
    phantom_obstacles: frozenset[Position]
    missed_victims: frozenset[Position]
    phantom_victims: frozenset[Position]
    total_cells: int
    matching_cells: int
    traversability_matches: int

    @classmethod
    def between(cls, truth: OccupancyGrid, belief: OccupancyGrid) -> GridComparison:
        """Score ``belief`` against ``truth``.

        Raises:
            DomainValidationError: If the grids are different shapes, which
                means one of them was built for a different city.
        """
        if truth.cells.shape != belief.cells.shape:
            raise DomainValidationError(
                f"cannot compare a {truth.width}x{truth.height} grid with a "
                f"{belief.width}x{belief.height} one"
            )
        truth_drivable = _traversable_mask(truth.cells)
        belief_drivable = _traversable_mask(belief.cells)
        is_victim = OccupancyCode.VICTIM

        return cls(
            scores={code: cls._score(truth, belief, code) for code in OccupancyCode},
            missed_hazards=_positions(belief_drivable & ~truth_drivable),
            phantom_obstacles=_positions(~belief_drivable & truth_drivable),
            missed_victims=_positions((truth.cells == is_victim) & (belief.cells != is_victim)),
            phantom_victims=_positions((belief.cells == is_victim) & (truth.cells != is_victim)),
            total_cells=truth.cells.size,
            matching_cells=int(np.count_nonzero(truth.cells == belief.cells)),
            traversability_matches=int(np.count_nonzero(truth_drivable == belief_drivable)),
        )

    @staticmethod
    def _score(truth: OccupancyGrid, belief: OccupancyGrid, code: OccupancyCode) -> CodeScore:
        in_truth = truth.cells == code
        in_belief = belief.cells == code
        return CodeScore(
            code=code,
            true_positives=int(np.count_nonzero(in_truth & in_belief)),
            false_positives=int(np.count_nonzero(~in_truth & in_belief)),
            false_negatives=int(np.count_nonzero(in_truth & ~in_belief)),
        )

    @property
    def cell_agreement(self) -> float:
        """Share of cells whose code matches exactly.

        Read this with suspicion — see the module docstring. Most of the map
        is static terrain the builder copies from the survey, so this number
        is high before perception has done anything.
        """
        return self.matching_cells / self.total_cells

    @property
    def traversability_agreement(self) -> float:
        """Share of cells the two grids agree are drivable or not.

        The number A\\* would care about: it never reads a code, only whether
        it may pass.
        """
        return self.traversability_matches / self.total_cells

    @property
    def is_drivable(self) -> bool:
        """Whether the belief contains no hazard the vehicle would drive into."""
        return not self.missed_hazards

    @property
    def finds_every_victim(self) -> bool:
        """Whether every victim in the world is on the belief map."""
        return not self.missed_victims

    def summary(self) -> str:
        """A short human-readable report, for scripts and mission logs."""
        lines = [
            f"cell agreement           {self.cell_agreement:.4f}",
            f"traversability agreement {self.traversability_agreement:.4f}",
            f"missed hazards           {len(self.missed_hazards)}",
            f"phantom obstacles        {len(self.phantom_obstacles)}",
            f"missed victims           {len(self.missed_victims)}",
            f"phantom victims          {len(self.phantom_victims)}",
            "",
            f"{'code':<10}{'prec':>8}{'recall':>8}{'IoU':>8}{'truth':>8}",
        ]
        lines.extend(
            f"{score.code.name.lower():<10}{score.precision:>8.3f}{score.recall:>8.3f}"
            f"{score.iou:>8.3f}{score.true_positives + score.false_negatives:>8d}"
            for score in self.scores.values()
        )
        return "\n".join(lines)
