"""Unit tests for sentry_ai.decision.fusion.

The camera features are the part of fusion that must be right by
construction, not by training: debris on the tile in front must read as
"ahead" in every heading, and a tile off the map's edge must simply read as
empty. The network is tested for its contract — shapes, masking, checkpoint
round-trip — and the camera-veto rule for its one decision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="fusion needs torch (requirements-ml.txt)")

from sentry_ai.common.exceptions import AssetNotFoundError  # noqa: E402
from sentry_ai.decision.fusion import (  # noqa: E402
    CAMERA_TILES,
    CHECKPOINT_FORMAT,
    FEATURE_GROUPS,
    FUSION_FEATURES,
    CameraVetoFusion,
    FusionArchitecture,
    FusionNet,
    MlpFusion,
    camera_features,
    fusion_features,
    group_mask,
)
from sentry_ai.domain.entities import Position  # noqa: E402
from sentry_ai.domain.enums import EntityKind, Heading  # noqa: E402
from sentry_ai.interfaces.decision import FinalAction, IDecisionFusion  # noqa: E402
from sentry_ai.interfaces.navigation import (  # noqa: E402
    LOCAL_ACTION_ORDER,
    LocalAction,
    LocalDecision,
    LocalObservation,
)
from sentry_ai.interfaces.perception import WorldDetection  # noqa: E402
from sentry_ai.interfaces.sequence import BehaviourClass, BehaviourSignal  # noqa: E402

_ROWS = {name: index for index, (name, _) in enumerate(CAMERA_TILES)}
_OBSTACLE = 2  # YOLO_CLASSES order: victim, fire, obstacle


def _observation(heading: Heading, x: int = 10, y: int = 10) -> LocalObservation:
    return LocalObservation(
        position=Position(x, y),
        heading=heading,
        battery_percent=70.0,
        next_waypoint=Position(x + heading.delta[0], y + heading.delta[1]),
        blocked_ahead=False,
        fire_proximity=0.0,
    )


def _debris(x: int, y: int, confidence: float = 0.9) -> WorldDetection:
    return WorldDetection(EntityKind.OBSTACLE, frozenset({Position(x, y)}), confidence)


def _decision(action: LocalAction = LocalAction.MOVE_FORWARD) -> LocalDecision:
    return LocalDecision(
        action=action,
        q_values={a: (1.0 if a is action else 0.0) for a in LOCAL_ACTION_ORDER},
    )


_BEHAVIOUR = BehaviourSignal(BehaviourClass.ADVANCE, 0.8)


def _camera(observation: LocalObservation, sightings: list[WorldDetection]) -> np.ndarray:
    return camera_features(observation, sightings).reshape(len(CAMERA_TILES), -1)


class TestCameraFeatures:
    @pytest.mark.parametrize("heading", list(Heading))
    def test_debris_in_front_reads_as_ahead_in_every_heading(self, heading: Heading) -> None:
        dx, dy = heading.delta
        camera = _camera(_observation(heading), [_debris(10 + dx, 10 + dy)])
        assert camera[_ROWS["ahead"], _OBSTACLE] == pytest.approx(0.9)
        assert camera.sum() == pytest.approx(0.9)

    @pytest.mark.parametrize("heading", list(Heading))
    def test_debris_behind_reads_as_behind(self, heading: Heading) -> None:
        dx, dy = heading.delta
        camera = _camera(_observation(heading), [_debris(10 - dx, 10 - dy)])
        assert camera[_ROWS["behind"], _OBSTACLE] == pytest.approx(0.9)

    def test_right_is_clockwise_of_ahead(self) -> None:
        """Facing north, the tile to the east is on the right."""
        camera = _camera(_observation(Heading.NORTH), [_debris(11, 10)])
        assert camera[_ROWS["right"], _OBSTACLE] == pytest.approx(0.9)

    def test_the_strongest_sighting_wins(self) -> None:
        camera = _camera(_observation(Heading.EAST), [_debris(11, 10, 0.3), _debris(11, 10, 0.7)])
        assert camera[_ROWS["ahead"], _OBSTACLE] == pytest.approx(0.7)

    def test_a_fire_footprint_counts_on_every_tile_it_covers(self) -> None:
        fire = WorldDetection(
            EntityKind.FIRE, frozenset({Position(11, 10), Position(12, 10)}), 0.6
        )
        camera = _camera(_observation(Heading.EAST), [fire])
        assert camera[_ROWS["ahead"], 1] == pytest.approx(0.6)
        assert camera[_ROWS["ahead2"], 1] == pytest.approx(0.6)

    def test_at_the_map_edge_off_map_tiles_read_empty(self) -> None:
        """Behind a vehicle at (0, 0) facing east is (-1, 0): no crash, no signal."""
        assert camera_features(_observation(Heading.EAST, 0, 0), [_debris(1, 0)]).sum() > 0

    def test_distant_sightings_are_ignored(self) -> None:
        assert camera_features(_observation(Heading.NORTH), [_debris(20, 3)]).sum() == 0.0


class TestFusionFeatures:
    def test_the_vector_has_the_declared_layout(self) -> None:
        features = fusion_features(_observation(Heading.NORTH), [], _BEHAVIOUR, _decision())
        assert features.shape == (FUSION_FEATURES,)
        assert sum(width for _, width in FEATURE_GROUPS) == FUSION_FEATURES

    def test_dqn_values_become_a_distribution(self) -> None:
        features = fusion_features(_observation(Heading.NORTH), [], _BEHAVIOUR, _decision())
        dqn = features[: len(LOCAL_ACTION_ORDER)]
        assert dqn.sum() == pytest.approx(1.0)
        assert int(dqn.argmax()) == LOCAL_ACTION_ORDER.index(LocalAction.MOVE_FORWARD)

    def test_the_behaviour_is_one_hot_times_confidence(self) -> None:
        features = fusion_features(_observation(Heading.NORTH), [], _BEHAVIOUR, _decision())
        lstm = features[len(LOCAL_ACTION_ORDER) : len(LOCAL_ACTION_ORDER) + 4]
        assert lstm.tolist() == pytest.approx([0.8, 0.0, 0.0, 0.0])


class TestMasking:
    def test_a_mask_covers_exactly_its_groups(self) -> None:
        mask = group_mask(("dqn",))
        assert mask[: len(LOCAL_ACTION_ORDER)].sum() == len(LOCAL_ACTION_ORDER)
        assert mask.sum() == len(LOCAL_ACTION_ORDER)

    def test_unknown_groups_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            group_mask(("dqn", "radar"))

    def test_masked_features_cannot_change_the_output(self) -> None:
        """The single-signal baselines really see only their signal."""
        torch.manual_seed(0)
        network = FusionNet(FusionArchitecture(groups=("dqn",), dropout=0.0)).eval()
        base = torch.rand(1, FUSION_FEATURES)
        changed = base.clone()
        width = len(LOCAL_ACTION_ORDER)
        changed[0, width:] = torch.rand(FUSION_FEATURES - width)
        assert torch.equal(network(base), network(changed))

    @pytest.mark.parametrize(
        "overrides",
        [{"groups": ()}, {"hidden_sizes": ()}, {"hidden_sizes": (0,)}, {"dropout": 1.0}],
    )
    def test_degenerate_architectures_are_rejected(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            FusionArchitecture(**overrides)  # type: ignore[arg-type]


class TestMlpFusion:
    def _fusion(self) -> MlpFusion:
        torch.manual_seed(0)
        return MlpFusion(FusionNet(FusionArchitecture()))

    def test_it_is_an_idecisionfusion(self) -> None:
        assert isinstance(self._fusion(), IDecisionFusion)

    def test_it_returns_a_valid_final_action(self) -> None:
        observation = _observation(Heading.EAST)
        final = self._fusion().fuse(observation, [_debris(11, 10)], _BEHAVIOUR, _decision())
        assert isinstance(final, FinalAction)
        assert final.action in LOCAL_ACTION_ORDER
        assert 0.0 <= final.rationale_score <= 1.0

    def test_a_checkpoint_rebuilds_the_same_network(self, tmp_path: Path) -> None:
        torch.manual_seed(1)
        network = FusionNet(FusionArchitecture(groups=("dqn", "camera"), hidden_sizes=(8,)))
        path = tmp_path / "best.pt"
        torch.save(network.to_checkpoint(), path)
        restored = MlpFusion.from_checkpoint(path)
        args = (_observation(Heading.SOUTH), [_debris(10, 11)], _BEHAVIOUR, _decision())
        assert restored.fuse(*args) == MlpFusion(network).fuse(*args)

    def test_an_unknown_format_fails_clearly(self) -> None:
        checkpoint = FusionNet(FusionArchitecture()).to_checkpoint()
        checkpoint["format"] = CHECKPOINT_FORMAT + 1
        with pytest.raises(ValueError, match="checkpoint format"):
            FusionNet.from_checkpoint(checkpoint)

    def test_missing_weights_fail_before_anything_loads(self, tmp_path: Path) -> None:
        with pytest.raises(AssetNotFoundError):
            MlpFusion.from_checkpoint(tmp_path / "absent.pt")


class TestCameraVetoRule:
    def test_it_stops_before_debris_ahead(self) -> None:
        final = CameraVetoFusion().fuse(
            _observation(Heading.EAST), [_debris(11, 10)], _BEHAVIOUR, _decision()
        )
        assert final.action is LocalAction.STOP

    def test_it_stops_before_reversing_into_debris(self) -> None:
        final = CameraVetoFusion().fuse(
            _observation(Heading.EAST), [_debris(9, 10)], _BEHAVIOUR, _decision(LocalAction.REVERSE)
        )
        assert final.action is LocalAction.STOP

    def test_turning_is_never_vetoed(self) -> None:
        final = CameraVetoFusion().fuse(
            _observation(Heading.EAST),
            [_debris(11, 10)],
            _BEHAVIOUR,
            _decision(LocalAction.TURN_LEFT),
        )
        assert final.action is LocalAction.TURN_LEFT

    def test_weak_sightings_are_ignored(self) -> None:
        final = CameraVetoFusion(threshold=0.5).fuse(
            _observation(Heading.EAST), [_debris(11, 10, 0.2)], _BEHAVIOUR, _decision()
        )
        assert final.action is LocalAction.MOVE_FORWARD

    def test_a_victim_ahead_is_not_a_hazard(self) -> None:
        victim = WorldDetection(EntityKind.VICTIM, frozenset({Position(11, 10)}), 0.99)
        observation = _observation(Heading.EAST)
        final = CameraVetoFusion().fuse(observation, [victim], _BEHAVIOUR, _decision())
        assert final.action is LocalAction.MOVE_FORWARD
