"""Unit tests for sentry_ai.decision.mlp_fusion (Phase 7, Unit I)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sentry_ai.common.exceptions import AssetNotFoundError
from sentry_ai.decision.mlp_fusion import (
    FUSION_FEATURES,
    GROUP_WIDTHS,
    FeatureGroup,
    FusionArchitecture,
    FusionNet,
    MlpFusion,
    fusion_features,
    group_mask,
)
from sentry_ai.domain.enums import EntityKind
from sentry_ai.interfaces.decision import Region, SceneEvidence
from sentry_ai.interfaces.navigation import LOCAL_ACTION_ORDER, LocalAction, LocalDecision
from sentry_ai.interfaces.sequence import BehaviourClass, BehaviourSignal

POLICY = GROUP_WIDTHS[FeatureGroup.POLICY]
BEHAVIOUR = GROUP_WIDTHS[FeatureGroup.BEHAVIOUR]


def _decision(best: LocalAction = LocalAction.MOVE_FORWARD) -> LocalDecision:
    return LocalDecision(
        action=best,
        q_values={action: (2.0 if action is best else -1.0) for action in LOCAL_ACTION_ORDER},
    )


def _signal() -> BehaviourSignal:
    return BehaviourSignal(BehaviourClass.DIVERT, 0.6)


class TestFusionFeatures:
    def test_has_the_documented_length(self) -> None:
        features = fusion_features(SceneEvidence.empty(), _signal(), _decision())
        assert features.shape == (FUSION_FEATURES,)
        assert features.dtype == np.float32

    def test_policy_block_is_zero_at_the_dqn_choice_and_negative_elsewhere(self) -> None:
        features = fusion_features(SceneEvidence.empty(), _signal(), _decision(LocalAction.STOP))
        policy = features[:POLICY]
        assert policy[LOCAL_ACTION_ORDER.index(LocalAction.STOP)] == 0.0
        assert policy.min() == pytest.approx(-1.0)

    def test_equal_q_values_give_a_zero_policy_block(self) -> None:
        flat = LocalDecision(LocalAction.STOP, {action: 1.0 for action in LOCAL_ACTION_ORDER})
        features = fusion_features(SceneEvidence.empty(), _signal(), flat)
        assert not features[:POLICY].any()

    def test_behaviour_is_one_hot_times_confidence(self) -> None:
        features = fusion_features(SceneEvidence.empty(), _signal(), _decision())
        behaviour = features[POLICY : POLICY + BEHAVIOUR]
        assert sorted(behaviour.tolist()) == pytest.approx([0.0, 0.0, 0.0, 0.6])

    def test_camera_block_carries_the_evidence(self) -> None:
        evidence = SceneEvidence({(EntityKind.OBSTACLE, Region.AHEAD): 0.95})
        features = fusion_features(evidence, _signal(), _decision())
        assert features[POLICY + BEHAVIOUR :].max() == pytest.approx(0.95)


class TestGroupMask:
    def test_masks_everything_but_the_chosen_group(self) -> None:
        mask = group_mask((FeatureGroup.CAMERA,))
        assert mask[: POLICY + BEHAVIOUR].sum() == 0.0
        assert mask[POLICY + BEHAVIOUR :].all()


class TestFusionNet:
    def test_maps_features_to_one_logit_per_action(self) -> None:
        network = FusionNet(FusionArchitecture())
        logits = network(torch.zeros(3, FUSION_FEATURES))
        assert logits.shape == (3, len(LOCAL_ACTION_ORDER))

    def test_untrained_network_already_agrees_with_the_dqn(self) -> None:
        torch.manual_seed(0)
        fusion = MlpFusion(FusionNet(FusionArchitecture(dropout=0.0)))
        final = fusion.fuse(SceneEvidence.empty(), _signal(), _decision(LocalAction.TURN_LEFT))
        assert final.action is LocalAction.TURN_LEFT

    def test_a_camera_only_network_ignores_the_policy(self) -> None:
        network = FusionNet(FusionArchitecture(groups=(FeatureGroup.CAMERA,))).eval()
        a = fusion_features(SceneEvidence.empty(), _signal(), _decision(LocalAction.STOP))
        b = fusion_features(SceneEvidence.empty(), _signal(), _decision(LocalAction.REVERSE))
        with torch.no_grad():
            out = network(torch.from_numpy(np.stack([a, b])))
        assert torch.allclose(out[0], out[1])

    def test_checkpoint_round_trip_keeps_weights_and_groups(self) -> None:
        network = FusionNet(FusionArchitecture(groups=(FeatureGroup.POLICY,)))
        restored = FusionNet.from_checkpoint(network.to_checkpoint())
        assert restored.architecture == network.architecture
        for mine, theirs in zip(
            network.state_dict().values(), restored.state_dict().values(), strict=True
        ):
            assert torch.equal(mine, theirs)

    def test_rejects_an_unknown_checkpoint_format(self) -> None:
        checkpoint = FusionNet(FusionArchitecture()).to_checkpoint()
        checkpoint["format"] = 99
        with pytest.raises(ValueError, match="format"):
            FusionNet.from_checkpoint(checkpoint)

    def test_architecture_needs_a_group(self) -> None:
        with pytest.raises(ValueError):
            FusionArchitecture(groups=())


class TestMlpFusion:
    def test_probabilities_sum_to_one(self) -> None:
        fusion = MlpFusion(FusionNet(FusionArchitecture()))
        probabilities = fusion.probabilities(SceneEvidence.empty(), _signal(), _decision())
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert set(probabilities) == set(LOCAL_ACTION_ORDER)

    def test_rationale_score_is_the_winning_probability(self) -> None:
        fusion = MlpFusion(FusionNet(FusionArchitecture()))
        probabilities = fusion.probabilities(SceneEvidence.empty(), _signal(), _decision())
        final = fusion.fuse(SceneEvidence.empty(), _signal(), _decision())
        assert final.rationale_score == pytest.approx(max(probabilities.values()))

    def test_loads_from_a_saved_checkpoint(self, tmp_path: Path) -> None:
        path = tmp_path / "best.pt"
        torch.save(FusionNet(FusionArchitecture()).to_checkpoint(), path)
        assert MlpFusion.from_checkpoint(path).groups == tuple(FeatureGroup)

    def test_missing_weights_raise(self, tmp_path: Path) -> None:
        with pytest.raises(AssetNotFoundError):
            MlpFusion.from_checkpoint(tmp_path / "absent.pt")


class TestOverrideThreshold:
    def test_an_unsure_override_leaves_the_dqn_in_charge(self) -> None:
        torch.manual_seed(0)
        network = FusionNet(FusionArchitecture(groups=(FeatureGroup.CAMERA,), dropout=0.0))
        free = MlpFusion(network)
        strict = MlpFusion(network, override_threshold=1.0)
        decision = _decision(LocalAction.REVERSE)
        assert (
            free.fuse(SceneEvidence.empty(), _signal(), decision).action is not LocalAction.REVERSE
        )
        assert strict.fuse(SceneEvidence.empty(), _signal(), decision).action is LocalAction.REVERSE
