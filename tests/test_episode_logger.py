"""Tests for LearningEpisode persistence."""
from __future__ import annotations
import pytest
from research.episode_logger import build_episode
from research.actions import ActionType, ActionResult
from research.rewards import RewardComponents

class TestBuildEpisode:
    def test_builds_valid_episode(self):
        result = ActionResult(action=ActionType.SEARCH_PUBMED, evidence_items_added=5)
        reward = RewardComponents(evidence_gain=1.6)
        episode = build_episode(
            step_count=42, subject_ref="traj:draper_001",
            action_result=result, reward=reward, protocol_ref="proto:draper_001_v2",
        )
        assert episode.type == "LearningEpisode"
        assert episode.trigger == "step:42"
        assert episode.subject_ref == "traj:draper_001"
        assert episode.protocol_ref == "proto:draper_001_v2"
        assert episode.body["action"] == "search_pubmed"
        assert episode.body["evidence_items_added"] == 5
        assert episode.body["reward_total"] == reward.total()

    def test_episode_id_format(self):
        result = ActionResult(action=ActionType.GENERATE_HYPOTHESIS)
        reward = RewardComponents()
        episode = build_episode(
            step_count=1, subject_ref="traj:draper_001",
            action_result=result, reward=reward,
        )
        assert episode.id.startswith("episode:")
