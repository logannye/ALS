"""LearningEpisode construction for the research loop."""
from __future__ import annotations
from typing import Optional
from ontology.meta import LearningEpisode
from research.actions import ActionResult
from research.rewards import RewardComponents

def build_episode(
    step_count: int,
    subject_ref: str,
    action_result: ActionResult,
    reward: RewardComponents,
    protocol_ref: Optional[str] = None,
    state_snapshot_ref: Optional[str] = None,
) -> LearningEpisode:
    return LearningEpisode(
        id=f"episode:{subject_ref.split(':')[-1]}_{step_count:05d}",
        subject_ref=subject_ref,
        trigger=f"step:{step_count}",
        state_snapshot_ref=state_snapshot_ref,
        protocol_ref=protocol_ref,
        body={
            "action": action_result.action.value,
            "success": action_result.success,
            "error": action_result.error,
            "evidence_items_added": action_result.evidence_items_added,
            "interventions_added": action_result.interventions_added,
            "hypothesis_generated": action_result.hypothesis_generated,
            "hypothesis_resolved": action_result.hypothesis_resolved,
            "causal_depth_added": action_result.causal_depth_added,
            "protocol_regenerated": action_result.protocol_regenerated,
            "reward_components": reward.to_dict(),
            "reward_total": reward.total(),
        },
    )
