"""8-component reward computation for the research loop."""
from __future__ import annotations
import math
from dataclasses import dataclass

WEIGHTS = {
    "evidence_gain": 3.0,
    "uncertainty_reduction": 4.0,
    "protocol_improvement": 3.5,
    "hypothesis_resolution": 2.5,
    "causal_depth": 2.0,
    "interaction_safety": 2.0,
    "erik_eligibility": 1.5,
    "convergence_bonus": 1.0,
}

@dataclass
class RewardComponents:
    """Individual reward signal components before weighting."""
    evidence_gain: float = 0.0
    uncertainty_reduction: float = 0.0
    protocol_improvement: float = 0.0
    hypothesis_resolution: float = 0.0
    causal_depth: float = 0.0
    interaction_safety: float = 0.0
    erik_eligibility: float = 0.0
    convergence_bonus: float = 0.0

    def total(self) -> float:
        return (
            WEIGHTS["evidence_gain"] * self.evidence_gain
            + WEIGHTS["uncertainty_reduction"] * self.uncertainty_reduction
            + WEIGHTS["protocol_improvement"] * self.protocol_improvement
            + WEIGHTS["hypothesis_resolution"] * self.hypothesis_resolution
            + WEIGHTS["causal_depth"] * self.causal_depth
            + WEIGHTS["interaction_safety"] * self.interaction_safety
            + WEIGHTS["erik_eligibility"] * self.erik_eligibility
            + WEIGHTS["convergence_bonus"] * self.convergence_bonus
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "evidence_gain": self.evidence_gain,
            "uncertainty_reduction": self.uncertainty_reduction,
            "protocol_improvement": self.protocol_improvement,
            "hypothesis_resolution": self.hypothesis_resolution,
            "causal_depth": self.causal_depth,
            "interaction_safety": self.interaction_safety,
            "erik_eligibility": self.erik_eligibility,
            "convergence_bonus": self.convergence_bonus,
            "total": self.total(),
        }

def compute_reward(
    evidence_items_added: int,
    uncertainty_before: float,
    uncertainty_after: float,
    protocol_score_delta: float,
    hypothesis_resolved: bool,
    causal_depth_added: int,
    interaction_safe: bool,
    eligibility_confirmed: bool,
    protocol_stable: bool,
) -> RewardComponents:
    evidence_gain = math.log1p(evidence_items_added) if evidence_items_added > 0 else 0.0
    uncertainty_reduction = max(0.0, uncertainty_before - uncertainty_after)
    protocol_improvement = max(0.0, protocol_score_delta)
    hypothesis_resolution_val = 1.0 if hypothesis_resolved else 0.0
    interaction_safety_val = 1.0 if interaction_safe else 0.0
    eligibility_val = 1.0 if eligibility_confirmed else 0.0
    convergence_val = 1.0 if protocol_stable else 0.0
    causal_depth_val = math.log1p(causal_depth_added) if causal_depth_added > 0 else 0.0
    # Diminishing returns: depth without evidence is 90% discounted
    if causal_depth_added > 0 and evidence_items_added == 0:
        causal_depth_val *= 0.1

    return RewardComponents(
        evidence_gain=evidence_gain,
        uncertainty_reduction=uncertainty_reduction,
        protocol_improvement=protocol_improvement,
        hypothesis_resolution=hypothesis_resolution_val,
        causal_depth=causal_depth_val,
        interaction_safety=interaction_safety_val,
        erik_eligibility=eligibility_val,
        convergence_bonus=convergence_val,
    )
