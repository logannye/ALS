"""9-component reward computation for the research loop.

Reward signals are weighted to prioritize causal depth and gap closure
over evidence breadth — reflecting the system's drug discovery mission.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

WEIGHTS = {
    "evidence_gain": 1.5,           # Deprioritized: breadth has diminishing returns at 3K+ items
    "uncertainty_reduction": 1.5,   # Deprioritized: layer coverage is not the bottleneck
    "protocol_improvement": 4.0,    # Raised: protocol synthesis drives convergence
    "hypothesis_resolution": 4.0,   # Raised: closing knowledge gaps advances drug discovery
    "causal_depth": 5.0,            # PRIMARY: deep mechanistic understanding IS the mission
    "interaction_safety": 2.0,      # Unchanged: safety is non-negotiable
    "erik_eligibility": 3.0,        # Raised: trial access is time-sensitive (ALSFRS-R declining)
    "convergence_bonus": 1.0,       # Unchanged
    "gap_closure": 4.0,             # Closing structured causal gaps directly advances drug design
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
    gap_closure: float = 0.0

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
            + WEIGHTS["gap_closure"] * self.gap_closure
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
            "gap_closure": self.gap_closure,
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
    gaps_updated: int = 0,
) -> RewardComponents:
    evidence_gain = math.log1p(evidence_items_added) if evidence_items_added > 0 else 0.0
    uncertainty_reduction = max(0.0, uncertainty_before - uncertainty_after)
    protocol_improvement = max(0.0, protocol_score_delta)
    hypothesis_resolution_val = 1.0 if hypothesis_resolved else 0.0
    interaction_safety_val = 1.0 if interaction_safe else 0.0
    eligibility_val = 1.0 if eligibility_confirmed else 0.0
    convergence_val = 1.0 if protocol_stable else 0.0
    causal_depth_val = math.log1p(causal_depth_added) if causal_depth_added > 0 else 0.0
    # Mild discount: depth without simultaneous evidence is still valuable
    # but grounding in data prevents hollow reasoning loops
    if causal_depth_added > 0 and evidence_items_added == 0:
        causal_depth_val *= 0.5
    gap_closure_val = math.log1p(gaps_updated) if gaps_updated > 0 else 0.0

    return RewardComponents(
        evidence_gain=evidence_gain,
        uncertainty_reduction=uncertainty_reduction,
        protocol_improvement=protocol_improvement,
        hypothesis_resolution=hypothesis_resolution_val,
        causal_depth=causal_depth_val,
        interaction_safety=interaction_safety_val,
        erik_eligibility=eligibility_val,
        convergence_bonus=convergence_val,
        gap_closure=gap_closure_val,
    )
