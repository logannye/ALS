"""ResearchState — point-in-time snapshot of the autonomous research loop."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from ontology.enums import ProtocolLayer

ALL_LAYERS = [layer.value for layer in ProtocolLayer]
ALL_STRENGTHS = ["strong", "moderate", "emerging", "preclinical", "unknown"]

@dataclass
class ResearchState:
    """Point-in-time snapshot of the research loop's knowledge."""
    subject_ref: str
    current_protocol_id: Optional[str] = None
    protocol_version: int = 0
    protocol_stable_cycles: int = 0
    total_evidence_items: int = 0
    evidence_by_layer: dict[str, int] = field(default_factory=dict)
    evidence_by_strength: dict[str, int] = field(default_factory=dict)
    active_hypotheses: list[str] = field(default_factory=list)
    resolved_hypotheses: int = 0
    causal_chains: dict[str, int] = field(default_factory=dict)
    top_uncertainties: list[str] = field(default_factory=list)
    missing_measurements: list[str] = field(default_factory=list)
    step_count: int = 0
    action_values: dict[str, float] = field(default_factory=dict)
    action_counts: dict[str, int] = field(default_factory=dict)
    last_action: str = ""
    last_reward: float = 0.0
    converged: bool = False
    new_evidence_since_regen: int = 0
    uncertainty_score: float = 1.0
    uncertainty_history: list[float] = field(default_factory=list)
    challenge_counts: dict[str, int] = field(default_factory=dict)
    action_posteriors: dict[str, tuple[float, float]] = field(default_factory=dict)
    last_action_per_type: dict[str, int] = field(default_factory=dict)
    last_gap_layers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_ref": self.subject_ref,
            "current_protocol_id": self.current_protocol_id,
            "protocol_version": self.protocol_version,
            "protocol_stable_cycles": self.protocol_stable_cycles,
            "total_evidence_items": self.total_evidence_items,
            "evidence_by_layer": dict(self.evidence_by_layer),
            "evidence_by_strength": dict(self.evidence_by_strength),
            "active_hypotheses": list(self.active_hypotheses),
            "resolved_hypotheses": self.resolved_hypotheses,
            "causal_chains": dict(self.causal_chains),
            "top_uncertainties": list(self.top_uncertainties),
            "missing_measurements": list(self.missing_measurements),
            "step_count": self.step_count,
            "action_values": dict(self.action_values),
            "action_counts": dict(self.action_counts),
            "last_action": self.last_action,
            "last_reward": self.last_reward,
            "converged": self.converged,
            "new_evidence_since_regen": self.new_evidence_since_regen,
            "uncertainty_score": self.uncertainty_score,
            "uncertainty_history": list(self.uncertainty_history),
            "challenge_counts": dict(self.challenge_counts),
            "action_posteriors": {k: list(v) for k, v in self.action_posteriors.items()},
            "last_action_per_type": dict(self.last_action_per_type),
            "last_gap_layers": list(self.last_gap_layers),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResearchState:
        clean = {}
        for k, v in d.items():
            if k not in cls.__dataclass_fields__:
                continue
            if k == "action_posteriors" and isinstance(v, dict):
                clean[k] = {key: tuple(val) for key, val in v.items()}
            else:
                clean[k] = v
        return cls(**clean)


def initial_state(subject_ref: str) -> ResearchState:
    return ResearchState(
        subject_ref=subject_ref,
        evidence_by_layer={layer: 0 for layer in ALL_LAYERS},
        evidence_by_strength={s: 0 for s in ALL_STRENGTHS},
        missing_measurements=[
            "genetic_testing", "csf_biomarkers",
            "cryptic_exon_splicing_assay", "tdp43_in_vivo_measurement",
            "cortical_excitability_tms", "transcriptomics", "proteomics",
        ],
    )
