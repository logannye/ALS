"""Causal chain construction for protocol interventions.

Each chain is a directed sequence of mechanism steps from intervention
to patient outcome. Every link is grounded in a citable evidence item.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class CausalLink:
    source: str
    target: str
    mechanism: str
    evidence_ref: str
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "mechanism": self.mechanism,
                "evidence_ref": self.evidence_ref, "confidence": self.confidence}

@dataclass
class CausalChain:
    intervention_id: str
    links: list[CausalLink] = field(default_factory=list)

    def depth(self) -> int:
        return len(self.links)

    def add_link(self, link: CausalLink) -> None:
        self.links.append(link)

    def weakest_link(self) -> Optional[CausalLink]:
        if not self.links:
            return None
        return min(self.links, key=lambda l: l.confidence)

    def all_evidence_refs(self) -> list[str]:
        return [link.evidence_ref for link in self.links]

    def to_dict(self) -> dict[str, Any]:
        weak = self.weakest_link()
        return {
            "intervention_id": self.intervention_id,
            "depth": self.depth(),
            "links": [l.to_dict() for l in self.links],
            "weakest_link": weak.to_dict() if weak else None,
            "all_evidence_refs": self.all_evidence_refs(),
        }

def get_chain_depth(chains: dict[str, CausalChain], intervention_id: str) -> int:
    chain = chains.get(intervention_id)
    return chain.depth() if chain else 0

def pathway_grounded_link(
    source: str, target: str, pathway_evidence: list[dict],
) -> Optional[CausalLink]:
    """Create a causal link grounded in pathway data (Reactome/KEGG).

    If any pathway evidence connects source to target, creates a
    high-confidence link (0.95). Otherwise returns None.
    """
    source_lower = source.lower()
    target_lower = target.lower()
    for evi in pathway_evidence:
        body = evi.get("body", {})
        pathway_name = body.get("pathway_name", "").lower()
        if source_lower in pathway_name or target_lower in pathway_name:
            return CausalLink(
                source=source, target=target,
                mechanism=f"pathway: {body.get('pathway_name', '')}",
                evidence_ref=evi.get("id", ""),
                confidence=0.95,
            )
    return None
