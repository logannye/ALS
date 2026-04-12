"""TCG data models: nodes, edges, hypotheses, and acquisition queue items."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TCGNode:
    """A biological entity in the ALS mechanistic model."""

    id: str
    entity_type: str
    name: str
    pathway_cluster: Optional[str] = None
    description: Optional[str] = None
    druggability_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "name": self.name,
            "pathway_cluster": self.pathway_cluster,
            "description": self.description,
            "druggability_score": self.druggability_score,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TCGNode:
        d = dict(d)
        for k in ("created_at", "updated_at"):
            if isinstance(d.get(k), str):
                d[k] = datetime.fromisoformat(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TCGEdge:
    """A directed mechanistic link between two TCG nodes."""

    id: str
    source_id: str
    target_id: str
    edge_type: str
    confidence: float = 0.1
    evidence_ids: list[str] = field(default_factory=list)
    contradiction_ids: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    intervention_potential: dict[str, Any] = field(default_factory=dict)
    last_reasoned_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def therapeutic_priority(self) -> float:
        """Higher = more important to investigate. Relevance * uncertainty."""
        relevance = self.intervention_potential.get("therapeutic_relevance", 0.5)
        return relevance * (1.0 - self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
            "contradiction_ids": self.contradiction_ids,
            "open_questions": self.open_questions,
            "intervention_potential": self.intervention_potential,
            "last_reasoned_at": self.last_reasoned_at.isoformat() if self.last_reasoned_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TCGEdge:
        d = dict(d)
        for k in ("last_reasoned_at", "created_at", "updated_at"):
            if isinstance(d.get(k), str):
                d[k] = datetime.fromisoformat(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TCGHypothesis:
    """A therapeutic hypothesis with causal justification through the TCG."""

    id: str
    hypothesis: str
    supporting_path: list[str] = field(default_factory=list)
    confidence: float = 0.1
    status: str = "proposed"
    generated_by: Optional[str] = None
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    therapeutic_relevance: float = 0.0
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "supporting_path": self.supporting_path,
            "confidence": self.confidence,
            "status": self.status,
            "generated_by": self.generated_by,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "open_questions": self.open_questions,
            "therapeutic_relevance": self.therapeutic_relevance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TCGHypothesis:
        d = dict(d)
        for k in ("created_at", "updated_at"):
            if isinstance(d.get(k), str):
                d[k] = datetime.fromisoformat(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AcquisitionItem:
    """A targeted evidence request driven by a TCG open question."""

    tcg_edge_id: str
    open_question: str
    suggested_sources: list[str] = field(default_factory=list)
    exhausted_sources: list[str] = field(default_factory=list)
    priority: float = 0.0
    status: str = "pending"
    created_by: Optional[str] = None
    id: Optional[int] = None
    created_at: datetime = field(default_factory=_now)
    answered_at: Optional[datetime] = None
