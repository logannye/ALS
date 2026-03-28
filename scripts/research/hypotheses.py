"""Hypothesis generation, lifecycle management, and validation action planning."""
from __future__ import annotations
import hashlib
from enum import Enum
from typing import Optional
from ontology.discovery import MechanismHypothesis
from ontology.enums import EvidenceDirection

class HypothesisStatus(str, Enum):
    GENERATED = "generated"
    SEARCHING = "searching"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INSUFFICIENT = "insufficient"

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "root_cause_suppression": ["ALS", "TDP-43", "C9orf72", "SOD1", "FUS", "loss-of-function", "gene therapy"],
    "pathology_reversal": ["ALS", "aggregation", "proteostasis", "sigma-1R", "autophagy", "clearance"],
    "circuit_stabilization": ["ALS", "neuroprotection", "glutamate", "excitotoxicity", "riluzole"],
    "regeneration_reinnervation": ["ALS", "motor neuron", "NMJ", "BDNF", "GDNF", "regeneration"],
    "adaptive_maintenance": ["ALS", "biomarker", "neurofilament", "ALSFRS-R", "monitoring"],
}

def create_hypothesis(
    statement: str, subject_ref: str, topic: str, cited_evidence: Optional[list[str]] = None,
) -> MechanismHypothesis:
    stmt_hash = hashlib.sha256(statement.encode()).hexdigest()[:12]
    hyp_id = f"hyp:{stmt_hash}"
    return MechanismHypothesis(
        id=hyp_id, statement=statement, subject_scope=subject_ref,
        predicted_observables=[], candidate_tests=[],
        current_support_direction=EvidenceDirection.insufficient,
        body={"topic": topic, "status": HypothesisStatus.GENERATED.value,
              "evidence_for": cited_evidence or [], "evidence_against": []},
    )

def is_duplicate_hypothesis(
    statement: str,
    existing: list[str],
    threshold: float = 0.6,
) -> bool:
    """Check if *statement* is a near-duplicate of any hypothesis in *existing*.

    Uses Jaccard similarity on keyword sets (words longer than 3 chars).
    Returns True if similarity exceeds *threshold*.
    """
    new_words = {w.lower() for w in statement.split() if len(w) > 3}
    if not new_words:
        return False
    for existing_stmt in existing:
        existing_words = {w.lower() for w in existing_stmt.split() if len(w) > 3}
        if not existing_words:
            continue
        intersection = len(new_words & existing_words)
        union = len(new_words | existing_words)
        if union > 0 and intersection / union > threshold:
            return True
    return False


def plan_validation_actions(statement: str, topic: str) -> list[dict]:
    actions: list[dict] = []
    keywords = _TOPIC_KEYWORDS.get(topic, ["ALS"])
    terms = [w for w in statement.split() if len(w) > 4][:3]
    query = " ".join(keywords[:3] + terms)
    actions.append({"action": "search_pubmed", "query": query, "max_results": 15})

    if topic in ("root_cause_suppression", "pathology_reversal"):
        for kw in ["TDP-43", "sigma-1R", "SOD1", "FUS", "mTOR"]:
            if kw.lower() in statement.lower():
                actions.append({"action": "query_chembl", "target_name": kw})
                break

    if "eligible" in statement.lower() or "trial" in statement.lower():
        actions.append({"action": "search_trials", "max_results": 20})

    return actions

def resolve_hypothesis(
    hypothesis: MechanismHypothesis, evidence_for: list[str], evidence_against: list[str],
) -> MechanismHypothesis:
    n_for = len(evidence_for)
    n_against = len(evidence_against)

    if n_for >= 2 and n_against == 0:
        direction = EvidenceDirection.supports
    elif n_against >= 2 and n_for == 0:
        direction = EvidenceDirection.refutes
    elif n_for > 0 and n_against > 0:
        direction = EvidenceDirection.mixed
    else:
        direction = EvidenceDirection.insufficient

    hypothesis.current_support_direction = direction
    hypothesis.body = dict(hypothesis.body)
    hypothesis.body["evidence_for"] = evidence_for
    hypothesis.body["evidence_against"] = evidence_against
    hypothesis.body["status"] = (
        HypothesisStatus.SUPPORTED.value if direction == EvidenceDirection.supports
        else HypothesisStatus.REFUTED.value if direction == EvidenceDirection.refutes
        else HypothesisStatus.INSUFFICIENT.value
    )
    return hypothesis
