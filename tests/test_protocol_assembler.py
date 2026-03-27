"""Tests for Stage 4 — Protocol assembly from scored interventions."""
from __future__ import annotations

import pytest

from ontology.enums import ApprovalState, ProtocolLayer
from world_model.intervention_scorer import InterventionScore
from world_model.protocol_assembler import (
    assemble_protocol,
    select_layer_interventions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score(
    int_id: str,
    name: str,
    layer: str,
    relevance: float,
    eligible=True,
    cited: list[str] | None = None,
) -> InterventionScore:
    return InterventionScore(
        intervention_id=int_id,
        intervention_name=name,
        protocol_layer=layer,
        relevance_score=relevance,
        erik_eligible=eligible,
        key_uncertainties=[],
        cited_evidence=cited or [f"evi:{int_id}"],
        contested_claims=[],
    )


# ---------------------------------------------------------------------------
# select_layer_interventions
# ---------------------------------------------------------------------------

class TestSelectLayerInterventions:

    def test_selects_top_for_layer(self):
        scores = [
            _score("int:a", "A", "root_cause_suppression", 0.9),
            _score("int:b", "B", "root_cause_suppression", 0.5),
            _score("int:c", "C", "pathology_reversal", 0.8),
        ]
        selected = select_layer_interventions(scores, "root_cause_suppression")
        assert len(selected) >= 1
        assert selected[0].intervention_id == "int:a"

    def test_skips_ineligible(self):
        scores = [
            _score("int:a", "A", "root_cause_suppression", 0.9, eligible=False),
            _score("int:b", "B", "root_cause_suppression", 0.5, eligible=True),
        ]
        selected = select_layer_interventions(scores, "root_cause_suppression")
        assert selected[0].intervention_id == "int:b"

    def test_allows_pending_genetics(self):
        scores = [
            _score("int:a", "A", "root_cause_suppression", 0.9, eligible="pending_genetics"),
        ]
        selected = select_layer_interventions(scores, "root_cause_suppression")
        assert len(selected) == 1

    def test_returns_empty_for_missing_layer(self):
        scores = [_score("int:a", "A", "circuit_stabilization", 0.7)]
        selected = select_layer_interventions(scores, "root_cause_suppression")
        assert selected == []

    def test_max_per_layer_limit(self):
        scores = [
            _score(f"int:{i}", f"Drug{i}", "root_cause_suppression", 0.9 - i * 0.1)
            for i in range(5)
        ]
        selected = select_layer_interventions(scores, "root_cause_suppression", max_per_layer=2)
        assert len(selected) == 2


# ---------------------------------------------------------------------------
# assemble_protocol
# ---------------------------------------------------------------------------

class TestAssembleProtocol:

    def test_has_5_layers(self):
        scores = [
            _score(f"int:{layer}", layer, layer, 0.7)
            for layer in [
                "root_cause_suppression",
                "pathology_reversal",
                "circuit_stabilization",
                "regeneration_reinnervation",
                "adaptive_maintenance",
            ]
        ]
        protocol = assemble_protocol(scores, "traj:draper_001")
        assert protocol.type == "CureProtocolCandidate"
        assert len(protocol.layers) == 5

    def test_approval_state_pending(self):
        scores = [_score("int:a", "A", "root_cause_suppression", 0.7)]
        protocol = assemble_protocol(scores, "traj:draper_001")
        assert protocol.approval_state == ApprovalState.pending

    def test_abstains_empty_layer(self):
        """Layers with no scored interventions get ABSTENTION notes."""
        scores = [
            _score("int:a", "A", "circuit_stabilization", 0.7),
        ]
        protocol = assemble_protocol(scores, "traj:draper_001")
        abstained = [l for l in protocol.layers if "ABSTENTION" in l.notes]
        assert len(abstained) >= 1

    def test_evidence_bundle_refs_collected(self):
        scores = [
            _score("int:a", "A", "root_cause_suppression", 0.9, cited=["evi:x", "evi:y"]),
            _score("int:b", "B", "pathology_reversal", 0.8, cited=["evi:y", "evi:z"]),
        ]
        protocol = assemble_protocol(scores, "traj:draper_001")
        assert "evi:x" in protocol.evidence_bundle_refs
        assert "evi:z" in protocol.evidence_bundle_refs

    def test_protocol_id_format(self):
        scores = [_score("int:a", "A", "root_cause_suppression", 0.7)]
        protocol = assemble_protocol(scores, "traj:draper_001")
        assert protocol.id.startswith("proto:")

    def test_body_contains_scores(self):
        scores = [_score("int:a", "A", "root_cause_suppression", 0.7)]
        protocol = assemble_protocol(scores, "traj:draper_001")
        assert "all_intervention_scores" in protocol.body
        assert "total_evidence_items_cited" in protocol.body
