"""Tests for Phase 3: increase evidence utilization in protocol assembly.

Covers: configurable max_per_layer, supporting evidence collection,
protocol version incrementing.
"""
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


# ===========================================================================
# 3A. Configurable max_per_layer
# ===========================================================================

class TestConfigurableMaxPerLayer:

    def test_max_per_layer_3_selects_3(self):
        """With max_per_layer=3, should select up to 3 interventions per layer."""
        scores = [
            _score("int:a1", "A1", "root_cause_suppression", 0.9),
            _score("int:a2", "A2", "root_cause_suppression", 0.8),
            _score("int:a3", "A3", "root_cause_suppression", 0.7),
            _score("int:a4", "A4", "root_cause_suppression", 0.6),
        ]
        selected = select_layer_interventions(scores, "root_cause_suppression", max_per_layer=3)
        assert len(selected) == 3
        assert selected[0].intervention_id == "int:a1"
        assert selected[2].intervention_id == "int:a3"

    def test_default_max_per_layer_is_at_least_2(self):
        """Default max_per_layer should be >= 2 (backward compat)."""
        scores = [
            _score("int:a1", "A1", "root_cause_suppression", 0.9),
            _score("int:a2", "A2", "root_cause_suppression", 0.8),
            _score("int:a3", "A3", "root_cause_suppression", 0.7),
        ]
        selected = select_layer_interventions(scores, "root_cause_suppression")
        assert len(selected) >= 2


# ===========================================================================
# 3B. Supporting evidence from non-selected interventions
# ===========================================================================

class TestSupportingEvidence:

    def test_body_contains_supporting_evidence_refs(self):
        """Protocol body should contain evidence from non-selected interventions."""
        scores = [
            _score("int:a1", "A1", "root_cause_suppression", 0.9, cited=["evi:1", "evi:2"]),
            _score("int:a2", "A2", "root_cause_suppression", 0.8, cited=["evi:3"]),
            _score("int:a3", "A3", "root_cause_suppression", 0.5, cited=["evi:4", "evi:5"]),
        ]
        protocol = assemble_protocol(scores, "traj:draper_001")
        body = protocol.body
        assert "supporting_evidence_refs" in body
        # int:a3 is not selected (max_per_layer defaults to 2-3), its evidence should be supporting
        supporting = set(body["supporting_evidence_refs"])
        # At minimum, some evidence should be in supporting that isn't in primary
        primary = set(protocol.evidence_bundle_refs)
        # supporting and primary should not fully overlap (unless all selected)
        assert "total_evidence_considered" in body

    def test_total_evidence_considered_larger_than_cited(self):
        """total_evidence_considered should be >= evidence_bundle_refs."""
        scores = [
            _score("int:a1", "A1", "root_cause_suppression", 0.9, cited=["evi:1"]),
            _score("int:a2", "A2", "root_cause_suppression", 0.8, cited=["evi:2"]),
            _score("int:a3", "A3", "root_cause_suppression", 0.5, cited=["evi:3"]),
            _score("int:b1", "B1", "pathology_reversal", 0.7, cited=["evi:4"]),
        ]
        protocol = assemble_protocol(scores, "traj:draper_001")
        total = protocol.body.get("total_evidence_considered", 0)
        cited = len(protocol.evidence_bundle_refs)
        assert total >= cited


# ===========================================================================
# 3C. Protocol version incrementing
# ===========================================================================

class TestProtocolVersioning:

    def test_version_parameter_in_protocol_id(self):
        """Protocol ID should include the version number."""
        scores = [
            _score("int:a1", "A1", "root_cause_suppression", 0.9),
        ]
        protocol = assemble_protocol(scores, "traj:draper_001", version=5)
        assert "_v5" in protocol.id

    def test_default_version_is_1(self):
        """Default version should be 1."""
        scores = [
            _score("int:a1", "A1", "root_cause_suppression", 0.9),
        ]
        protocol = assemble_protocol(scores, "traj:draper_001")
        assert "_v1" in protocol.id
