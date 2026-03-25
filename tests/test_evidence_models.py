"""Tests for ontology.evidence — EvidenceItem and EvidenceBundle."""
import pytest

from ontology.enums import EvidenceDirection, EvidenceStrength
from ontology.evidence import EvidenceBundle, EvidenceItem


# ---------------------------------------------------------------------------
# EvidenceItem
# ---------------------------------------------------------------------------

class TestEvidenceItem:
    def _make(self) -> EvidenceItem:
        return EvidenceItem(
            id="evidence_item:tdp43_nuclear_clearance_v1",
            claim="TDP-43 nuclear clearance correlates with ALS severity",
            direction=EvidenceDirection.supports,
            source_refs=["observation:tdp43_igg_2025_05", "literature:pmid_12345678"],
            strength=EvidenceStrength.strong,
            notes="Observed in patient cohort study n=142",
        )

    def test_type_is_evidence_item(self):
        e = self._make()
        assert e.type == "EvidenceItem"

    def test_direction_is_supports(self):
        e = self._make()
        assert e.direction == EvidenceDirection.supports

    def test_claim_stored(self):
        e = self._make()
        assert "TDP-43" in e.claim

    def test_source_refs(self):
        e = self._make()
        assert len(e.source_refs) == 2
        assert "literature:pmid_12345678" in e.source_refs

    def test_strength(self):
        e = self._make()
        assert e.strength == EvidenceStrength.strong

    def test_notes(self):
        e = self._make()
        assert "cohort" in e.notes

    def test_minimal_creation(self):
        e = EvidenceItem(
            id="evidence_item:minimal",
            claim="Minimal claim",
            direction=EvidenceDirection.insufficient,
            source_refs=[],
            strength=EvidenceStrength.unknown,
            notes="",
        )
        assert e.type == "EvidenceItem"
        assert e.direction == EvidenceDirection.insufficient

    def test_refutes_direction(self):
        e = EvidenceItem(
            id="evidence_item:refutes",
            claim="SOD1 mutation not found",
            direction=EvidenceDirection.refutes,
            source_refs=["observation:genomic_001"],
            strength=EvidenceStrength.strong,
            notes="",
        )
        assert e.direction == EvidenceDirection.refutes

    def test_mixed_direction(self):
        e = EvidenceItem(
            id="evidence_item:mixed",
            claim="Riluzole efficacy",
            direction=EvidenceDirection.mixed,
            source_refs=[],
            strength=EvidenceStrength.moderate,
            notes="",
        )
        assert e.direction == EvidenceDirection.mixed


# ---------------------------------------------------------------------------
# EvidenceBundle
# ---------------------------------------------------------------------------

class TestEvidenceBundle:
    def _make(self) -> EvidenceBundle:
        return EvidenceBundle(
            id="bundle:subtype_evidence_v1",
            subject_ref="patient:erik_draper",
            topic="sporadic_tdp43_subtype_classification",
            evidence_item_refs=[
                "evidence_item:tdp43_nuclear_clearance_v1",
                "evidence_item:nfl_elevation_v1",
                "evidence_item:no_pathogenic_mutation_v1",
            ],
            contradiction_refs=[],
            coverage_score=0.75,
            grounding_score=0.88,
        )

    def test_type_is_evidence_bundle(self):
        b = self._make()
        assert b.type == "EvidenceBundle"

    def test_subject_ref(self):
        b = self._make()
        assert b.subject_ref == "patient:erik_draper"

    def test_topic(self):
        b = self._make()
        assert b.topic == "sporadic_tdp43_subtype_classification"

    def test_evidence_item_refs(self):
        b = self._make()
        assert len(b.evidence_item_refs) == 3
        assert "evidence_item:nfl_elevation_v1" in b.evidence_item_refs

    def test_contradiction_refs_empty(self):
        b = self._make()
        assert b.contradiction_refs == []

    def test_coverage_score(self):
        b = self._make()
        assert b.coverage_score == pytest.approx(0.75)

    def test_grounding_score(self):
        b = self._make()
        assert b.grounding_score == pytest.approx(0.88)

    def test_with_contradictions(self):
        b = EvidenceBundle(
            id="bundle:conflict_v1",
            subject_ref="patient:x",
            topic="conflicting_evidence",
            evidence_item_refs=["evidence_item:a", "evidence_item:b"],
            contradiction_refs=["evidence_item:c"],
            coverage_score=0.5,
            grounding_score=0.6,
        )
        assert len(b.contradiction_refs) == 1
        assert "evidence_item:c" in b.contradiction_refs
