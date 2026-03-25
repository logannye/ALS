"""Tests for EvidenceStrength enum and extended InterventionClass."""
import pytest

from ontology.enums import EvidenceStrength, InterventionClass
from ontology.evidence import EvidenceItem
from ontology.enums import EvidenceDirection


class TestEvidenceStrengthEnum:
    def test_has_five_values(self):
        assert len(EvidenceStrength) == 5

    def test_strong_value(self):
        assert EvidenceStrength.strong == "strong"

    def test_moderate_value(self):
        assert EvidenceStrength.moderate == "moderate"

    def test_emerging_value(self):
        assert EvidenceStrength.emerging == "emerging"

    def test_preclinical_value(self):
        assert EvidenceStrength.preclinical == "preclinical"

    def test_unknown_value(self):
        assert EvidenceStrength.unknown == "unknown"

    def test_is_str_enum(self):
        assert isinstance(EvidenceStrength.strong, str)


class TestInterventionClassExtended:
    def test_gene_therapy_member(self):
        assert InterventionClass.gene_therapy == "gene_therapy"

    def test_cell_therapy_member(self):
        assert InterventionClass.cell_therapy == "cell_therapy"

    def test_peptide_member(self):
        assert InterventionClass.peptide == "peptide"


class TestEvidenceItemWithStrengthEnum:
    def test_evidence_item_accepts_evidence_strength(self):
        e = EvidenceItem(
            id="evidence_item:test_enum_strength",
            claim="Test claim",
            direction=EvidenceDirection.supports,
            source_refs=[],
            strength=EvidenceStrength.strong,
        )
        assert e.strength == EvidenceStrength.strong

    def test_strength_defaults_to_unknown(self):
        e = EvidenceItem(
            id="evidence_item:test_default_strength",
            claim="Test claim",
            direction=EvidenceDirection.supports,
            source_refs=[],
        )
        assert e.strength == EvidenceStrength.unknown

    def test_strength_moderate(self):
        e = EvidenceItem(
            id="evidence_item:test_moderate",
            claim="Test claim",
            direction=EvidenceDirection.mixed,
            source_refs=[],
            strength=EvidenceStrength.moderate,
        )
        assert e.strength == EvidenceStrength.moderate

    def test_strength_emerging(self):
        e = EvidenceItem(
            id="evidence_item:test_emerging",
            claim="Test claim",
            direction=EvidenceDirection.supports,
            source_refs=[],
            strength=EvidenceStrength.emerging,
        )
        assert e.strength == EvidenceStrength.emerging


class TestEvidenceItemSupersedesRef:
    def test_supersedes_ref_defaults_to_none(self):
        e = EvidenceItem(
            id="evidence_item:test_supersedes_default",
            claim="Test claim",
            direction=EvidenceDirection.supports,
            source_refs=[],
        )
        assert e.supersedes_ref is None

    def test_supersedes_ref_can_be_set(self):
        e = EvidenceItem(
            id="evidence_item:test_supersedes_set",
            claim="Test claim",
            direction=EvidenceDirection.supports,
            source_refs=[],
            supersedes_ref="evidence_item:old_version",
        )
        assert e.supersedes_ref == "evidence_item:old_version"

    def test_supersedes_ref_is_optional_string(self):
        e = EvidenceItem(
            id="evidence_item:test_supersedes_none",
            claim="Test claim",
            direction=EvidenceDirection.refutes,
            source_refs=["obs:x"],
            supersedes_ref=None,
        )
        assert e.supersedes_ref is None
