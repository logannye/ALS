"""Tests for ontology.enums — canonical enum types for the Erik ALS engine."""
import pytest
from ontology.enums import (
    ObjectStatus,
    ConfidenceBand,
    PrivacyClass,
    ApprovalState,
    EvidenceDirection,
    ActionClass,
    ALSOnsetRegion,
    SubtypeClass,
    ProtocolLayer,
    PCHLayer,
    ObservationKind,
    InterpretationKind,
    InterventionClass,
    RelationCategory,
    SourceSystem,
)


class TestObjectStatus:
    def test_values_exist(self):
        assert ObjectStatus.active.value == "active"
        assert ObjectStatus.superseded.value == "superseded"
        assert ObjectStatus.deprecated.value == "deprecated"
        assert ObjectStatus.deleted_logically.value == "deleted_logically"

    def test_all_four_members(self):
        assert len(ObjectStatus) == 4


class TestSubtypeClass:
    def test_all_eight_values(self):
        expected = {"sod1", "c9orf72", "fus", "tardbp", "sporadic_tdp43", "glia_amplified", "mixed", "unresolved"}
        actual = {m.value for m in SubtypeClass}
        assert actual == expected

    def test_count(self):
        assert len(SubtypeClass) == 8


class TestProtocolLayer:
    def test_five_layers_exist(self):
        assert len(ProtocolLayer) == 5

    def test_correct_ordering(self):
        layers = list(ProtocolLayer)
        assert layers[0].value == "root_cause_suppression"
        assert layers[1].value == "pathology_reversal"
        assert layers[2].value == "circuit_stabilization"
        assert layers[3].value == "regeneration_reinnervation"
        assert layers[4].value == "adaptive_maintenance"


class TestPCHLayer:
    def test_integer_values(self):
        assert PCHLayer.L1_ASSOCIATIONAL == 1
        assert PCHLayer.L2_INTERVENTIONAL == 2
        assert PCHLayer.L3_COUNTERFACTUAL == 3

    def test_is_int_comparable(self):
        assert PCHLayer.L1_ASSOCIATIONAL < PCHLayer.L2_INTERVENTIONAL
        assert PCHLayer.L2_INTERVENTIONAL < PCHLayer.L3_COUNTERFACTUAL


class TestALSOnsetRegion:
    def test_all_six_options(self):
        expected = {"upper_limb", "lower_limb", "bulbar", "respiratory", "multifocal", "unknown"}
        actual = {m.value for m in ALSOnsetRegion}
        assert actual == expected

    def test_count(self):
        assert len(ALSOnsetRegion) == 6


class TestObservationKind:
    def test_includes_clinical_types(self):
        values = {m.value for m in ObservationKind}
        required = {
            "lab_result",
            "emg_feature",
            "respiratory_metric",
            "genomic_result",
            "imaging_finding",
            "functional_score",
        }
        assert required.issubset(values), f"Missing: {required - values}"

    def test_all_thirteen_members(self):
        assert len(ObservationKind) == 13


class TestConfidenceBand:
    def test_five_bands(self):
        assert len(ConfidenceBand) == 5

    def test_values(self):
        assert ConfidenceBand.very_low.value == "very_low"
        assert ConfidenceBand.high.value == "high"
        assert ConfidenceBand.very_high.value == "very_high"


class TestPrivacyClass:
    def test_four_classes(self):
        assert len(PrivacyClass) == 4

    def test_values(self):
        assert PrivacyClass.public.value == "public"
        assert PrivacyClass.phi.value == "phi"


class TestApprovalState:
    def test_five_states(self):
        assert len(ApprovalState) == 5


class TestEvidenceDirection:
    def test_four_directions(self):
        assert len(EvidenceDirection) == 4

    def test_values(self):
        assert EvidenceDirection.supports.value == "supports"
        assert EvidenceDirection.refutes.value == "refutes"
        assert EvidenceDirection.mixed.value == "mixed"
        assert EvidenceDirection.insufficient.value == "insufficient"


class TestActionClass:
    def test_seven_classes(self):
        assert len(ActionClass) == 7


class TestInterpretationKind:
    def test_eight_kinds(self):
        assert len(InterpretationKind) == 8


class TestInterventionClass:
    def test_twelve_classes(self):
        assert len(InterventionClass) == 12

    def test_includes_als_relevant_types(self):
        values = {m.value for m in InterventionClass}
        assert "aso" in values
        assert "gene_editing" in values
        assert "gene_silencing" in values


class TestRelationCategory:
    def test_six_categories(self):
        assert len(RelationCategory) == 6


class TestSourceSystem:
    def test_ten_systems(self):
        assert len(SourceSystem) == 10

    def test_includes_core_systems(self):
        values = {m.value for m in SourceSystem}
        for expected in {"ehr", "registry", "lims", "omics", "trial", "manual", "model", "workflow", "literature", "database"}:
            assert expected in values, f"Missing source system: {expected}"
