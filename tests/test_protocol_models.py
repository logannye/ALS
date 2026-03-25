"""Tests for ontology.intervention, ontology.protocol — Intervention, CureProtocolCandidate, MonitoringPlan."""
import pytest

from ontology.enums import ApprovalState, InterventionClass, ProtocolLayer
from ontology.intervention import Intervention
from ontology.protocol import CureProtocolCandidate, MonitoringPlan, ProtocolLayerEntry


# ---------------------------------------------------------------------------
# Intervention
# ---------------------------------------------------------------------------

class TestIntervention:
    def _make(self) -> Intervention:
        return Intervention(
            id="intervention:tdp43_aso_v1",
            name="TDP-43 ASO (ION363 analog)",
            intervention_class=InterventionClass.aso,
            targets=["TARDBP"],
            protocol_layer=ProtocolLayer.root_cause_suppression,
            route="intrathecal",
            intended_effects=["TDP-43_nuclear_restoration", "splicing_correction"],
            known_risks=["injection_site_reaction", "off_target_splicing"],
            contraindications=["active_infection", "coagulopathy"],
        )

    def test_type_is_intervention(self):
        i = self._make()
        assert i.type == "Intervention"

    def test_name(self):
        i = self._make()
        assert "ASO" in i.name

    def test_intervention_class(self):
        i = self._make()
        assert i.intervention_class == InterventionClass.aso

    def test_targets(self):
        i = self._make()
        assert "TARDBP" in i.targets

    def test_protocol_layer(self):
        i = self._make()
        assert i.protocol_layer == ProtocolLayer.root_cause_suppression

    def test_route(self):
        i = self._make()
        assert i.route == "intrathecal"

    def test_intended_effects(self):
        i = self._make()
        assert "TDP-43_nuclear_restoration" in i.intended_effects

    def test_known_risks(self):
        i = self._make()
        assert "off_target_splicing" in i.known_risks

    def test_contraindications(self):
        i = self._make()
        assert "active_infection" in i.contraindications

    def test_optional_protocol_layer_defaults_none(self):
        i = Intervention(
            id="intervention:minimal",
            name="Riluzole",
            intervention_class=InterventionClass.drug,
            targets=["EAAT2"],
            route="oral",
            intended_effects=["glutamate_excitotoxicity_reduction"],
            known_risks=["hepatotoxicity"],
            contraindications=[],
        )
        assert i.protocol_layer is None


# ---------------------------------------------------------------------------
# ProtocolLayerEntry
# ---------------------------------------------------------------------------

class TestProtocolLayerEntry:
    def test_creation(self):
        entry = ProtocolLayerEntry(
            layer=ProtocolLayer.root_cause_suppression,
            intervention_refs=["intervention:tdp43_aso_v1"],
            start_offset_days=0,
            notes="Initiate at trial enrollment",
        )
        assert entry.layer == ProtocolLayer.root_cause_suppression
        assert len(entry.intervention_refs) == 1
        assert entry.start_offset_days == 0

    def test_start_offset_defaults_to_zero(self):
        entry = ProtocolLayerEntry(
            layer=ProtocolLayer.adaptive_maintenance,
            intervention_refs=[],
            notes="",
        )
        assert entry.start_offset_days == 0


# ---------------------------------------------------------------------------
# CureProtocolCandidate
# ---------------------------------------------------------------------------

class TestCureProtocolCandidate:
    def _make(self) -> CureProtocolCandidate:
        layer1 = ProtocolLayerEntry(
            layer=ProtocolLayer.root_cause_suppression,
            intervention_refs=["intervention:tdp43_aso_v1"],
            start_offset_days=0,
            notes="Primary ASO intervention",
        )
        layer2 = ProtocolLayerEntry(
            layer=ProtocolLayer.circuit_stabilization,
            intervention_refs=["intervention:riluzole_v1"],
            start_offset_days=14,
            notes="Adjunct neuroprotection",
        )
        return CureProtocolCandidate(
            id="protocol:erik_draper_phase1_candidate",
            subject_ref="patient:erik_draper",
            objective="Restore TDP-43 nuclear function and stabilize motor circuitry",
            eligibility_constraints=["alsfrs_r_total >= 38", "fvc_percent >= 70"],
            contraindications=["active_infection"],
            assumed_active_programs=["program:als_registry_v1"],
            layers=[layer1, layer2],
            monitoring_plan_ref="monitoring:erik_draper_v1",
            expected_state_shift_summary={
                "tdp43_nuclear_restoration": 0.65,
                "splicing_correction": 0.55,
            },
            dominant_failure_modes=["insufficient_aso_penetration", "nmj_irreversibility"],
            approval_state=ApprovalState.pending,
            required_approval_refs=["approval:irb_als_phase1"],
            evidence_bundle_refs=["bundle:subtype_evidence_v1"],
            uncertainty_ref="uncertainty:erik_draper_v1",
        )

    def test_type_is_cure_protocol_candidate(self):
        p = self._make()
        assert p.type == "CureProtocolCandidate"

    def test_subject_ref(self):
        p = self._make()
        assert p.subject_ref == "patient:erik_draper"

    def test_objective(self):
        p = self._make()
        assert "TDP-43" in p.objective

    def test_eligibility_constraints(self):
        p = self._make()
        assert "alsfrs_r_total >= 38" in p.eligibility_constraints

    def test_two_layers(self):
        p = self._make()
        assert len(p.layers) == 2

    def test_first_layer_type(self):
        p = self._make()
        assert p.layers[0].layer == ProtocolLayer.root_cause_suppression

    def test_second_layer_start_offset(self):
        p = self._make()
        assert p.layers[1].start_offset_days == 14

    def test_approval_state_is_pending(self):
        p = self._make()
        assert p.approval_state == ApprovalState.pending

    def test_approval_state_default_is_pending(self):
        layer = ProtocolLayerEntry(
            layer=ProtocolLayer.adaptive_maintenance,
            intervention_refs=[],
            notes="",
        )
        p = CureProtocolCandidate(
            id="protocol:minimal",
            subject_ref="patient:x",
            objective="Minimal protocol",
            eligibility_constraints=[],
            contraindications=[],
            assumed_active_programs=[],
            layers=[layer],
            monitoring_plan_ref=None,
            expected_state_shift_summary={},
            dominant_failure_modes=[],
            required_approval_refs=[],
            evidence_bundle_refs=[],
            uncertainty_ref=None,
        )
        assert p.approval_state == ApprovalState.pending

    def test_expected_state_shift_summary(self):
        p = self._make()
        assert p.expected_state_shift_summary["tdp43_nuclear_restoration"] == pytest.approx(0.65)

    def test_dominant_failure_modes(self):
        p = self._make()
        assert "nmj_irreversibility" in p.dominant_failure_modes

    def test_evidence_bundle_refs(self):
        p = self._make()
        assert "bundle:subtype_evidence_v1" in p.evidence_bundle_refs


# ---------------------------------------------------------------------------
# MonitoringPlan
# ---------------------------------------------------------------------------

class TestMonitoringPlan:
    def _make(self) -> MonitoringPlan:
        return MonitoringPlan(
            id="monitoring:erik_draper_v1",
            subject_ref="patient:erik_draper",
            scheduled_checks=[
                {"at_day": 30, "measurement": "ALSFRS-R", "method": "telehealth"},
                {"at_day": 30, "measurement": "NfL plasma", "method": "blood draw"},
                {"at_day": 90, "measurement": "EMG", "method": "in_person"},
            ],
            success_criteria=[
                "ALSFRS-R decline < 2 points at 3 months",
                "NfL reduction > 20% at 3 months",
            ],
            failure_triggers=[
                "ALSFRS-R decline > 5 points",
                "FVC < 60%",
                "Grade 3+ adverse event",
            ],
        )

    def test_type_is_monitoring_plan(self):
        m = self._make()
        assert m.type == "MonitoringPlan"

    def test_subject_ref(self):
        m = self._make()
        assert m.subject_ref == "patient:erik_draper"

    def test_scheduled_checks_count(self):
        m = self._make()
        assert len(m.scheduled_checks) == 3

    def test_scheduled_checks_content(self):
        m = self._make()
        check = m.scheduled_checks[0]
        assert check["measurement"] == "ALSFRS-R"
        assert check["at_day"] == 30

    def test_success_criteria(self):
        m = self._make()
        assert len(m.success_criteria) == 2
        assert any("NfL" in c for c in m.success_criteria)

    def test_failure_triggers(self):
        m = self._make()
        assert len(m.failure_triggers) == 3
        assert any("FVC" in t for t in m.failure_triggers)

    def test_minimal_creation(self):
        m = MonitoringPlan(
            id="monitoring:minimal",
            subject_ref="patient:x",
            scheduled_checks=[],
            success_criteria=[],
            failure_triggers=[],
        )
        assert m.type == "MonitoringPlan"
