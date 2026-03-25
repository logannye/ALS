"""Tests for ontology.state — latent state factor models."""
import pytest
from datetime import datetime, timezone
from typing import Optional

from ontology.state import (
    DiseaseStateSnapshot,
    FunctionalState,
    GlialState,
    NMJIntegrityState,
    RespiratoryReserveState,
    ReversibilityWindowEstimate,
    SplicingState,
    TDP43FunctionalState,
    UncertaintyState,
)


# ---------------------------------------------------------------------------
# TDP43FunctionalState
# ---------------------------------------------------------------------------

class TestTDP43FunctionalState:
    def _make(self) -> TDP43FunctionalState:
        return TDP43FunctionalState(
            id="tdp43:erik_draper_v1",
            subject_ref="patient:erik_draper",
            nuclear_function_score=0.55,
            cytoplasmic_pathology_probability=0.72,
            loss_of_function_probability=0.68,
            supporting_marker_refs=["observation:tdp43_igg_2025_05"],
            dominant_uncertainties=["nuclear_clearance_quantification"],
        )

    def test_type_is_tdp43_functional_state(self):
        s = self._make()
        assert s.type == "TDP43FunctionalState"

    def test_subject_ref(self):
        s = self._make()
        assert s.subject_ref == "patient:erik_draper"

    def test_nuclear_function_score(self):
        s = self._make()
        assert s.nuclear_function_score == pytest.approx(0.55)

    def test_cytoplasmic_pathology_probability(self):
        s = self._make()
        assert s.cytoplasmic_pathology_probability == pytest.approx(0.72)

    def test_loss_of_function_probability(self):
        s = self._make()
        assert s.loss_of_function_probability == pytest.approx(0.68)

    def test_supporting_marker_refs(self):
        s = self._make()
        assert "observation:tdp43_igg_2025_05" in s.supporting_marker_refs

    def test_dominant_uncertainties(self):
        s = self._make()
        assert "nuclear_clearance_quantification" in s.dominant_uncertainties

    def test_defaults_to_zero_scores(self):
        s = TDP43FunctionalState(
            id="tdp43:minimal",
            subject_ref="patient:x",
        )
        assert s.nuclear_function_score == 0
        assert s.cytoplasmic_pathology_probability == 0
        assert s.loss_of_function_probability == 0
        assert s.supporting_marker_refs == []
        assert s.dominant_uncertainties == []


# ---------------------------------------------------------------------------
# SplicingState
# ---------------------------------------------------------------------------

class TestSplicingState:
    def test_creation_and_type(self):
        s = SplicingState(
            id="splicing:erik_draper_v1",
            subject_ref="patient:erik_draper",
            cryptic_splicing_burden_score=0.65,
            stmn2_disruption_score=0.78,
            unc13a_disruption_score=0.60,
            other_target_scores={"PFKP": 0.30, "KALRN": 0.25},
            source_assay_refs=["assay:rna_seq_001"],
        )
        assert s.type == "SplicingState"
        assert s.stmn2_disruption_score == pytest.approx(0.78)
        assert "PFKP" in s.other_target_scores

    def test_defaults(self):
        s = SplicingState(
            id="splicing:minimal",
            subject_ref="patient:x",
            cryptic_splicing_burden_score=0.0,
            stmn2_disruption_score=0.0,
            unc13a_disruption_score=0.0,
        )
        assert s.other_target_scores == {}
        assert s.source_assay_refs == []


# ---------------------------------------------------------------------------
# GlialState
# ---------------------------------------------------------------------------

class TestGlialState:
    def test_creation_and_type(self):
        g = GlialState(
            id="glial:erik_draper_v1",
            subject_ref="patient:erik_draper",
            microglial_activation_score=0.55,
            astrocytic_toxicity_score=0.40,
            inflammatory_amplification_score=0.35,
            evidence_refs=["observation:csf_cytokines_v1"],
        )
        assert g.type == "GlialState"
        assert g.microglial_activation_score == pytest.approx(0.55)

    def test_defaults(self):
        g = GlialState(
            id="glial:minimal",
            subject_ref="patient:x",
            microglial_activation_score=0.0,
            astrocytic_toxicity_score=0.0,
            inflammatory_amplification_score=0.0,
        )
        assert g.evidence_refs == []


# ---------------------------------------------------------------------------
# NMJIntegrityState
# ---------------------------------------------------------------------------

class TestNMJIntegrityState:
    def test_creation_and_type(self):
        n = NMJIntegrityState(
            id="nmj:erik_draper_v1",
            subject_ref="patient:erik_draper",
            estimated_nmj_occupancy=0.72,
            denervation_rate_score=0.35,
            reinnervation_capacity_score=0.50,
            supporting_refs=["observation:emg_2025_04_20"],
        )
        assert n.type == "NMJIntegrityState"
        assert n.estimated_nmj_occupancy == pytest.approx(0.72)

    def test_defaults(self):
        n = NMJIntegrityState(
            id="nmj:minimal",
            subject_ref="patient:x",
            estimated_nmj_occupancy=0.0,
            denervation_rate_score=0.0,
            reinnervation_capacity_score=0.0,
        )
        assert n.supporting_refs == []


# ---------------------------------------------------------------------------
# RespiratoryReserveState
# ---------------------------------------------------------------------------

class TestRespiratoryReserveState:
    def test_creation_and_type(self):
        r = RespiratoryReserveState(
            id="resp_reserve:erik_draper_v1",
            subject_ref="patient:erik_draper",
            reserve_score=0.85,
            six_month_decline_risk=0.20,
            niv_transition_probability_6m=0.12,
            supporting_refs=["observation:resp_2025_05_15"],
        )
        assert r.type == "RespiratoryReserveState"
        assert r.reserve_score == pytest.approx(0.85)
        assert r.niv_transition_probability_6m == pytest.approx(0.12)

    def test_defaults(self):
        r = RespiratoryReserveState(
            id="resp_reserve:minimal",
            subject_ref="patient:x",
            reserve_score=0.0,
            six_month_decline_risk=0.0,
            niv_transition_probability_6m=0.0,
        )
        assert r.supporting_refs == []


# ---------------------------------------------------------------------------
# FunctionalState
# ---------------------------------------------------------------------------

class TestFunctionalState:
    def _make(self) -> FunctionalState:
        return FunctionalState(
            id="functional:erik_draper_v1",
            subject_ref="patient:erik_draper",
            alsfrs_r_total=43,
            bulbar_subscore=12,
            fine_motor_subscore=11,
            gross_motor_subscore=8,
            respiratory_subscore=12,
            speech_function_score=0.92,
            swallow_function_score=0.95,
            mobility_score=0.80,
            weight_kg=78.5,
        )

    def test_type_is_functional_state(self):
        s = self._make()
        assert s.type == "FunctionalState"

    def test_alsfrs_r_total(self):
        s = self._make()
        assert s.alsfrs_r_total == 43

    def test_bulbar_subscore(self):
        s = self._make()
        assert s.bulbar_subscore == pytest.approx(12)

    def test_fine_motor_subscore(self):
        s = self._make()
        assert s.fine_motor_subscore == pytest.approx(11)

    def test_gross_motor_subscore(self):
        s = self._make()
        assert s.gross_motor_subscore == pytest.approx(8)

    def test_respiratory_subscore(self):
        s = self._make()
        assert s.respiratory_subscore == pytest.approx(12)

    def test_weight_kg(self):
        s = self._make()
        assert s.weight_kg == pytest.approx(78.5)

    def test_optional_fields_default_none(self):
        s = FunctionalState(
            id="functional:minimal",
            subject_ref="patient:x",
        )
        assert s.alsfrs_r_total is None
        assert s.bulbar_subscore is None
        assert s.fine_motor_subscore is None
        assert s.weight_kg is None


# ---------------------------------------------------------------------------
# ReversibilityWindowEstimate
# ---------------------------------------------------------------------------

class TestReversibilityWindowEstimate:
    def _make(self) -> ReversibilityWindowEstimate:
        return ReversibilityWindowEstimate(
            id="reversibility:erik_draper_v1",
            subject_ref="patient:erik_draper",
            overall_reversibility_score=0.55,
            molecular_correction_plausibility=0.70,
            nmj_recovery_plausibility=0.50,
            functional_recovery_plausibility=0.40,
            dominant_bottleneck="nmj_denervation_rate",
            estimated_time_sensitivity_days=180,
        )

    def test_type_is_reversibility_window_estimate(self):
        r = self._make()
        assert r.type == "ReversibilityWindowEstimate"

    def test_overall_reversibility_score(self):
        r = self._make()
        assert r.overall_reversibility_score == pytest.approx(0.55)

    def test_dominant_bottleneck(self):
        r = self._make()
        assert r.dominant_bottleneck == "nmj_denervation_rate"

    def test_estimated_time_sensitivity_days(self):
        r = self._make()
        assert r.estimated_time_sensitivity_days == 180

    def test_optional_time_sensitivity_defaults_none(self):
        r = ReversibilityWindowEstimate(
            id="reversibility:minimal",
            subject_ref="patient:x",
            overall_reversibility_score=0.0,
            molecular_correction_plausibility=0.0,
            nmj_recovery_plausibility=0.0,
            functional_recovery_plausibility=0.0,
            dominant_bottleneck="unknown",
        )
        assert r.estimated_time_sensitivity_days is None


# ---------------------------------------------------------------------------
# UncertaintyState
# ---------------------------------------------------------------------------

class TestUncertaintyState:
    def _make(self) -> UncertaintyState:
        return UncertaintyState(
            id="uncertainty:erik_draper_v1",
            subject_ref="patient:erik_draper",
            subtype_ambiguity=0.15,
            missing_measurement_uncertainty=0.30,
            model_form_uncertainty=0.20,
            intervention_effect_uncertainty=0.25,
            transportability_uncertainty=0.10,
            evidence_conflict_uncertainty=0.05,
            dominant_missing_measurements=["tdp43_nuclear_clearance", "csf_neurofilament_heavy"],
        )

    def test_type_is_uncertainty_state(self):
        u = self._make()
        assert u.type == "UncertaintyState"

    def test_subtype_ambiguity(self):
        u = self._make()
        assert u.subtype_ambiguity == pytest.approx(0.15)

    def test_dominant_missing_measurements(self):
        u = self._make()
        assert "tdp43_nuclear_clearance" in u.dominant_missing_measurements
        assert len(u.dominant_missing_measurements) == 2

    def test_defaults_to_zero(self):
        u = UncertaintyState(
            id="uncertainty:minimal",
            subject_ref="patient:x",
        )
        assert u.subtype_ambiguity == 0
        assert u.missing_measurement_uncertainty == 0
        assert u.model_form_uncertainty == 0
        assert u.intervention_effect_uncertainty == 0
        assert u.transportability_uncertainty == 0
        assert u.evidence_conflict_uncertainty == 0
        assert u.dominant_missing_measurements == []


# ---------------------------------------------------------------------------
# DiseaseStateSnapshot
# ---------------------------------------------------------------------------

class TestDiseaseStateSnapshot:
    def _make(self) -> DiseaseStateSnapshot:
        return DiseaseStateSnapshot(
            id="snapshot:erik_draper_latest",
            subject_ref="patient:erik_draper",
            as_of=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            etiologic_driver_profile_ref="driver:erik_draper_v1",
            molecular_state_refs=[
                "tdp43:erik_draper_v1",
                "splicing:erik_draper_v1",
            ],
            compartment_state_refs=[
                "glial:erik_draper_v1",
                "nmj:erik_draper_v1",
            ],
            functional_state_ref="functional:erik_draper_v1",
            reversibility_window_ref="reversibility:erik_draper_v1",
            uncertainty_ref="uncertainty:erik_draper_v1",
        )

    def test_type_is_disease_state_snapshot(self):
        s = self._make()
        assert s.type == "DiseaseStateSnapshot"

    def test_subject_ref(self):
        s = self._make()
        assert s.subject_ref == "patient:erik_draper"

    def test_as_of_datetime(self):
        s = self._make()
        assert s.as_of == datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_etiologic_driver_profile_ref(self):
        s = self._make()
        assert s.etiologic_driver_profile_ref == "driver:erik_draper_v1"

    def test_molecular_state_refs(self):
        s = self._make()
        assert len(s.molecular_state_refs) == 2
        assert "tdp43:erik_draper_v1" in s.molecular_state_refs

    def test_compartment_state_refs(self):
        s = self._make()
        assert "nmj:erik_draper_v1" in s.compartment_state_refs

    def test_functional_state_ref(self):
        s = self._make()
        assert s.functional_state_ref == "functional:erik_draper_v1"

    def test_uncertainty_ref(self):
        s = self._make()
        assert s.uncertainty_ref == "uncertainty:erik_draper_v1"

    def test_all_refs_optional(self):
        s = DiseaseStateSnapshot(
            id="snapshot:minimal",
            subject_ref="patient:x",
            as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert s.type == "DiseaseStateSnapshot"
        assert s.etiologic_driver_profile_ref is None
        assert s.molecular_state_refs == []
        assert s.compartment_state_refs == []
        assert s.functional_state_ref is None
        assert s.reversibility_window_ref is None
        assert s.uncertainty_ref is None
