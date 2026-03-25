"""Tests for ontology.interpretation — Interpretation and EtiologicDriverProfile."""
import pytest

from ontology.enums import InterpretationKind, SubtypeClass
from ontology.interpretation import EtiologicDriverProfile, Interpretation


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

class TestInterpretation:
    def _make_interpretation(self) -> Interpretation:
        return Interpretation(
            id="interpretation:subtype_v1",
            subject_ref="patient:erik_draper",
            interpretation_kind=InterpretationKind.subtype_inference,
            value="Sporadic TDP-43 proteinopathy — high confidence",
            supporting_observation_refs=[
                "observation:tdp43_igg_2025_05",
                "observation:nfl_2025_05_10",
            ],
            evidence_bundle_ref="bundle:subtype_evidence_v1",
            notes="Consistent with sporadic TDP-43 ALS; no pathogenic mutations found.",
        )

    def test_type_is_interpretation(self):
        i = self._make_interpretation()
        assert i.type == "Interpretation"

    def test_subject_ref(self):
        i = self._make_interpretation()
        assert i.subject_ref == "patient:erik_draper"

    def test_interpretation_kind(self):
        i = self._make_interpretation()
        assert i.interpretation_kind == InterpretationKind.subtype_inference

    def test_value_stored(self):
        i = self._make_interpretation()
        assert "Sporadic TDP-43" in i.value

    def test_supporting_observation_refs(self):
        i = self._make_interpretation()
        assert len(i.supporting_observation_refs) == 2
        assert "observation:nfl_2025_05_10" in i.supporting_observation_refs

    def test_evidence_bundle_ref(self):
        i = self._make_interpretation()
        assert i.evidence_bundle_ref == "bundle:subtype_evidence_v1"

    def test_supersedes_ref_defaults_none(self):
        i = self._make_interpretation()
        assert i.supersedes_ref is None

    def test_supersedes_ref_can_be_set(self):
        i = Interpretation(
            id="interpretation:subtype_v2",
            subject_ref="patient:erik_draper",
            interpretation_kind=InterpretationKind.subtype_inference,
            value="Updated interpretation",
            supporting_observation_refs=[],
            notes="",
            supersedes_ref="interpretation:subtype_v1",
        )
        assert i.supersedes_ref == "interpretation:subtype_v1"

    def test_notes_stored(self):
        i = self._make_interpretation()
        assert "TDP-43" in i.notes

    def test_minimal_creation(self):
        i = Interpretation(
            id="interpretation:minimal",
            subject_ref="patient:x",
            interpretation_kind=InterpretationKind.diagnosis,
            value="ALS confirmed",
            supporting_observation_refs=[],
            notes="",
        )
        assert i.type == "Interpretation"
        assert i.evidence_bundle_ref is None
        assert i.supersedes_ref is None


# ---------------------------------------------------------------------------
# EtiologicDriverProfile
# ---------------------------------------------------------------------------

class TestEtiologicDriverProfile:
    def _make_profile(self) -> EtiologicDriverProfile:
        return EtiologicDriverProfile(
            id="driver:erik_draper_v1",
            subject_ref="patient:erik_draper",
            posterior={
                SubtypeClass.sod1: 0.02,
                SubtypeClass.c9orf72: 0.03,
                SubtypeClass.fus: 0.01,
                SubtypeClass.tardbp: 0.05,
                SubtypeClass.sporadic_tdp43: 0.73,
                SubtypeClass.glia_amplified: 0.08,
                SubtypeClass.mixed: 0.04,
                SubtypeClass.unresolved: 0.04,
            },
            supporting_evidence_refs=[
                "observation:tdp43_igg_2025_05",
                "bundle:subtype_evidence_v1",
            ],
        )

    def test_type_is_etiologic_driver_profile(self):
        p = self._make_profile()
        assert p.type == "EtiologicDriverProfile"

    def test_subject_ref(self):
        p = self._make_profile()
        assert p.subject_ref == "patient:erik_draper"

    def test_posterior_has_8_subtypes(self):
        p = self._make_profile()
        assert len(p.posterior) == 8

    def test_dominant_subtype_is_sporadic_tdp43(self):
        p = self._make_profile()
        assert p.dominant_subtype == SubtypeClass.sporadic_tdp43

    def test_dominant_subtype_value_is_0_73(self):
        p = self._make_profile()
        assert p.posterior[SubtypeClass.sporadic_tdp43] == pytest.approx(0.73)

    def test_posterior_sums_to_1(self):
        p = self._make_profile()
        total = sum(p.posterior.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_supporting_evidence_refs(self):
        p = self._make_profile()
        assert "observation:tdp43_igg_2025_05" in p.supporting_evidence_refs

    def test_dominant_subtype_changes_with_different_posterior(self):
        p = EtiologicDriverProfile(
            id="driver:x",
            subject_ref="patient:x",
            posterior={
                SubtypeClass.sod1: 0.90,
                SubtypeClass.c9orf72: 0.10,
            },
            supporting_evidence_refs=[],
        )
        assert p.dominant_subtype == SubtypeClass.sod1
