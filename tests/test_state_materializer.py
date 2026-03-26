"""Tests for world_model.state_materializer — Stage 1 observation materializer.

All tests use build_erik_draper() from ingestion.patient_builder (no DB needed).
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from ingestion.patient_builder import build_erik_draper
from ontology.state import (
    DiseaseStateSnapshot,
    FunctionalState,
    NMJIntegrityState,
    RespiratoryReserveState,
    UncertaintyState,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def erik():
    return build_erik_draper()


@pytest.fixture(scope="module")
def trajectory(erik):
    return erik[1]


@pytest.fixture(scope="module")
def observations(erik):
    return erik[2]


# ---------------------------------------------------------------------------
# materialize_functional_state
# ---------------------------------------------------------------------------

class TestMaterializeFunctionalState:
    def test_materialize_functional_state(self, trajectory, observations):
        from world_model.state_materializer import materialize_functional_state

        fs = materialize_functional_state(trajectory, observations)

        assert isinstance(fs, FunctionalState)
        assert fs.alsfrs_r_total == 43
        assert fs.bulbar_subscore == 12
        assert fs.fine_motor_subscore == 11
        assert fs.gross_motor_subscore == 8
        assert fs.respiratory_subscore == 12

    def test_materialize_functional_state_has_weight(self, trajectory, observations):
        from world_model.state_materializer import materialize_functional_state

        fs = materialize_functional_state(trajectory, observations)

        assert fs.weight_kg is not None
        assert fs.weight_kg > 100  # Erik's weight is ~111-118 kg

    def test_materialize_functional_state_id(self, trajectory, observations):
        from world_model.state_materializer import materialize_functional_state

        fs = materialize_functional_state(trajectory, observations)

        assert fs.id == f"func:{trajectory.id}"

    def test_materialize_functional_state_subject_ref(self, trajectory, observations):
        from world_model.state_materializer import materialize_functional_state

        fs = materialize_functional_state(trajectory, observations)

        assert fs.subject_ref == trajectory.patient_ref

    def test_materialize_functional_state_type(self, trajectory, observations):
        from world_model.state_materializer import materialize_functional_state

        fs = materialize_functional_state(trajectory, observations)

        assert fs.type == "FunctionalState"


# ---------------------------------------------------------------------------
# materialize_nmj_state
# ---------------------------------------------------------------------------

class TestMaterializeNMJState:
    def test_materialize_nmj_state_widespread(self, trajectory, observations):
        from world_model.state_materializer import materialize_nmj_state

        ns = materialize_nmj_state(trajectory, observations)

        assert isinstance(ns, NMJIntegrityState)
        # Erik has an EMG with supports_als=True → widespread denervation path
        assert ns.estimated_nmj_occupancy == pytest.approx(0.5)
        assert ns.denervation_rate_score == pytest.approx(0.7)
        assert ns.reinnervation_capacity_score == pytest.approx(0.4)

    def test_materialize_nmj_state_id(self, trajectory, observations):
        from world_model.state_materializer import materialize_nmj_state

        ns = materialize_nmj_state(trajectory, observations)

        assert ns.id == f"nmj:{trajectory.id}"

    def test_materialize_nmj_state_subject_ref(self, trajectory, observations):
        from world_model.state_materializer import materialize_nmj_state

        ns = materialize_nmj_state(trajectory, observations)

        assert ns.subject_ref == trajectory.patient_ref


# ---------------------------------------------------------------------------
# materialize_respiratory_state
# ---------------------------------------------------------------------------

class TestMaterializeRespiratoryState:
    def test_materialize_respiratory_state(self, trajectory, observations):
        from world_model.state_materializer import materialize_respiratory_state

        rs = materialize_respiratory_state(trajectory, observations)

        assert isinstance(rs, RespiratoryReserveState)
        # Erik's FVC is 100% predicted → reserve_score = 1.0, low risk
        assert rs.reserve_score == pytest.approx(1.0)
        assert rs.six_month_decline_risk == pytest.approx(0.2)
        assert rs.niv_transition_probability_6m == pytest.approx(0.1)

    def test_materialize_respiratory_state_id(self, trajectory, observations):
        from world_model.state_materializer import materialize_respiratory_state

        rs = materialize_respiratory_state(trajectory, observations)

        assert rs.id == f"resp:{trajectory.id}"

    def test_materialize_respiratory_state_subject_ref(self, trajectory, observations):
        from world_model.state_materializer import materialize_respiratory_state

        rs = materialize_respiratory_state(trajectory, observations)

        assert rs.subject_ref == trajectory.patient_ref


# ---------------------------------------------------------------------------
# materialize_uncertainty_state
# ---------------------------------------------------------------------------

class TestMaterializeUncertaintyState:
    def test_materialize_uncertainty_state(self, trajectory, observations):
        from world_model.state_materializer import materialize_uncertainty_state

        us = materialize_uncertainty_state(trajectory, observations)

        assert isinstance(us, UncertaintyState)
        assert "genetic_testing" in us.dominant_missing_measurements
        assert us.subtype_ambiguity > 0

    def test_materialize_uncertainty_state_missing_measurement_uncertainty(
        self, trajectory, observations
    ):
        from world_model.state_materializer import materialize_uncertainty_state

        us = materialize_uncertainty_state(trajectory, observations)

        assert us.missing_measurement_uncertainty > 0

    def test_materialize_uncertainty_state_id(self, trajectory, observations):
        from world_model.state_materializer import materialize_uncertainty_state

        us = materialize_uncertainty_state(trajectory, observations)

        assert us.id == f"unc:{trajectory.id}"

    def test_materialize_uncertainty_state_subject_ref(self, trajectory, observations):
        from world_model.state_materializer import materialize_uncertainty_state

        us = materialize_uncertainty_state(trajectory, observations)

        assert us.subject_ref == trajectory.patient_ref

    def test_materialize_uncertainty_state_lists_expected_gaps(
        self, trajectory, observations
    ):
        from world_model.state_materializer import materialize_uncertainty_state

        us = materialize_uncertainty_state(trajectory, observations)

        missing = us.dominant_missing_measurements
        # All seven standard gaps should appear
        expected = [
            "genetic_testing",
            "csf_biomarkers",
            "cryptic_exon_splicing_assay",
            "tdp43_in_vivo_measurement",
            "cortical_excitability_tms",
            "transcriptomics",
            "proteomics",
        ]
        for gap in expected:
            assert gap in missing, f"Expected gap {gap!r} not found in {missing}"


# ---------------------------------------------------------------------------
# materialize_state (full snapshot, use_llm=False)
# ---------------------------------------------------------------------------

class TestMaterializeState:
    def test_materialize_state_returns_snapshot(self, trajectory, observations):
        from world_model.state_materializer import materialize_state

        snapshot = materialize_state(trajectory, observations, use_llm=False)

        assert isinstance(snapshot, DiseaseStateSnapshot)
        assert snapshot.type == "DiseaseStateSnapshot"

    def test_materialize_state_functional_state_ref_set(
        self, trajectory, observations
    ):
        from world_model.state_materializer import materialize_state

        snapshot = materialize_state(trajectory, observations, use_llm=False)

        assert snapshot.functional_state_ref is not None
        assert snapshot.functional_state_ref == f"func:{trajectory.id}"

    def test_materialize_state_uncertainty_ref_set(self, trajectory, observations):
        from world_model.state_materializer import materialize_state

        snapshot = materialize_state(trajectory, observations, use_llm=False)

        assert snapshot.uncertainty_ref == f"unc:{trajectory.id}"

    def test_materialize_state_subject_ref(self, trajectory, observations):
        from world_model.state_materializer import materialize_state

        snapshot = materialize_state(trajectory, observations, use_llm=False)

        assert snapshot.subject_ref == trajectory.patient_ref

    def test_materialize_state_no_llm_skips_m_t_r_t(
        self, trajectory, observations
    ):
        """use_llm=False: molecular_state_refs and reversibility_window_ref are absent."""
        from world_model.state_materializer import materialize_state

        snapshot = materialize_state(trajectory, observations, use_llm=False)

        assert snapshot.molecular_state_refs == []
        assert snapshot.reversibility_window_ref is None

    def test_materialize_state_body_has_components(self, trajectory, observations):
        """body dict must contain serialised sub-state components."""
        from world_model.state_materializer import materialize_state

        snapshot = materialize_state(trajectory, observations, use_llm=False)

        assert "functional_state" in snapshot.body
        assert "nmj_state" in snapshot.body
        assert "respiratory_state" in snapshot.body
        assert "uncertainty_state" in snapshot.body
