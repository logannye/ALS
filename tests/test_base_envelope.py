"""Tests for ontology.base — BaseEnvelope and its sub-models."""
import json
import pytest
from datetime import datetime, timezone

from ontology.base import BaseEnvelope, TimeFields, Provenance, Uncertainty, Privacy
from ontology.enums import ObjectStatus, PrivacyClass, ConfidenceBand, SourceSystem


class TestBaseEnvelopeMinimal:
    def _minimal(self, **overrides) -> dict:
        base = {
            "id": "patient:erik_draper",
            "type": "Patient",
            "body": {},
        }
        base.update(overrides)
        return base

    def test_creation_with_minimal_fields(self):
        env = BaseEnvelope(**self._minimal())
        assert env.id == "patient:erik_draper"
        assert env.type == "Patient"

    def test_default_status_is_active(self):
        env = BaseEnvelope(**self._minimal())
        assert env.status == ObjectStatus.active

    def test_default_schema_version(self):
        env = BaseEnvelope(**self._minimal())
        assert env.schema_version == "1.0"

    def test_default_tenant_id(self):
        env = BaseEnvelope(**self._minimal())
        assert env.tenant_id == "erik_default"

    def test_default_privacy_is_restricted(self):
        env = BaseEnvelope(**self._minimal())
        assert env.privacy.classification == PrivacyClass.restricted

    def test_default_uncertainty_confidence_is_none(self):
        env = BaseEnvelope(**self._minimal())
        assert env.uncertainty.confidence is None

    def test_body_can_hold_arbitrary_data(self):
        env = BaseEnvelope(**self._minimal(body={"alsfrs_r": 42, "onset_region": "bulbar"}))
        assert env.body["alsfrs_r"] == 42


class TestTimeFields:
    def test_recorded_at_auto_populated(self):
        tf = TimeFields()
        assert tf.recorded_at is not None
        assert isinstance(tf.recorded_at, datetime)

    def test_optional_fields_default_to_none(self):
        tf = TimeFields()
        assert tf.observed_at is None
        assert tf.effective_at is None
        assert tf.valid_from is None
        assert tf.valid_to is None

    def test_observed_at_can_be_set(self):
        now = datetime.now(timezone.utc)
        tf = TimeFields(observed_at=now)
        assert tf.observed_at == now

    def test_embedded_in_envelope(self):
        env = BaseEnvelope(id="x:y", type="Test", body={})
        assert env.time.recorded_at is not None


class TestProvenance:
    def test_default_source_system_is_manual(self):
        p = Provenance()
        assert p.source_system == SourceSystem.manual

    def test_optional_fields_default_to_none(self):
        p = Provenance()
        assert p.source_artifact_id is None
        assert p.asserted_by is None
        assert p.trace_id is None

    def test_can_set_all_fields(self):
        p = Provenance(
            source_system=SourceSystem.ehr,
            source_artifact_id="EHR-001",
            asserted_by="dr_miller",
            trace_id="trace-abc-123",
        )
        assert p.source_system == SourceSystem.ehr
        assert p.source_artifact_id == "EHR-001"
        assert p.asserted_by == "dr_miller"
        assert p.trace_id == "trace-abc-123"


class TestUncertainty:
    def test_sources_default_to_empty_list(self):
        u = Uncertainty()
        assert u.sources == []

    def test_confidence_and_band_can_be_set(self):
        u = Uncertainty(confidence=0.85, confidence_band=ConfidenceBand.high, sources=["pubmed:123"])
        assert u.confidence == 0.85
        assert u.confidence_band == ConfidenceBand.high
        assert "pubmed:123" in u.sources

    def test_confidence_is_optional(self):
        u = Uncertainty()
        assert u.confidence is None
        assert u.confidence_band is None


class TestPrivacy:
    def test_default_classification_is_restricted(self):
        p = Privacy()
        assert p.classification == PrivacyClass.restricted

    def test_phi_can_be_set(self):
        p = Privacy(classification=PrivacyClass.phi)
        assert p.classification == PrivacyClass.phi


class TestBaseEnvelopeValidation:
    def test_rejects_empty_id(self):
        with pytest.raises(Exception):
            BaseEnvelope(id="", type="Patient", body={})

    def test_rejects_blank_id(self):
        with pytest.raises(Exception):
            BaseEnvelope(id="   ", type="Patient", body={})

    def test_rejects_empty_type(self):
        with pytest.raises(Exception):
            BaseEnvelope(id="patient:test", type="", body={})

    def test_rejects_blank_type(self):
        with pytest.raises(Exception):
            BaseEnvelope(id="patient:test", type="   ", body={})


class TestBaseEnvelopeJSONRoundtrip:
    def _make_env(self) -> BaseEnvelope:
        return BaseEnvelope(
            id="patient:erik_draper",
            type="Patient",
            schema_version="1.0",
            tenant_id="erik_default",
            status=ObjectStatus.active,
            provenance=Provenance(
                source_system=SourceSystem.ehr,
                source_artifact_id="EHR-001",
                asserted_by="dr_miller",
            ),
            uncertainty=Uncertainty(
                confidence=0.9,
                confidence_band=ConfidenceBand.very_high,
                sources=["pubmed:12345678"],
            ),
            privacy=Privacy(classification=PrivacyClass.phi),
            body={"alsfrs_r": 42},
        )

    def test_model_dump_json_is_valid_json(self):
        env = self._make_env()
        serialised = env.model_dump_json()
        parsed = json.loads(serialised)
        assert parsed["id"] == "patient:erik_draper"

    def test_roundtrip_preserves_all_fields(self):
        env = self._make_env()
        serialised = env.model_dump_json()
        restored = BaseEnvelope.model_validate_json(serialised)

        assert restored.id == env.id
        assert restored.type == env.type
        assert restored.status == env.status
        assert restored.provenance.source_system == env.provenance.source_system
        assert restored.provenance.asserted_by == env.provenance.asserted_by
        assert restored.uncertainty.confidence == env.uncertainty.confidence
        assert restored.uncertainty.confidence_band == env.uncertainty.confidence_band
        assert restored.privacy.classification == env.privacy.classification
        assert restored.body == env.body

    def test_roundtrip_preserves_time_recorded_at(self):
        env = self._make_env()
        serialised = env.model_dump_json()
        restored = BaseEnvelope.model_validate_json(serialised)
        assert restored.time.recorded_at == env.time.recorded_at
