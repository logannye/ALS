"""Tests for EvidenceStore — TDD, failing first before implementation."""

import uuid

import pytest

from ontology.enums import (
    EvidenceDirection,
    EvidenceStrength,
    InterventionClass,
    ProtocolLayer,
)
from ontology.evidence import EvidenceItem
from ontology.intervention import Intervention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence_item(
    *,
    protocol_layer: str = "pathology_reversal",
    mechanism_target: str | None = None,
    suffix: str = "",
) -> EvidenceItem:
    uid = uuid.uuid4().hex[:8]
    body: dict = {"protocol_layer": protocol_layer}
    if mechanism_target:
        body["mechanism_target"] = mechanism_target
    item = EvidenceItem(
        id=f"evidenceitem:test_{uid}{suffix}",
        claim=f"Test claim {uid}",
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.moderate,
        source_refs=["pubmed:123"],
        notes="test note",
        body=body,
    )
    return item


def _make_intervention(suffix: str = "") -> Intervention:
    uid = uuid.uuid4().hex[:8]
    return Intervention(
        id=f"intervention:test_{uid}{suffix}",
        name=f"Test Drug {uid}",
        intervention_class=InterventionClass.drug,
        targets=["SOD1"],
        protocol_layer=ProtocolLayer.pathology_reversal,
        route="oral",
        intended_effects=["neuroprotection"],
        known_risks=["nausea"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvidenceStore:

    def test_upsert_and_retrieve_evidence_item(self, db_available, evidence_store):
        """Store an EvidenceItem and verify claim + body.protocol_layer survive roundtrip."""
        item = _make_evidence_item(protocol_layer="circuit_stabilization")
        evidence_store.upsert_evidence_item(item)

        result = evidence_store.get_evidence_item(item.id)
        assert result is not None
        assert result["id"] == item.id
        assert result["type"] == "EvidenceItem"
        assert result["claim"] == item.claim
        assert result["body"]["protocol_layer"] == "circuit_stabilization"

        evidence_store.delete(item.id)

    def test_upsert_and_retrieve_intervention(self, db_available, evidence_store):
        """Store an Intervention and verify name survives roundtrip."""
        interv = _make_intervention()
        evidence_store.upsert_intervention(interv)

        result = evidence_store.get_intervention(interv.id)
        assert result is not None
        assert result["id"] == interv.id
        assert result["type"] == "Intervention"
        assert result["name"] == interv.name

        evidence_store.delete(interv.id)

    def test_query_by_protocol_layer(self, db_available, evidence_store):
        """Insert 3 EvidenceItems with pathology_reversal; query returns >= 3."""
        items = [_make_evidence_item(protocol_layer="pathology_reversal", suffix=f"_pl{i}") for i in range(3)]
        for item in items:
            evidence_store.upsert_evidence_item(item)

        results = evidence_store.query_by_protocol_layer("pathology_reversal")
        assert len(results) >= 3

        for item in items:
            evidence_store.delete(item.id)

    def test_upsert_is_idempotent(self, db_available, evidence_store):
        """Inserting the same EvidenceItem twice must not raise an error."""
        item = _make_evidence_item()
        evidence_store.upsert_evidence_item(item)
        # Second upsert — must not raise
        evidence_store.upsert_evidence_item(item)

        result = evidence_store.get_evidence_item(item.id)
        assert result is not None
        assert result["claim"] == item.claim

        evidence_store.delete(item.id)

    def test_count_by_type(self, db_available, evidence_store):
        """After inserting one EvidenceItem, count_by_type >= 1."""
        item = _make_evidence_item()
        evidence_store.upsert_evidence_item(item)

        count = evidence_store.count_by_type("EvidenceItem")
        assert count >= 1

        evidence_store.delete(item.id)

    def test_query_by_mechanism_target(self, db_available, evidence_store):
        """Insert items with a mechanism_target; query returns them."""
        items = [
            _make_evidence_item(mechanism_target="TDP-43", suffix=f"_mt{i}")
            for i in range(2)
        ]
        for item in items:
            evidence_store.upsert_evidence_item(item)

        results = evidence_store.query_by_mechanism_target("TDP-43")
        assert len(results) >= 2

        for item in items:
            evidence_store.delete(item.id)

    def test_get_evidence_item_returns_none_for_missing(self, db_available, evidence_store):
        """Querying a non-existent ID returns None."""
        result = evidence_store.get_evidence_item("evidenceitem:does_not_exist_xyz")
        assert result is None

    def test_get_intervention_returns_none_for_missing(self, db_available, evidence_store):
        """Querying a non-existent Intervention ID returns None."""
        result = evidence_store.get_intervention("intervention:does_not_exist_xyz")
        assert result is None
