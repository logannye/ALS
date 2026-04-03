"""Tests for entity_extractor relationship inference fixes.

Verifies that all connector body formats produce correct entities AND
relationships through _extract_entities_from_body and _infer_relationships.
"""
import pytest

from knowledge_quality.entity_extractor import (
    _extract_entities_from_body,
    _infer_relationships,
)


# ---------------------------------------------------------------------------
# Bug 1: Field name mismatches — connectors using 'gene' not 'gene_symbol'
# ---------------------------------------------------------------------------

class TestFieldNameMapping:
    """Connectors that store 'gene' (not 'gene_symbol') must produce gene entities."""

    def test_pharmgkb_gene_field_produces_gene_entity(self):
        """PharmGKB stores 'gene' not 'gene_symbol' — must still extract gene."""
        body = {
            "pharmgkb_id": "PA166",
            "drug_name": "riluzole",
            "gene": "SOD1",
            "annotation": "test",
            "level": "1A",
            "category": "dosing",
            "pch_layer": 2,
            "data_source": "pharmgkb",
            "claim": "PharmGKB: riluzole - SOD1",
        }
        entities = _extract_entities_from_body(body, "evi:pharmgkb:PA166_SOD1")
        types = {e["entity_type"] for e in entities}
        assert "drug" in types, "drug_name field must produce drug entity"
        assert "gene" in types, "'gene' field must produce gene entity (not just 'gene_symbol')"

    def test_pharmgkb_produces_drug_targets_gene_relationship(self):
        """PharmGKB with drug + gene should produce a 'targets' relationship."""
        body = {
            "drug_name": "riluzole",
            "gene": "SOD1",
            "pch_layer": 2,
            "evidence_strength": "strong",
            "claim": "PharmGKB: riluzole - SOD1",
        }
        entities = _extract_entities_from_body(body, "evi:test:1")
        rels = _infer_relationships(entities, body, "evi:test:1")
        assert len(rels) >= 1, "drug + gene co-occurrence must produce relationship"
        assert any(r["relationship_type"] == "targets" for r in rels)

    def test_clinvar_gene_field_produces_gene_entity(self):
        """ClinVar stores 'gene' not 'gene_symbol'."""
        body = {
            "variation_id": 12345,
            "variant_name": "p.Ala5Val",
            "gene": "FUS",
            "clinical_significance": "Pathogenic",
            "review_status": "criteria_provided",
            "pch_layer": 1,
            "data_source": "clinvar",
            "claim": "ClinVar: p.Ala5Val — Pathogenic",
        }
        entities = _extract_entities_from_body(body, "evi:clinvar:12345")
        types = {e["entity_type"] for e in entities}
        assert "gene" in types, "'gene' field must produce gene entity"

    def test_drugbank_gene_field_produces_gene_entity(self):
        """DrugBank Local stores 'gene' (not 'gene_symbol')."""
        body = {
            "gene": "TARDBP",
            "drug_name": "edaravone",
            "pch_layer": 1,
            "claim": "DrugBank: edaravone targets TARDBP",
        }
        entities = _extract_entities_from_body(body, "evi:drugbank:1")
        types = {e["entity_type"] for e in entities}
        assert "gene" in types
        assert "drug" in types

    def test_drugbank_gene_drug_produces_relationship(self):
        """DrugBank with gene + drug must produce 'targets' relationship."""
        body = {
            "gene": "TARDBP",
            "drug_name": "edaravone",
            "pch_layer": 1,
            "evidence_strength": "moderate",
            "claim": "DrugBank: edaravone targets TARDBP",
        }
        entities = _extract_entities_from_body(body, "evi:test:2")
        rels = _infer_relationships(entities, body, "evi:test:2")
        assert len(rels) >= 1


# ---------------------------------------------------------------------------
# Bug 2: No relationship rules for 'protein' entity type
# ---------------------------------------------------------------------------

class TestProteinRelationships:
    """Protein entities must participate in relationship inference."""

    def test_bindingdb_drug_protein_produces_relationship(self):
        """BindingDB: drug_name + target_name (protein) must produce relationship."""
        body = {
            "drug_name": "riluzole",
            "target_name": "EAAT2",
            "ki_nm": 10.5,
            "evidence_strength": "strong",
            "pch_layer": 2,
            "claim": "riluzole binds EAAT2 with Ki=10.5nM",
        }
        entities = _extract_entities_from_body(body, "evi:bindingdb:1")
        types = {e["entity_type"] for e in entities}
        assert "drug" in types
        assert "protein" in types

        rels = _infer_relationships(entities, body, "evi:bindingdb:1")
        assert len(rels) >= 1, "drug + protein co-occurrence must produce 'binds' relationship"
        assert any(r["relationship_type"] == "binds" for r in rels)

    def test_galen_kg_protein_protein_produces_relationship(self):
        """Galen KG: source_name + target_name (both protein) must produce relationship."""
        body = {
            "source_name": "TDP-43",
            "target_name": "FUS",
            "relationship_type": "interacts_with",
            "provenance_source_system": "galen_cross_reference",
            "claim": "TDP-43 → FUS (interacts_with)",
        }
        entities = _extract_entities_from_body(body, "evi:galen:1")
        types = {e["entity_type"] for e in entities}
        assert "protein" in types

        rels = _infer_relationships(entities, body, "evi:galen:1")
        assert len(rels) >= 1, "protein + protein co-occurrence must produce relationship"
        assert any(r["relationship_type"] == "interacts_with" for r in rels)

    def test_protein_protein_relationship_is_observational(self):
        """Protein-protein relationships should be capped at L1 (observational)."""
        body = {
            "source_name": "SOD1",
            "target_name": "OPTN",
            "pch_layer": 2,
            "claim": "SOD1 → OPTN",
        }
        entities = _extract_entities_from_body(body, "evi:test:pp")
        rels = _infer_relationships(entities, body, "evi:test:pp")
        for r in rels:
            assert r["pch_layer"] <= 1, "protein-protein is observational, max L1"


# ---------------------------------------------------------------------------
# Existing rules still work (regression guard)
# ---------------------------------------------------------------------------

class TestExistingRulesUnchanged:
    """Existing relationship inference rules must not regress."""

    def test_gene_gene_associated_with(self):
        """Two genes → associated_with relationship."""
        body = {
            "gene_a": "SOD1",
            "gene_b": "FUS",
            "pch_layer": 1,
            "evidence_strength": "moderate",
            "claim": "STRING: SOD1-FUS interaction",
        }
        entities = _extract_entities_from_body(body, "evi:string:1")
        rels = _infer_relationships(entities, body, "evi:string:1")
        assert any(r["relationship_type"] == "associated_with" for r in rels)

    def test_drug_mechanism_suppresses(self):
        """Drug + mechanism → suppresses relationship."""
        body = {
            "intervention_ref": "int:riluzole",
            "mechanism_target": "glutamate_excitotoxicity",
            "direction": "supports",
            "pch_layer": 2,
            "evidence_strength": "strong",
            "claim": "riluzole suppresses glutamate excitotoxicity",
        }
        entities = _extract_entities_from_body(body, "evi:test:dm")
        rels = _infer_relationships(entities, body, "evi:test:dm")
        assert any(r["relationship_type"] == "suppresses" for r in rels)

    def test_gene_mechanism_contributes_to(self):
        """Gene + mechanism → contributes_to relationship."""
        body = {
            "gene_symbol": "SOD1",
            "mechanism_target": "oxidative_stress",
            "pch_layer": 2,
            "evidence_strength": "moderate",
            "claim": "SOD1 contributes to oxidative stress",
        }
        entities = _extract_entities_from_body(body, "evi:test:gm")
        rels = _infer_relationships(entities, body, "evi:test:gm")
        assert any(r["relationship_type"] == "contributes_to" for r in rels)

    def test_claim_text_gene_extraction_still_works(self):
        """Gene names in claim text should still extract gene entities."""
        body = {
            "claim": "Study shows SOD1 and FUS mutations are co-occurring in ALS patients",
            "pch_layer": 1,
        }
        entities = _extract_entities_from_body(body, "evi:test:claim")
        gene_names = {e["name"] for e in entities if e["entity_type"] == "gene"}
        assert "SOD1" in gene_names
        assert "FUS" in gene_names

    def test_claim_text_drug_extraction_still_works(self):
        """Drug names in claim text should still extract drug entities."""
        body = {
            "claim": "riluzole shows neuroprotective effects",
            "pch_layer": 1,
        }
        entities = _extract_entities_from_body(body, "evi:test:drugclaim")
        drug_names = {e["name"] for e in entities if e["entity_type"] == "drug"}
        assert "riluzole" in drug_names
