"""Tests for ontology.relations — relation vocabulary and observational guards."""
import pytest

from ontology.relations import (
    RELATION_TYPES,
    OBSERVATIONAL_RELATION_TYPES,
    is_observational,
    get_relation_category,
)


# ---------------------------------------------------------------------------
# Core relations exist
# ---------------------------------------------------------------------------

class TestCoreRelationsExist:
    def test_causes_exists(self):
        assert "causes" in RELATION_TYPES

    def test_contributes_to_exists(self):
        assert "contributes_to" in RELATION_TYPES

    def test_targets_exists(self):
        assert "targets" in RELATION_TYPES

    def test_treats_exists(self):
        assert "treats" in RELATION_TYPES

    def test_supports_exists(self):
        assert "supports" in RELATION_TYPES

    def test_refutes_exists(self):
        assert "refutes" in RELATION_TYPES

    def test_all_entries_have_category(self):
        for name, meta in RELATION_TYPES.items():
            assert "category" in meta, f"{name} missing 'category' key"
            assert isinstance(meta["category"], str), f"{name} category is not a str"


# ---------------------------------------------------------------------------
# OBSERVATIONAL_RELATION_TYPES is a frozenset
# ---------------------------------------------------------------------------

class TestObservationalRelationTypes:
    def test_is_frozenset(self):
        assert isinstance(OBSERVATIONAL_RELATION_TYPES, frozenset)

    def test_variant_in_gene_in_set(self):
        assert "variant_in_gene" in OBSERVATIONAL_RELATION_TYPES

    def test_located_in_in_set(self):
        assert "located_in" in OBSERVATIONAL_RELATION_TYPES

    def test_member_of_in_set(self):
        assert "member_of" in OBSERVATIONAL_RELATION_TYPES

    def test_observed_in_in_set(self):
        assert "observed_in" in OBSERVATIONAL_RELATION_TYPES

    def test_associated_with_in_set(self):
        assert "associated_with" in OBSERVATIONAL_RELATION_TYPES

    def test_derived_from_in_set(self):
        assert "derived_from" in OBSERVATIONAL_RELATION_TYPES

    def test_measures_in_set(self):
        assert "measures" in OBSERVATIONAL_RELATION_TYPES

    def test_has_part_in_set(self):
        assert "has_part" in OBSERVATIONAL_RELATION_TYPES

    def test_part_of_in_set(self):
        assert "part_of" in OBSERVATIONAL_RELATION_TYPES

    def test_subtype_of_in_set(self):
        assert "subtype_of" in OBSERVATIONAL_RELATION_TYPES

    def test_instance_of_in_set(self):
        assert "instance_of" in OBSERVATIONAL_RELATION_TYPES

    def test_expressed_in_in_set(self):
        assert "expressed_in" in OBSERVATIONAL_RELATION_TYPES


# ---------------------------------------------------------------------------
# is_observational
# ---------------------------------------------------------------------------

class TestIsObservational:
    def test_variant_in_gene_is_observational(self):
        assert is_observational("variant_in_gene") is True

    def test_located_in_is_observational(self):
        assert is_observational("located_in") is True

    def test_member_of_is_observational(self):
        assert is_observational("member_of") is True

    def test_observed_in_is_observational(self):
        assert is_observational("observed_in") is True

    def test_associated_with_is_observational(self):
        assert is_observational("associated_with") is True

    def test_derived_from_is_observational(self):
        assert is_observational("derived_from") is True

    def test_measures_is_observational(self):
        assert is_observational("measures") is True

    # Causal relations must NOT be observational
    def test_causes_is_not_observational(self):
        assert is_observational("causes") is False

    def test_contributes_to_is_not_observational(self):
        assert is_observational("contributes_to") is False

    def test_amplifies_is_not_observational(self):
        assert is_observational("amplifies") is False

    def test_suppresses_is_not_observational(self):
        assert is_observational("suppresses") is False

    # Unknown relation is not observational
    def test_unknown_relation_is_not_observational(self):
        assert is_observational("nonexistent_relation") is False


# ---------------------------------------------------------------------------
# get_relation_category
# ---------------------------------------------------------------------------

class TestGetRelationCategory:
    def test_causes_is_causal(self):
        assert get_relation_category("causes") == "causal"

    def test_contributes_to_is_causal(self):
        assert get_relation_category("contributes_to") == "causal"

    def test_amplifies_is_causal(self):
        assert get_relation_category("amplifies") == "causal"

    def test_suppresses_is_causal(self):
        assert get_relation_category("suppresses") == "causal"

    def test_has_part_is_structural(self):
        assert get_relation_category("has_part") == "structural"

    def test_part_of_is_structural(self):
        assert get_relation_category("part_of") == "structural"

    def test_subtype_of_is_structural(self):
        assert get_relation_category("subtype_of") == "structural"

    def test_observed_in_is_observational(self):
        assert get_relation_category("observed_in") == "observational"

    def test_associated_with_is_observational(self):
        assert get_relation_category("associated_with") == "observational"

    def test_precedes_is_temporal(self):
        assert get_relation_category("precedes") == "temporal"

    def test_follows_is_temporal(self):
        assert get_relation_category("follows") == "temporal"

    def test_supports_is_evidential(self):
        assert get_relation_category("supports") == "evidential"

    def test_refutes_is_evidential(self):
        assert get_relation_category("refutes") == "evidential"

    def test_targets_is_therapeutic(self):
        assert get_relation_category("targets") == "therapeutic"

    def test_treats_is_therapeutic(self):
        assert get_relation_category("treats") == "therapeutic"

    def test_constrained_by_is_governance(self):
        assert get_relation_category("constrained_by") == "governance"

    def test_unknown_returns_unknown(self):
        assert get_relation_category("totally_made_up") == "unknown"
