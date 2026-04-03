"""Tests for Galen KG cross-reference connector."""
import pytest
from research.actions import ActionType, NETWORK_ACTIONS
from connectors.galen_kg import ALS_CROSS_REFERENCE_GENES


class TestGalenKGActionType:
    def test_action_type_exists(self):
        assert hasattr(ActionType, "QUERY_GALEN_KG")
        assert ActionType.QUERY_GALEN_KG.value == "query_galen_kg"

    def test_in_network_actions(self):
        assert ActionType.QUERY_GALEN_KG in NETWORK_ACTIONS


class TestGalenKGGeneList:
    def test_comprehensive_gene_list(self):
        assert "SOD1" in ALS_CROSS_REFERENCE_GENES
        assert "TARDBP" in ALS_CROSS_REFERENCE_GENES
        assert "FUS" in ALS_CROSS_REFERENCE_GENES
        assert "C9orf72" in ALS_CROSS_REFERENCE_GENES
        assert "MTOR" in ALS_CROSS_REFERENCE_GENES

    def test_gene_list_not_empty(self):
        assert len(ALS_CROSS_REFERENCE_GENES) >= 10

    def test_cross_reference_genes_count(self):
        assert len(ALS_CROSS_REFERENCE_GENES) >= 25

    def test_cross_reference_genes_includes_expanded_targets(self):
        for gene in ["ANXA11", "CHCHD10", "KIF5A", "VCP", "UBQLN2",
                      "SQSTM1", "ATXN2", "PFN1", "VAPB", "TUBA4A",
                      "HNRNPA1", "SARM1", "BDNF", "GDNF", "C5"]:
            assert gene in ALS_CROSS_REFERENCE_GENES, f"{gene} missing"


class TestGalenKGConnectorUnit:
    def test_connector_importable(self):
        from connectors.galen_kg import GalenKGConnector
        assert GalenKGConnector is not None

    def test_connector_instantiates(self):
        from connectors.galen_kg import GalenKGConnector
        connector = GalenKGConnector(store=None)
        assert connector is not None

    def test_fetch_without_db_returns_errors(self):
        """When psycopg cannot connect, fetch should return errors gracefully."""
        from connectors.galen_kg import GalenKGConnector
        connector = GalenKGConnector(store=None, database="nonexistent_db_xyz")
        cr = connector.fetch(genes=["SOD1"])
        # Should have errors (can't connect) but should not raise
        assert cr.errors
        assert cr.evidence_items_added == 0

    def test_fetch_accepts_gene_list(self):
        """fetch() should accept a genes parameter."""
        from connectors.galen_kg import GalenKGConnector
        connector = GalenKGConnector(store=None, database="nonexistent_db_xyz")
        cr = connector.fetch(genes=["SOD1", "FUS"])
        assert isinstance(cr.errors, list)
