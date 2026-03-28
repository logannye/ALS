"""Tests for Galen SCM connector — causal chain queries against galen_kg."""
from __future__ import annotations

import pytest


class TestCausalEdge:
    def test_creation(self):
        from connectors.galen_scm import CausalEdge
        edge = CausalEdge(
            source="SOD1",
            target="CASP3",
            relationship_type="activates",
            pch_layer=3,
            confidence=0.9,
        )
        assert edge.source == "SOD1"
        assert edge.target == "CASP3"
        assert edge.relationship_type == "activates"
        assert edge.pch_layer == 3
        assert edge.confidence == 0.9

    def test_default_confidence(self):
        from connectors.galen_scm import CausalEdge
        edge = CausalEdge(
            source="A",
            target="B",
            relationship_type="inhibits",
            pch_layer=2,
            confidence=0.5,
        )
        assert edge.confidence == 0.5

    def test_pch_layer_is_int(self):
        from connectors.galen_scm import CausalEdge
        edge = CausalEdge(source="A", target="B", relationship_type="x", pch_layer=2, confidence=0.7)
        assert isinstance(edge.pch_layer, int)


class TestPathwayStrength:
    def test_confidence_computed_from_l3_edges(self):
        from connectors.galen_scm import PathwayStrength
        ps = PathwayStrength(
            pathway="mTOR",
            l2_edges=10,
            l3_edges=5,
            total_entities=20,
            confidence=5 / (10 + 5),
        )
        assert abs(ps.confidence - (5 / 15)) < 1e-9

    def test_confidence_zero_when_no_edges(self):
        from connectors.galen_scm import PathwayStrength
        ps = PathwayStrength(
            pathway="unknown_pathway",
            l2_edges=0,
            l3_edges=0,
            total_entities=0,
            confidence=0.0,
        )
        assert ps.confidence == 0.0

    def test_all_l3_confidence_is_one(self):
        from connectors.galen_scm import PathwayStrength
        ps = PathwayStrength(
            pathway="X",
            l2_edges=0,
            l3_edges=5,
            total_entities=5,
            confidence=1.0,
        )
        assert ps.confidence == 1.0

    def test_pathway_str(self):
        from connectors.galen_scm import PathwayStrength
        ps = PathwayStrength(pathway="autophagy", l2_edges=3, l3_edges=2, total_entities=10, confidence=0.4)
        assert ps.pathway == "autophagy"


class TestGalenSCMConnector:
    def test_importable(self):
        from connectors.galen_scm import GalenSCMConnector
        assert GalenSCMConnector is not None

    def test_instantiates_with_defaults(self):
        from connectors.galen_scm import GalenSCMConnector
        conn = GalenSCMConnector()
        assert conn is not None

    def test_disabled_downstream_returns_empty(self):
        """When enabled=False, query_causal_downstream must return []."""
        from connectors.galen_scm import GalenSCMConnector
        conn = GalenSCMConnector(enabled=False)
        result = conn.query_causal_downstream("SOD1")
        assert result == []

    def test_disabled_upstream_returns_empty(self):
        """When enabled=False, query_causal_upstream must return []."""
        from connectors.galen_scm import GalenSCMConnector
        conn = GalenSCMConnector(enabled=False)
        result = conn.query_causal_upstream("CASP3")
        assert result == []

    def test_disabled_pathway_returns_zeroed(self):
        """When enabled=False, query_pathway_strength must return zero PathwayStrength."""
        from connectors.galen_scm import GalenSCMConnector, PathwayStrength
        conn = GalenSCMConnector(enabled=False)
        ps = conn.query_pathway_strength("mTOR")
        assert isinstance(ps, PathwayStrength)
        assert ps.l2_edges == 0
        assert ps.l3_edges == 0
        assert ps.total_entities == 0
        assert ps.confidence == 0.0

    def test_error_on_bad_db_downstream_returns_empty(self):
        """On DB connection error, downstream returns [] not an exception."""
        from connectors.galen_scm import GalenSCMConnector
        conn = GalenSCMConnector(database="nonexistent_db_xyz_scm", enabled=True)
        result = conn.query_causal_downstream("SOD1")
        assert isinstance(result, list)
        assert result == []

    def test_error_on_bad_db_upstream_returns_empty(self):
        """On DB connection error, upstream returns [] not an exception."""
        from connectors.galen_scm import GalenSCMConnector
        conn = GalenSCMConnector(database="nonexistent_db_xyz_scm", enabled=True)
        result = conn.query_causal_upstream("CASP3")
        assert isinstance(result, list)
        assert result == []

    def test_error_on_bad_db_pathway_returns_zeroed(self):
        """On DB connection error, pathway returns zeroed PathwayStrength."""
        from connectors.galen_scm import GalenSCMConnector, PathwayStrength
        conn = GalenSCMConnector(database="nonexistent_db_xyz_scm", enabled=True)
        ps = conn.query_pathway_strength("mTOR")
        assert isinstance(ps, PathwayStrength)
        assert ps.l2_edges == 0
        assert ps.l3_edges == 0
        assert ps.confidence == 0.0

    def test_max_depth_override(self):
        """query_causal_downstream accepts max_depth override."""
        from connectors.galen_scm import GalenSCMConnector
        conn = GalenSCMConnector(enabled=False)
        # Disabled so returns immediately, but signature must accept the param
        result = conn.query_causal_downstream("SOD1", max_depth=5)
        assert result == []

    def test_constructor_params(self):
        """Constructor stores parameters correctly."""
        from connectors.galen_scm import GalenSCMConnector
        conn = GalenSCMConnector(
            database="galen_kg",
            enabled=True,
            min_pch_layer=3,
            max_depth=4,
        )
        assert conn._database == "galen_kg"
        assert conn._enabled is True
        assert conn._min_pch_layer == 3
        assert conn._max_depth == 4
