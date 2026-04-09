"""Tests for mechanism→mechanism chain builder.

TDD: These tests define the expected behaviour of the chain builder
before implementation exists.
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge(source_id: str, target_id: str, confidence: float = 0.8,
          evidence_id: str = "ev:1") -> dict:
    """Create a minimal contributes_to edge dict as returned by the DB query."""
    return {
        "source_id": source_id,
        "target_id": target_id,
        "relationship_type": "contributes_to",
        "confidence": confidence,
        "evidence": evidence_id,
    }


# ---------------------------------------------------------------------------
# Unit tests – _find_mechanism_chains
# ---------------------------------------------------------------------------

class TestFindMechanismChains:

    def test_identifies_chain_from_gene_mechanism_pairs(self):
        """Two mechanisms sharing a gene produce one chain link."""
        from research.chain_builder import _find_mechanism_chains

        edges = [
            _edge("gene:tardbp", "mechanism:tdp_43_mislocalization", 0.9, "ev:1"),
            _edge("gene:tardbp", "mechanism:cryptic_exon_inclusion", 0.8, "ev:2"),
        ]
        chains = _find_mechanism_chains(edges)
        assert len(chains) >= 1
        # The two mechanisms should be linked via gene:tardbp
        src_targets = {(c["source_mechanism"], c["target_mechanism"]) for c in chains}
        assert (
            ("mechanism:tdp_43_mislocalization", "mechanism:cryptic_exon_inclusion") in src_targets
            or ("mechanism:cryptic_exon_inclusion", "mechanism:tdp_43_mislocalization") in src_targets
        )

    def test_chain_link_has_required_fields(self):
        from research.chain_builder import _find_mechanism_chains

        edges = [
            _edge("gene:sod1", "mechanism:oxidative_stress", 0.9, "ev:10"),
            _edge("gene:sod1", "mechanism:motor_neuron_death", 0.7, "ev:11"),
        ]
        chains = _find_mechanism_chains(edges)
        assert len(chains) >= 1
        link = chains[0]
        for key in ("source_mechanism", "target_mechanism", "via_gene", "confidence", "evidence_ids"):
            assert key in link, f"Missing required field: {key}"

    def test_no_self_loops(self):
        """A mechanism appearing twice for the same gene must not link to itself."""
        from research.chain_builder import _find_mechanism_chains

        edges = [
            _edge("gene:fus", "mechanism:rna_granule_dysfunction", 0.9, "ev:a"),
            _edge("gene:fus", "mechanism:rna_granule_dysfunction", 0.85, "ev:b"),
        ]
        chains = _find_mechanism_chains(edges)
        for link in chains:
            assert link["source_mechanism"] != link["target_mechanism"]

    def test_confidence_is_min_of_supporting_edges(self):
        from research.chain_builder import _find_mechanism_chains

        edges = [
            _edge("gene:tardbp", "mechanism:a", 0.9, "ev:1"),
            _edge("gene:tardbp", "mechanism:b", 0.6, "ev:2"),
        ]
        chains = _find_mechanism_chains(edges)
        assert len(chains) >= 1
        # confidence should be min(0.9, 0.6) = 0.6
        link = chains[0]
        assert link["confidence"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Unit tests – _build_chain_relationship
# ---------------------------------------------------------------------------

class TestBuildChainRelationship:

    def test_creates_contributes_to_edge(self):
        from research.chain_builder import _build_chain_relationship

        rel = _build_chain_relationship(
            source_mechanism="mechanism:tdp_43_mislocalization",
            target_mechanism="mechanism:cryptic_exon_inclusion",
            via_gene="gene:tardbp",
            confidence=0.8,
            evidence_ids=["ev:1", "ev:2"],
        )
        assert rel["relationship_type"] == "contributes_to"
        assert rel["source_id"] == "mechanism:tdp_43_mislocalization"
        assert rel["target_id"] == "mechanism:cryptic_exon_inclusion"
        assert rel["confidence"] == 0.8
        assert rel["evidence_type"] == "inferred_chain"

    def test_pch_layer_is_2(self):
        from research.chain_builder import _build_chain_relationship

        rel = _build_chain_relationship(
            source_mechanism="mechanism:a",
            target_mechanism="mechanism:b",
            via_gene="gene:x",
            confidence=0.7,
            evidence_ids=["ev:1"],
        )
        assert rel["pch_layer"] == 2

    def test_id_is_sha256_based(self):
        from research.chain_builder import _build_chain_relationship

        rel = _build_chain_relationship(
            source_mechanism="mechanism:a",
            target_mechanism="mechanism:b",
            via_gene="gene:x",
            confidence=0.7,
            evidence_ids=["ev:1"],
        )
        expected_raw = "mechanism:a|contributes_to|mechanism:b"
        expected_hash = hashlib.sha256(expected_raw.encode()).hexdigest()[:12]
        assert rel["id"] == f"rel:{expected_hash}"


# ---------------------------------------------------------------------------
# Integration-level – build_mechanism_chains (dry_run)
# ---------------------------------------------------------------------------

class TestBuildMechanismChains:

    @patch("research.chain_builder.get_connection")
    def test_returns_stats_dict(self, mock_conn_ctx):
        """build_mechanism_chains(dry_run=True) returns dict with required keys."""
        from research.chain_builder import build_mechanism_chains

        # Mock the DB to return some edges
        mock_conn = MagicMock()
        mock_conn_ctx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=False)
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = [
            ("gene:tardbp", "mechanism:a", 0.9, "ev:1"),
            ("gene:tardbp", "mechanism:b", 0.8, "ev:2"),
        ]

        result = build_mechanism_chains(dry_run=True)
        assert isinstance(result, dict)
        assert "chains_found" in result
        assert "edges_created" in result
        # In dry_run mode, edges_created should be 0
        assert result["edges_created"] == 0
        assert result["chains_found"] >= 1
