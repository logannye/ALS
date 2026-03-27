"""Tests for causal chain construction and deepening."""
from __future__ import annotations
import pytest
from research.causal_chains import CausalChain, CausalLink, get_chain_depth

class TestCausalLink:
    def test_construction(self):
        link = CausalLink(source="pridopidine", target="sigma-1R activation",
                          mechanism="agonist binding", evidence_ref="evi:sigma1r_pridopidine", confidence=0.85)
        assert link.source == "pridopidine"
        assert link.confidence == 0.85

class TestCausalChain:
    def test_empty_chain(self):
        chain = CausalChain(intervention_id="int:pridopidine")
        assert chain.depth() == 0

    def test_chain_with_links(self):
        chain = CausalChain(intervention_id="int:pridopidine")
        chain.add_link(CausalLink(source="pridopidine", target="sigma-1R activation",
                                  mechanism="agonist binding", evidence_ref="evi:a", confidence=0.9))
        chain.add_link(CausalLink(source="sigma-1R activation", target="ER calcium homeostasis",
                                  mechanism="calcium channel modulation", evidence_ref="evi:b", confidence=0.7))
        assert chain.depth() == 2

    def test_weakest_link(self):
        chain = CausalChain(intervention_id="int:pridopidine")
        chain.add_link(CausalLink(source="A", target="B", mechanism="x", evidence_ref="evi:strong", confidence=0.9))
        chain.add_link(CausalLink(source="B", target="C", mechanism="y", evidence_ref="evi:weak", confidence=0.3))
        weak = chain.weakest_link()
        assert weak is not None
        assert weak.evidence_ref == "evi:weak"

    def test_to_dict(self):
        chain = CausalChain(intervention_id="int:test")
        chain.add_link(CausalLink(source="A", target="B", mechanism="x", evidence_ref="evi:a", confidence=0.8))
        d = chain.to_dict()
        assert d["intervention_id"] == "int:test"
        assert len(d["links"]) == 1
        assert d["depth"] == 1

    def test_all_evidence_refs(self):
        chain = CausalChain(intervention_id="int:test")
        chain.add_link(CausalLink(source="A", target="B", mechanism="x", evidence_ref="evi:1", confidence=0.8))
        chain.add_link(CausalLink(source="B", target="C", mechanism="y", evidence_ref="evi:2", confidence=0.7))
        refs = chain.all_evidence_refs()
        assert refs == ["evi:1", "evi:2"]

class TestGetChainDepth:
    def test_returns_zero_for_missing(self):
        assert get_chain_depth({}, "int:unknown") == 0

    def test_returns_depth(self):
        chains = {"int:a": CausalChain(intervention_id="int:a")}
        chains["int:a"].add_link(CausalLink(source="A", target="B", mechanism="x", evidence_ref="evi:1", confidence=0.8))
        assert get_chain_depth(chains, "int:a") == 1

class TestPathwayGroundedLink:
    def test_finds_pathway_connection(self):
        from research.causal_chains import pathway_grounded_link
        evidence = [{"id": "evi:reactome:R-HSA-123", "body": {"pathway_name": "Cellular response to ER stress", "data_source": "reactome"}}]
        link = pathway_grounded_link("sigma-1R", "ER stress", evidence)
        assert link is not None
        assert link.confidence == 0.95
        assert "reactome" in link.evidence_ref

    def test_returns_none_when_no_match(self):
        from research.causal_chains import pathway_grounded_link
        evidence = [{"id": "evi:reactome:R-HSA-999", "body": {"pathway_name": "Cholesterol biosynthesis", "data_source": "reactome"}}]
        link = pathway_grounded_link("sigma-1R", "TDP-43 proteostasis", evidence)
        assert link is None
