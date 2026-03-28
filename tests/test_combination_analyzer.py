"""Tests for world_model.combination_analyzer — Drug Combination Synergy Modeling."""
import pytest
from research.causal_chains import CausalChain, CausalLink
from world_model.combination_analyzer import (
    InteractionFlag,
    CombinationAnalysis,
    compute_pathway_overlap,
    analyze_combinations,
    apply_interaction_flags,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chain(intervention_id: str, targets: list[str]) -> CausalChain:
    """Build a CausalChain whose links each have a target node from *targets*."""
    chain = CausalChain(intervention_id=intervention_id)
    for i, target in enumerate(targets):
        chain.add_link(CausalLink(
            source=f"src_{i}",
            target=target,
            mechanism=f"mech_{i}",
            evidence_ref=f"ref_{i}",
            confidence=0.8,
        ))
    return chain


# ---------------------------------------------------------------------------
# TestPathwayOverlap
# ---------------------------------------------------------------------------

class TestPathwayOverlap:
    def test_identical_chains_return_high_overlap(self):
        nodes = ["SOD1", "TDP43", "FUS", "ATXN2"]
        chain_a = make_chain("int_a", nodes)
        chain_b = make_chain("int_b", nodes)
        score = compute_pathway_overlap(chain_a, chain_b)
        assert score == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self):
        chain_a = make_chain("int_a", ["SOD1", "TDP43"])
        chain_b = make_chain("int_b", ["FUS", "ATXN2"])
        score = compute_pathway_overlap(chain_a, chain_b)
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self):
        chain_a = make_chain("int_a", ["SOD1", "TDP43", "FUS"])
        chain_b = make_chain("int_b", ["TDP43", "FUS", "ATXN2"])
        # intersection = {TDP43, FUS} = 2, union = {SOD1, TDP43, FUS, ATXN2} = 4
        score = compute_pathway_overlap(chain_a, chain_b)
        assert score == pytest.approx(2.0 / 4.0)

    def test_empty_chain_a_returns_zero(self):
        chain_a = make_chain("int_a", [])
        chain_b = make_chain("int_b", ["SOD1"])
        assert compute_pathway_overlap(chain_a, chain_b) == pytest.approx(0.0)

    def test_empty_chain_b_returns_zero(self):
        chain_a = make_chain("int_a", ["SOD1"])
        chain_b = make_chain("int_b", [])
        assert compute_pathway_overlap(chain_a, chain_b) == pytest.approx(0.0)

    def test_both_empty_returns_zero(self):
        chain_a = make_chain("int_a", [])
        chain_b = make_chain("int_b", [])
        assert compute_pathway_overlap(chain_a, chain_b) == pytest.approx(0.0)

    def test_case_insensitive_matching(self):
        chain_a = make_chain("int_a", ["SOD1", "TDP43"])
        chain_b = make_chain("int_b", ["sod1", "tdp43"])
        # Both lowercased — should be identical
        score = compute_pathway_overlap(chain_a, chain_b)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestCombinationAnalysis
# ---------------------------------------------------------------------------

class TestCombinationAnalysis:
    def test_default_creation(self):
        ca = CombinationAnalysis()
        assert ca.flags == []
        assert ca.overall_coherence == pytest.approx(1.0)
        assert ca.suggested_substitutions == []

    def test_flags_set(self):
        flag = InteractionFlag(
            intervention_a="riluzole",
            intervention_b="edaravone",
            interaction_type="synergy",
            mechanism="complementary_oxidative_stress_reduction",
            confidence=0.8,
        )
        ca = CombinationAnalysis(flags=[flag], overall_coherence=0.9)
        assert len(ca.flags) == 1
        assert ca.flags[0].interaction_type == "synergy"
        assert ca.overall_coherence == pytest.approx(0.9)

    def test_interaction_flag_defaults(self):
        flag = InteractionFlag(
            intervention_a="a",
            intervention_b="b",
            interaction_type="antagonism",
            mechanism="competing_pathway",
        )
        assert flag.confidence == pytest.approx(0.0)
        assert flag.cited_evidence == []

    def test_interaction_flag_with_evidence(self):
        flag = InteractionFlag(
            intervention_a="a",
            intervention_b="b",
            interaction_type="redundancy",
            mechanism="same_target",
            confidence=0.75,
            cited_evidence=["pmid:12345", "pmid:67890"],
        )
        assert len(flag.cited_evidence) == 2


# ---------------------------------------------------------------------------
# TestAnalyzeCombinations
# ---------------------------------------------------------------------------

class TestAnalyzeCombinations:
    def test_high_overlap_flagged_as_redundancy(self):
        nodes = ["SOD1", "TDP43", "FUS", "ATXN2"]
        chains = {
            "int_a": make_chain("int_a", nodes),
            "int_b": make_chain("int_b", nodes),
        }
        result = analyze_combinations(chains, overlap_threshold=0.6)
        assert len(result.flags) == 1
        assert result.flags[0].interaction_type == "redundancy"
        assert result.flags[0].intervention_a in ("int_a", "int_b")
        assert result.flags[0].intervention_b in ("int_a", "int_b")

    def test_no_overlap_no_flags(self):
        chains = {
            "int_a": make_chain("int_a", ["SOD1", "TDP43"]),
            "int_b": make_chain("int_b", ["FUS", "ATXN2"]),
        }
        result = analyze_combinations(chains, overlap_threshold=0.6)
        assert len(result.flags) == 0

    def test_coherence_reduced_per_flag(self):
        nodes = ["SOD1", "TDP43", "FUS", "ATXN2"]
        chains = {
            "int_a": make_chain("int_a", nodes),
            "int_b": make_chain("int_b", nodes),
        }
        result = analyze_combinations(chains, overlap_threshold=0.6)
        # 1 flag: 1.0 - 1 * 0.2 = 0.8
        assert result.overall_coherence == pytest.approx(0.8)

    def test_coherence_clamped_at_zero(self):
        # 6 chains all identical → C(6,2) = 15 flags → 1.0 - 15*0.2 = -2.0 → clamped 0.0
        nodes = ["SOD1", "TDP43", "FUS"]
        chains = {f"int_{i}": make_chain(f"int_{i}", nodes) for i in range(6)}
        result = analyze_combinations(chains, overlap_threshold=0.5)
        assert result.overall_coherence == pytest.approx(0.0)

    def test_single_chain_no_pairs(self):
        chains = {"int_a": make_chain("int_a", ["SOD1", "TDP43"])}
        result = analyze_combinations(chains)
        assert result.flags == []
        assert result.overall_coherence == pytest.approx(1.0)

    def test_empty_chains_dict(self):
        result = analyze_combinations({})
        assert result.flags == []
        assert result.overall_coherence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestApplyInteractionFlags
# ---------------------------------------------------------------------------

class TestApplyInteractionFlags:
    def _make_scores(self) -> dict[str, list[dict]]:
        return {
            "layer_1": [
                {"intervention": "drug_a", "score": 0.9},
                {"intervention": "drug_b", "score": 0.6},
            ],
            "layer_2": [
                {"intervention": "drug_c", "score": 0.85},
                {"intervention": "drug_d", "score": 0.7},
            ],
        }

    def test_synergy_flag_no_swap(self):
        scores = self._make_scores()
        flags = [InteractionFlag(
            intervention_a="drug_a",
            intervention_b="drug_c",
            interaction_type="synergy",
            mechanism="complementary",
            confidence=0.9,
        )]
        result = apply_interaction_flags(scores, flags, threshold=0.7)
        # Synergy: nothing removed
        assert len(result["layer_1"]) == 2
        assert len(result["layer_2"]) == 2

    def test_antagonism_removes_lower_scored(self):
        scores = self._make_scores()
        flags = [InteractionFlag(
            intervention_a="drug_a",   # score 0.9
            intervention_b="drug_b",   # score 0.6  ← lower
            interaction_type="antagonism",
            mechanism="competing",
            confidence=0.9,
        )]
        result = apply_interaction_flags(scores, flags, threshold=0.7)
        layer1_interventions = [e["intervention"] for e in result["layer_1"]]
        assert "drug_b" not in layer1_interventions
        assert "drug_a" in layer1_interventions

    def test_redundancy_removes_lower_scored(self):
        scores = self._make_scores()
        flags = [InteractionFlag(
            intervention_a="drug_c",   # score 0.85 (layer_2)
            intervention_b="drug_b",   # score 0.6  (layer_1) ← lower
            interaction_type="redundancy",
            mechanism="same_pathway",
            confidence=0.8,
        )]
        result = apply_interaction_flags(scores, flags, threshold=0.7)
        layer1_interventions = [e["intervention"] for e in result["layer_1"]]
        layer2_interventions = [e["intervention"] for e in result["layer_2"]]
        assert "drug_b" not in layer1_interventions
        assert "drug_c" in layer2_interventions

    def test_low_confidence_flag_no_swap(self):
        scores = self._make_scores()
        flags = [InteractionFlag(
            intervention_a="drug_a",
            intervention_b="drug_b",
            interaction_type="antagonism",
            mechanism="competing",
            confidence=0.5,  # below threshold=0.7
        )]
        result = apply_interaction_flags(scores, flags, threshold=0.7)
        # Confidence too low — no removal
        assert len(result["layer_1"]) == 2

    def test_original_not_mutated(self):
        scores = self._make_scores()
        flags = [InteractionFlag(
            intervention_a="drug_a",
            intervention_b="drug_b",
            interaction_type="antagonism",
            mechanism="competing",
            confidence=0.9,
        )]
        original_len = len(scores["layer_1"])
        apply_interaction_flags(scores, flags, threshold=0.7)
        assert len(scores["layer_1"]) == original_len

    def test_intervention_not_in_scores_ignored(self):
        scores = self._make_scores()
        flags = [InteractionFlag(
            intervention_a="drug_x",
            intervention_b="drug_y",
            interaction_type="antagonism",
            mechanism="competing",
            confidence=0.9,
        )]
        result = apply_interaction_flags(scores, flags, threshold=0.7)
        # Unknown interventions — no change
        assert len(result["layer_1"]) == 2
        assert len(result["layer_2"]) == 2
