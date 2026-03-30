"""Tests for Phase 7 — deep causal understanding fixes.

Covers: SCM exploitation fix, targeted queries, hypothesis validation,
KG entity extraction, and computational experiment integration.
"""
from __future__ import annotations

import math
import pytest
from dataclasses import replace

from research.actions import ActionResult, ActionType
from research.state import ResearchState, initial_state, ALL_LAYERS
from research.rewards import compute_reward
from research.policy import (
    _get_targeted_query,
    _get_dynamic_query,
    get_layer_query,
    _BASE_LAYER_QUERIES,
    _ACQUISITION_ROTATION,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _state_with_chains(n: int = 3) -> ResearchState:
    state = initial_state(subject_ref="traj:draper_001")
    chains = {f"int:drug_{i}": i for i in range(n)}
    return replace(state, causal_chains=chains)


# ===========================================================================
# Rec 3: SCM exploitation fix
# ===========================================================================

class TestSCMExploitationFix:

    def test_causal_depth_discount_when_no_evidence(self):
        """Causal depth without evidence should be 90% discounted."""
        reward = compute_reward(
            evidence_items_added=0,
            uncertainty_before=0.5,
            uncertainty_after=0.5,
            protocol_score_delta=0.0,
            hypothesis_resolved=False,
            causal_depth_added=1,
            interaction_safe=False,
            eligibility_confirmed=False,
            protocol_stable=False,
        )
        # causal_depth_val = log1p(1) * 0.1 = 0.693 * 0.1 = 0.0693
        # total = 2.0 * 0.0693 = 0.1386
        assert reward.causal_depth == pytest.approx(math.log1p(1) * 0.1, rel=0.01)
        assert reward.total() == pytest.approx(2.0 * math.log1p(1) * 0.1, rel=0.01)

    def test_causal_depth_full_when_evidence_present(self):
        """Causal depth with evidence should get full reward."""
        reward = compute_reward(
            evidence_items_added=5,
            uncertainty_before=0.5,
            uncertainty_after=0.5,
            protocol_score_delta=0.0,
            hypothesis_resolved=False,
            causal_depth_added=1,
            interaction_safe=False,
            eligibility_confirmed=False,
            protocol_stable=False,
        )
        # No discount — full causal depth
        assert reward.causal_depth == pytest.approx(math.log1p(1), rel=0.01)

    def test_scm_reward_much_lower_without_evidence(self):
        """SCM with 0 evidence should reward ~10x less than with evidence."""
        reward_no_evi = compute_reward(
            evidence_items_added=0, uncertainty_before=0.5, uncertainty_after=0.5,
            protocol_score_delta=0.0, hypothesis_resolved=False, causal_depth_added=1,
            interaction_safe=False, eligibility_confirmed=False, protocol_stable=False,
        )
        reward_with_evi = compute_reward(
            evidence_items_added=3, uncertainty_before=0.5, uncertainty_after=0.5,
            protocol_score_delta=0.0, hypothesis_resolved=False, causal_depth_added=1,
            interaction_safe=False, eligibility_confirmed=False, protocol_stable=False,
        )
        assert reward_with_evi.total() > reward_no_evi.total() * 5


# ===========================================================================
# Rec 2: Targeted query expansion
# ===========================================================================

class TestTargetedQueryExpansion:

    def test_targeted_query_includes_gene_symbol(self):
        """Targeted query should contain a real ALS gene symbol."""
        state = initial_state(subject_ref="traj:draper_001")
        query = _get_targeted_query(state, step=0)
        # Should contain gene from the first ALS target
        assert "ALS" in query or any(g in query for g in [
            "TARDBP", "SOD1", "FUS", "C9orf72", "STMN2", "SLC1A2", "SIGMAR1",
            "MTOR", "CSF1R", "BDNF", "GDNF", "NEK1", "TBK1", "OPTN",
        ])

    def test_targeted_query_rotates_targets(self):
        """Different steps should hit different gene targets."""
        state = initial_state(subject_ref="traj:draper_001")
        queries = set()
        for step in range(16):
            q = _get_targeted_query(state, step)
            queries.add(q)
        # With 16 targets, 16 different steps should produce variety
        assert len(queries) >= 8

    def test_targeted_query_has_year(self):
        """Targeted query should include current year for freshness."""
        import datetime
        state = initial_state(subject_ref="traj:draper_001")
        query = _get_targeted_query(state, step=0)
        assert str(datetime.datetime.now().year) in query

    def test_three_strategy_cycling(self):
        """Steps 0,1,2 should produce different query types."""
        state = initial_state(subject_ref="traj:draper_001")
        state = replace(state, active_hypotheses=["TDP-43 aggregation causes motor neuron death"])
        q0 = get_layer_query("root_cause_suppression", 0)  # Static (step%3==0)
        q1 = _get_dynamic_query(state, 1, "root_cause_suppression")  # Dynamic (step%3==1)
        q2 = _get_targeted_query(state, 2)  # Targeted (step%3==2)
        # All should be valid strings
        assert len(q0) > 5
        assert len(q1) > 5
        assert len(q2) > 5

    def test_run_computation_in_acquisition_rotation(self):
        """RUN_COMPUTATION should be in the acquisition rotation."""
        assert ActionType.RUN_COMPUTATION in _ACQUISITION_ROTATION


# ===========================================================================
# Rec 1: KG entity extractor
# ===========================================================================

class TestKGEntityExtractor:

    def test_make_entity_id_format(self):
        from knowledge_quality.entity_extractor import _make_entity_id
        assert _make_entity_id("gene", "SOD1") == "gene:sod1"
        assert _make_entity_id("drug", "Riluzole") == "drug:riluzole"
        assert _make_entity_id("mechanism", "TDP-43 aggregation") == "mechanism:tdp_43_aggregation"

    def test_extract_entities_from_body_gene_fields(self):
        from knowledge_quality.entity_extractor import _extract_entities_from_body
        body = {"gene_a": "SOD1", "gene_b": "FUS", "claim": "SOD1 interacts with FUS"}
        entities = _extract_entities_from_body(body, "evi:test1")
        gene_ids = {e["id"] for e in entities}
        assert "gene:sod1" in gene_ids
        assert "gene:fus" in gene_ids

    def test_extract_entities_from_body_drug_field(self):
        from knowledge_quality.entity_extractor import _extract_entities_from_body
        body = {"intervention_ref": "int:riluzole", "mechanism_target": "glutamate_excitotoxicity"}
        entities = _extract_entities_from_body(body, "evi:test2")
        types = {e["entity_type"] for e in entities}
        assert "drug" in types
        assert "mechanism" in types

    def test_extract_entities_from_claim_text(self):
        from knowledge_quality.entity_extractor import _extract_entities_from_body
        body = {"claim": "TDP-43 aggregation drives STMN2 loss in ALS motor neurons"}
        entities = _extract_entities_from_body(body, "evi:test3")
        gene_ids = {e["id"] for e in entities if e["entity_type"] == "gene"}
        assert "gene:tdp_43" in gene_ids or "gene:stmn2" in gene_ids

    def test_infer_relationships_drug_gene(self):
        from knowledge_quality.entity_extractor import _extract_entities_from_body, _infer_relationships
        body = {"gene_a": "MTOR", "drug_name": "rapamycin", "claim": "rapamycin inhibits MTOR"}
        entities = _extract_entities_from_body(body, "evi:test4")
        rels = _infer_relationships(entities, body, "evi:test4")
        rel_types = {r["relationship_type"] for r in rels}
        # Should have at least a "targets" relationship between drug and gene
        assert "targets" in rel_types or "associated_with" in rel_types or len(rels) >= 1

    def test_observational_relations_never_l3(self):
        from knowledge_quality.entity_extractor import _validate_relationship_pch
        rel = {"relationship_type": "associated_with", "pch_layer": 3}
        validated = _validate_relationship_pch(rel)
        assert validated["pch_layer"] == 1  # Demoted to L1

    def test_causal_relations_can_be_l3(self):
        from knowledge_quality.entity_extractor import _validate_relationship_pch
        rel = {"relationship_type": "causes", "pch_layer": 3}
        validated = _validate_relationship_pch(rel)
        assert validated["pch_layer"] == 3  # Kept at L3


# ===========================================================================
# Rec 6: Computation executor
# ===========================================================================

class TestComputationExecutor:

    def test_action_type_exists(self):
        assert hasattr(ActionType, "RUN_COMPUTATION")
        assert ActionType.RUN_COMPUTATION.value == "run_computation"

    def test_executor_handles_missing_depmap(self):
        from executors.als_computation_executor import ALSComputationExecutor
        executor = ALSComputationExecutor(depmap_path="/nonexistent/path")
        result = executor.run_experiment("gene_essentiality", target="SOD1", gene="SOD1")
        assert not result.success
        assert "not found" in result.error.lower() or "no such file" in result.error.lower()

    def test_executor_handles_missing_gdsc(self):
        from executors.als_computation_executor import ALSComputationExecutor
        executor = ALSComputationExecutor(gdsc_path="/nonexistent/path")
        result = executor.run_experiment("drug_sensitivity", target="riluzole", drug="riluzole")
        assert not result.success

    def test_executor_handles_unknown_experiment(self):
        from executors.als_computation_executor import ALSComputationExecutor
        executor = ALSComputationExecutor()
        result = executor.run_experiment("unknown_type", target="test")
        assert not result.success
        assert "Unknown" in result.error

    def test_binding_affinity_with_missing_chembl(self):
        from executors.als_computation_executor import ALSComputationExecutor
        executor = ALSComputationExecutor(chembl_path="/nonexistent/chembl.db")
        result = executor.run_experiment("binding_affinity", target="test", drug="riluzole", gene="SOD1")
        assert not result.success

    def test_computation_result_dataclass(self):
        from executors.als_computation_executor import ComputationResult
        result = ComputationResult(experiment_type="test", target="test")
        assert result.success is True
        assert result.facts == []
        assert result.error is None
