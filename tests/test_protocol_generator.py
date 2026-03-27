"""Tests for Stage 6 — Full pipeline orchestrator."""
from __future__ import annotations

import pytest

from world_model.protocol_generator import generate_cure_protocol


class TestGenerateCureProtocolNoLLM:
    """Tests that run without the LLM (deterministic stages only)."""

    def test_returns_dict(self):
        result = generate_cure_protocol(use_llm=False)
        assert isinstance(result, dict)

    def test_snapshot_present(self):
        result = generate_cure_protocol(use_llm=False)
        assert result["snapshot"] is not None
        assert result["snapshot"].type == "DiseaseStateSnapshot"

    def test_snapshot_has_functional_state(self):
        result = generate_cure_protocol(use_llm=False)
        assert result["snapshot"].functional_state_ref is not None

    def test_snapshot_has_uncertainty(self):
        result = generate_cure_protocol(use_llm=False)
        assert result["snapshot"].uncertainty_ref is not None

    def test_patient_present(self):
        result = generate_cure_protocol(use_llm=False)
        assert result["patient"] is not None
        assert result["trajectory"] is not None

    def test_no_protocol_without_llm(self):
        """Without LLM, stages 2-6 are skipped."""
        result = generate_cure_protocol(use_llm=False)
        assert result["protocol"] is None
        assert result["subtype_profile"] is None
        assert result["intervention_scores"] == []
        assert result["counterfactuals"] == []


@pytest.mark.llm
class TestGenerateCureProtocolWithLLM:
    """Integration tests — require a local LLM model on disk."""

    def test_full_pipeline_produces_protocol(self):
        result = generate_cure_protocol(use_llm=True)
        assert result is not None
        assert result["protocol"] is not None
        assert result["protocol"].type == "CureProtocolCandidate"

    def test_protocol_has_5_layers(self):
        result = generate_cure_protocol(use_llm=True)
        assert len(result["protocol"].layers) == 5

    def test_protocol_is_pending(self):
        result = generate_cure_protocol(use_llm=True)
        assert result["protocol"].approval_state.value == "pending"

    def test_has_evidence_citations(self):
        result = generate_cure_protocol(use_llm=True)
        assert len(result["protocol"].evidence_bundle_refs) > 0

    def test_has_counterfactuals(self):
        result = generate_cure_protocol(use_llm=True)
        assert len(result["counterfactuals"]) == 5
