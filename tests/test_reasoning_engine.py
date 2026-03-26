"""Tests for world_model.reasoning_engine.

Unit tests do NOT require a local LLM model.
Real LLM tests are marked @pytest.mark.llm.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# validate_citations
# ---------------------------------------------------------------------------

class TestValidateCitations:
    def test_validate_citations_all_valid(self):
        from world_model.reasoning_engine import validate_citations

        output = {
            "reasoning": "TDP-43 mislocalization is supported by evi:abc123.",
            "cited_evidence": ["evi:abc123", "int:xyz789"],
        }
        valid_ids = {"evi:abc123", "int:xyz789", "evi:other"}
        cleaned, warnings = validate_citations(output, valid_ids)

        assert cleaned["cited_evidence"] == ["evi:abc123", "int:xyz789"]
        assert warnings == []

    def test_validate_citations_catches_hallucinated(self):
        from world_model.reasoning_engine import validate_citations

        output = {
            "cited_evidence": ["evi:real", "evi:FAKE_HALLUCINATED"],
        }
        valid_ids = {"evi:real"}
        cleaned, warnings = validate_citations(output, valid_ids)

        assert "evi:FAKE_HALLUCINATED" not in cleaned["cited_evidence"]
        assert "evi:real" in cleaned["cited_evidence"]
        assert len(warnings) == 1
        assert "evi:FAKE_HALLUCINATED" in warnings[0]

    def test_validate_citations_empty_input(self):
        from world_model.reasoning_engine import validate_citations

        output = {"reasoning": "no citations here"}
        valid_ids = {"evi:abc"}
        cleaned, warnings = validate_citations(output, valid_ids)

        assert cleaned["cited_evidence"] == []
        assert warnings == []

    def test_validate_citations_all_hallucinated(self):
        from world_model.reasoning_engine import validate_citations

        output = {"cited_evidence": ["evi:fake1", "evi:fake2"]}
        valid_ids = {"evi:real"}
        cleaned, warnings = validate_citations(output, valid_ids)

        assert cleaned["cited_evidence"] == []
        assert len(warnings) == 2

    def test_validate_citations_preserves_other_fields(self):
        from world_model.reasoning_engine import validate_citations

        output = {
            "overall_score": 0.75,
            "reasoning": "some reasoning",
            "cited_evidence": ["evi:abc"],
        }
        valid_ids = {"evi:abc"}
        cleaned, warnings = validate_citations(output, valid_ids)

        assert cleaned["overall_score"] == 0.75
        assert cleaned["reasoning"] == "some reasoning"


# ---------------------------------------------------------------------------
# strip_uncited_claims
# ---------------------------------------------------------------------------

class TestStripUncitedClaims:
    def test_strip_uncited_claims_removes_text(self):
        from world_model.reasoning_engine import strip_uncited_claims

        text = (
            "This is an unsupported long claim about ALS that has no citation. "
            "TDP-43 mislocalization is confirmed [evi:abc123]."
        )
        valid_ids = {"evi:abc123"}
        result = strip_uncited_claims(text, valid_ids)

        # The cited sentence should be kept
        assert "evi:abc123" in result
        # The unsupported long sentence should be gone
        assert "unsupported long claim" not in result

    def test_strip_uncited_claims_keeps_cited(self):
        from world_model.reasoning_engine import strip_uncited_claims

        text = (
            "Evidence shows TDP-43 aggregation [evi:x1]. "
            "Motor neuron loss is confirmed [evi:x2]."
        )
        valid_ids = {"evi:x1", "evi:x2"}
        result = strip_uncited_claims(text, valid_ids)

        assert "evi:x1" in result
        assert "evi:x2" in result

    def test_strip_uncited_claims_keeps_short_structural(self):
        from world_model.reasoning_engine import strip_uncited_claims

        # Short sentences (< 8 words) are kept even without citations
        text = "Overall. This is the summary. TDP-43 mislocalization is confirmed by study xyz [evi:abc]."
        valid_ids = {"evi:abc"}
        result = strip_uncited_claims(text, valid_ids)

        # Short sentences kept
        assert "Overall" in result or "summary" in result

    def test_strip_uncited_claims_empty_text(self):
        from world_model.reasoning_engine import strip_uncited_claims

        result = strip_uncited_claims("", {"evi:abc"})
        assert result == ""

    def test_strip_uncited_claims_int_reference(self):
        from world_model.reasoning_engine import strip_uncited_claims

        text = "The intervention suppressed TDP-43 aggregation by 40% [int:ril001]."
        valid_ids = {"int:ril001"}
        result = strip_uncited_claims(text, valid_ids)

        assert "int:ril001" in result


# ---------------------------------------------------------------------------
# ReasoningEngine
# ---------------------------------------------------------------------------

class TestReasoningEngineInstantiation:
    def test_reasoning_engine_instantiates_lazy(self):
        from world_model.reasoning_engine import ReasoningEngine

        engine = ReasoningEngine(lazy=True)
        assert engine is not None
        # LLM should NOT be loaded yet
        assert engine._llm._model is None

    def test_reasoning_engine_custom_model_path(self):
        from world_model.reasoning_engine import ReasoningEngine

        engine = ReasoningEngine(lazy=True, model_path="/tmp/fake-model")
        assert engine._llm.model_path == "/tmp/fake-model"


class TestReasoningEngineBuildPrompt:
    def test_reasoning_engine_builds_prompt(self):
        """_build_prompt must embed evidence items in the rendered template."""
        from world_model.reasoning_engine import ReasoningEngine
        from world_model.prompts.templates import REVERSIBILITY_TEMPLATE

        engine = ReasoningEngine(lazy=True)

        # Minimal evidence-like dicts
        evidence_items = [
            {"id": "evi:abc123", "summary": "TDP-43 aggregation in motor neurons"},
            {"id": "evi:def456", "summary": "STMN2 cryptic exon inclusion observed"},
        ]

        prompt = engine._build_prompt(
            template=REVERSIBILITY_TEMPLATE,
            evidence_items=evidence_items,
            extra_context={"patient_state_json": '{"age": 67}'},
        )

        # Evidence IDs must appear in the rendered prompt
        assert "evi:abc123" in prompt
        assert "evi:def456" in prompt
        # Template body text should appear
        assert "reversibility" in prompt.lower()

    def test_build_prompt_includes_system_prompt(self):
        """reason() must prepend SYSTEM_PROMPT."""
        from world_model.reasoning_engine import ReasoningEngine
        from world_model.prompts.templates import REVERSIBILITY_TEMPLATE, SYSTEM_PROMPT

        engine = ReasoningEngine(lazy=True)

        evidence_items = [{"id": "evi:x", "summary": "test"}]
        prompt = engine._build_prompt(
            template=REVERSIBILITY_TEMPLATE,
            evidence_items=evidence_items,
            extra_context={"patient_state_json": "{}"},
        )

        assert SYSTEM_PROMPT in prompt


# ---------------------------------------------------------------------------
# Real LLM tests — require local model
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_reasoning_engine_reason_returns_dict():
    from world_model.reasoning_engine import ReasoningEngine
    from world_model.prompts.templates import REVERSIBILITY_TEMPLATE

    engine = ReasoningEngine()
    evidence_items = [
        {
            "id": "evi:test001",
            "summary": "TDP-43 aggregation linked to motor neuron loss in ALS.",
        }
    ]
    result = engine.reason(
        template=REVERSIBILITY_TEMPLATE,
        evidence_items=evidence_items,
        extra_context={"patient_state_json": '{"age": 67, "diagnosis": "ALS"}'},
        max_tokens=500,
    )
    # May be None if LLM fails to produce valid JSON, but if it does succeed:
    if result is not None:
        assert isinstance(result, dict)
        assert "cited_evidence" in result
