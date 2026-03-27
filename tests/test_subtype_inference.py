"""Tests for world_model.subtype_inference — Stage 2 subtype inference.

All tests avoid LLM calls (no @pytest.mark.llm).
"""
from __future__ import annotations

import pytest

from ontology.enums import SubtypeClass


# ---------------------------------------------------------------------------
# _parse_subtype_response
# ---------------------------------------------------------------------------

class TestParseSubtypeResponse:
    def _make_response(self, posterior: dict | None = None) -> dict:
        """Build a minimal valid response dict."""
        if posterior is None:
            posterior = {
                "sporadic_tdp43": 0.6,
                "sod1": 0.2,
                "unresolved": 0.2,
            }
        return {
            "posterior": posterior,
            "conditional_on_genetics": "No known pathogenic variants; sporadic presentation assumed.",
            "reasoning": "Patient lacks SOD1/C9orf72/FUS variants [evi:test001].",
            "cited_evidence": ["evi:test001"],
        }

    def test_parse_subtype_response_maps_strings_to_enum(self):
        from world_model.subtype_inference import _parse_subtype_response

        response = self._make_response(
            posterior={"sporadic_tdp43": 0.7, "sod1": 0.2, "unresolved": 0.1}
        )
        profile = _parse_subtype_response(response, subject_ref="patient:erik_001")

        # All keys must be SubtypeClass enum instances
        for key in profile.posterior.keys():
            assert isinstance(key, SubtypeClass), f"Expected SubtypeClass, got {type(key)}"

    def test_parse_subtype_response_normalizes(self):
        from world_model.subtype_inference import _parse_subtype_response

        # Posterior sums to 0.6, not 1.0 — must be normalized
        response = self._make_response(
            posterior={"sporadic_tdp43": 0.3, "sod1": 0.2, "unresolved": 0.1}
        )
        profile = _parse_subtype_response(response, subject_ref="patient:erik_001")

        total = sum(profile.posterior.values())
        assert abs(total - 1.0) < 1e-6, f"Expected sum ~1.0, got {total}"

    def test_parse_subtype_response_dominant_subtype(self):
        from world_model.subtype_inference import _parse_subtype_response

        response = self._make_response(
            posterior={
                "sporadic_tdp43": 0.65,
                "sod1": 0.15,
                "c9orf72": 0.10,
                "unresolved": 0.10,
            }
        )
        profile = _parse_subtype_response(response, subject_ref="patient:erik_001")

        assert profile.dominant_subtype == SubtypeClass.sporadic_tdp43

    def test_parse_subtype_response_conditional_genetics(self):
        from world_model.subtype_inference import _parse_subtype_response

        response = self._make_response()
        response["conditional_on_genetics"] = "No pathogenic variants found; sporadic."
        profile = _parse_subtype_response(response, subject_ref="patient:erik_001")

        assert "conditional_on_genetics" in profile.body
        assert profile.body["conditional_on_genetics"] == "No pathogenic variants found; sporadic."

    def test_parse_subtype_response_body_contains_reasoning(self):
        from world_model.subtype_inference import _parse_subtype_response

        response = self._make_response()
        profile = _parse_subtype_response(response, subject_ref="patient:erik_001")

        assert "reasoning" in profile.body

    def test_parse_subtype_response_id(self):
        from world_model.subtype_inference import _parse_subtype_response

        response = self._make_response()
        profile = _parse_subtype_response(response, subject_ref="patient:erik_001")

        assert profile.id == "driver:patient:erik_001"

    def test_parse_subtype_response_supporting_evidence_refs(self):
        from world_model.subtype_inference import _parse_subtype_response

        response = self._make_response()
        response["cited_evidence"] = ["evi:abc", "evi:def"]
        profile = _parse_subtype_response(response, subject_ref="patient:erik_001")

        assert "evi:abc" in profile.supporting_evidence_refs
        assert "evi:def" in profile.supporting_evidence_refs

    def test_parse_subtype_response_skips_invalid_keys(self):
        from world_model.subtype_inference import _parse_subtype_response

        response = self._make_response(
            posterior={
                "sporadic_tdp43": 0.5,
                "NOT_A_REAL_SUBTYPE": 0.3,
                "sod1": 0.2,
            }
        )
        profile = _parse_subtype_response(response, subject_ref="patient:erik_001")

        # Invalid key must not appear
        for key in profile.posterior.keys():
            assert key != "NOT_A_REAL_SUBTYPE"
        # Valid keys present
        assert SubtypeClass.sporadic_tdp43 in profile.posterior
        assert SubtypeClass.sod1 in profile.posterior

    def test_parse_subtype_response_subject_ref(self):
        from world_model.subtype_inference import _parse_subtype_response

        response = self._make_response()
        profile = _parse_subtype_response(response, subject_ref="patient:erik_001")

        assert profile.subject_ref == "patient:erik_001"


# ---------------------------------------------------------------------------
# infer_subtype
# ---------------------------------------------------------------------------

class TestInferSubtype:
    def test_infer_subtype_returns_uniform_prior_on_none_response(self):
        """When reasoning_engine.reason() returns None, return uniform prior."""
        from unittest.mock import MagicMock

        from world_model.subtype_inference import infer_subtype

        engine = MagicMock()
        engine.reason.return_value = None

        profile = infer_subtype(
            patient_state_json='{"age": 67}',
            evidence_items=[],
            subject_ref="patient:erik_001",
            reasoning_engine=engine,
        )

        # Uniform prior: all 8 subtypes present, each = 1/8
        assert len(profile.posterior) == len(SubtypeClass)
        total = sum(profile.posterior.values())
        assert abs(total - 1.0) < 1e-6
        for val in profile.posterior.values():
            assert abs(val - 1.0 / len(SubtypeClass)) < 1e-6

    def test_infer_subtype_returns_profile_on_valid_response(self):
        """When reasoning_engine.reason() returns a valid dict, parse it."""
        from unittest.mock import MagicMock

        from world_model.subtype_inference import infer_subtype

        engine = MagicMock()
        engine.reason.return_value = {
            "posterior": {"sporadic_tdp43": 0.8, "sod1": 0.2},
            "conditional_on_genetics": "No variants found.",
            "reasoning": "Sporadic presentation.",
            "cited_evidence": ["evi:test001"],
        }

        profile = infer_subtype(
            patient_state_json='{"age": 67}',
            evidence_items=[{"id": "evi:test001", "claim": "some claim"}],
            subject_ref="patient:erik_001",
            reasoning_engine=engine,
        )

        assert profile.dominant_subtype == SubtypeClass.sporadic_tdp43

    def test_infer_subtype_calls_reason_with_subtype_template(self):
        """reason() must be called with SUBTYPE_TEMPLATE."""
        from unittest.mock import MagicMock

        from world_model.subtype_inference import infer_subtype
        from world_model.prompts.templates import SUBTYPE_TEMPLATE

        engine = MagicMock()
        engine.reason.return_value = None

        infer_subtype(
            patient_state_json='{"age": 67}',
            evidence_items=[],
            subject_ref="patient:erik_001",
            reasoning_engine=engine,
        )

        # The template must be passed as the first positional or keyword argument
        call_args = engine.reason.call_args
        assert call_args is not None
        # template is the first positional arg
        assert call_args[0][0] == SUBTYPE_TEMPLATE or call_args[1].get("template") == SUBTYPE_TEMPLATE

    def test_infer_subtype_creates_engine_when_none(self):
        """When reasoning_engine is None, engine is created with lazy=True."""
        from unittest.mock import MagicMock, patch

        from world_model.subtype_inference import infer_subtype

        mock_engine = MagicMock()
        mock_engine.reason.return_value = None

        with patch("world_model.subtype_inference.ReasoningEngine", return_value=mock_engine) as MockEngine:
            infer_subtype(
                patient_state_json='{}',
                evidence_items=[],
                subject_ref="patient:test",
                reasoning_engine=None,
            )

        MockEngine.assert_called_once_with(lazy=True)
