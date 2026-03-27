"""Tests for world_model.intervention_scorer — Stage 3 intervention scoring.

All tests avoid LLM calls (no @pytest.mark.llm).
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# InterventionScore model
# ---------------------------------------------------------------------------

class TestInterventionScoreModel:
    def test_intervention_score_model(self):
        from world_model.intervention_scorer import InterventionScore

        score = InterventionScore(
            intervention_id="int:ril001",
            intervention_name="Riluzole",
            protocol_layer="root_cause_suppression",
            relevance_score=0.75,
            mechanism_argument="Riluzole reduces glutamate excitotoxicity.",
            evidence_strength="moderate",
            erik_eligible=True,
            key_uncertainties=["Long-term tolerability unknown"],
            cited_evidence=["evi:abc123"],
            contested_claims=[],
        )

        assert score.intervention_id == "int:ril001"
        assert score.intervention_name == "Riluzole"
        assert score.relevance_score == 0.75
        assert score.erik_eligible is True
        assert "evi:abc123" in score.cited_evidence

    def test_intervention_score_model_defaults(self):
        from world_model.intervention_scorer import InterventionScore

        score = InterventionScore(
            intervention_id="int:test",
            intervention_name="TestDrug",
            relevance_score=0.5,
            key_uncertainties=[],
            cited_evidence=[],
            contested_claims=[],
        )

        assert score.protocol_layer == ""
        assert score.mechanism_argument == ""
        assert score.evidence_strength == "unknown"
        assert score.erik_eligible is True

    def test_intervention_score_relevance_bounds_valid(self):
        from world_model.intervention_scorer import InterventionScore

        # Boundary values should be accepted
        score_min = InterventionScore(
            intervention_id="int:a",
            intervention_name="A",
            relevance_score=0.0,
            key_uncertainties=[],
            cited_evidence=[],
            contested_claims=[],
        )
        score_max = InterventionScore(
            intervention_id="int:b",
            intervention_name="B",
            relevance_score=1.0,
            key_uncertainties=[],
            cited_evidence=[],
            contested_claims=[],
        )

        assert score_min.relevance_score == 0.0
        assert score_max.relevance_score == 1.0

    def test_intervention_score_relevance_ge_0(self):
        from pydantic import ValidationError
        from world_model.intervention_scorer import InterventionScore

        with pytest.raises(ValidationError):
            InterventionScore(
                intervention_id="int:x",
                intervention_name="X",
                relevance_score=-0.1,
                key_uncertainties=[],
                cited_evidence=[],
                contested_claims=[],
            )

    def test_intervention_score_relevance_le_1(self):
        from pydantic import ValidationError
        from world_model.intervention_scorer import InterventionScore

        with pytest.raises(ValidationError):
            InterventionScore(
                intervention_id="int:x",
                intervention_name="X",
                relevance_score=1.1,
                key_uncertainties=[],
                cited_evidence=[],
                contested_claims=[],
            )


# ---------------------------------------------------------------------------
# _parse_score_response
# ---------------------------------------------------------------------------

class TestParseScoreResponse:
    def _make_response(self, **overrides) -> dict:
        base = {
            "intervention_id": "int:ril001",
            "intervention_name": "Riluzole",
            "protocol_layer": "root_cause_suppression",
            "relevance_score": 0.72,
            "mechanism_argument": "Reduces glutamate excitotoxicity.",
            "evidence_strength": "moderate",
            "erik_eligible": True,
            "key_uncertainties": ["CNS penetrance uncertain"],
            "cited_evidence": ["evi:abc123"],
            "contested_claims": [],
        }
        base.update(overrides)
        return base

    def test_parse_score_response(self):
        from world_model.intervention_scorer import _parse_score_response

        response = self._make_response()
        score = _parse_score_response(response)

        assert isinstance(score.intervention_id, str)
        assert score.intervention_id == "int:ril001"
        assert score.intervention_name == "Riluzole"
        assert score.relevance_score == pytest.approx(0.72)
        assert score.evidence_strength == "moderate"
        assert "evi:abc123" in score.cited_evidence

    def test_parse_score_response_clamps_score(self):
        from world_model.intervention_scorer import _parse_score_response

        response = self._make_response(relevance_score=1.5)
        score = _parse_score_response(response)

        assert score.relevance_score == pytest.approx(1.0)

    def test_parse_score_response_clamps_negative(self):
        from world_model.intervention_scorer import _parse_score_response

        response = self._make_response(relevance_score=-0.3)
        score = _parse_score_response(response)

        assert score.relevance_score == pytest.approx(0.0)

    def test_parse_score_response_maps_all_fields(self):
        from world_model.intervention_scorer import _parse_score_response

        response = self._make_response(
            protocol_layer="circuit_stabilization",
            mechanism_argument="Stabilises motor circuit.",
            erik_eligible=False,
            key_uncertainties=["Dosing unclear", "Off-target effects"],
            contested_claims=["Efficacy in ALS disputed"],
        )
        score = _parse_score_response(response)

        assert score.protocol_layer == "circuit_stabilization"
        assert "Stabilises motor circuit." in score.mechanism_argument
        assert score.erik_eligible is False
        assert len(score.key_uncertainties) == 2
        assert len(score.contested_claims) == 1

    def test_parse_score_response_evidence_strength_values(self):
        from world_model.intervention_scorer import _parse_score_response

        for strength in ("strong", "moderate", "weak", "absent"):
            response = self._make_response(evidence_strength=strength)
            score = _parse_score_response(response)
            assert score.evidence_strength == strength


# ---------------------------------------------------------------------------
# score_intervention
# ---------------------------------------------------------------------------

class TestScoreIntervention:
    def _make_intervention_dict(self) -> dict:
        return {
            "id": "int:ril001",
            "name": "Riluzole",
            "protocol_layer": "root_cause_suppression",
            "targets": ["glutamate_receptor"],
        }

    def _make_valid_response(self) -> dict:
        return {
            "intervention_id": "int:ril001",
            "intervention_name": "Riluzole",
            "protocol_layer": "root_cause_suppression",
            "relevance_score": 0.72,
            "mechanism_argument": "Reduces glutamate.",
            "evidence_strength": "moderate",
            "erik_eligible": True,
            "key_uncertainties": [],
            "cited_evidence": ["evi:abc"],
            "contested_claims": [],
        }

    def test_score_intervention_returns_score_on_success(self):
        from unittest.mock import MagicMock

        from world_model.intervention_scorer import score_intervention, InterventionScore

        engine = MagicMock()
        engine.reason.return_value = self._make_valid_response()

        result = score_intervention(
            intervention=self._make_intervention_dict(),
            evidence_items=[{"id": "evi:abc", "claim": "test"}],
            patient_state_json='{"age": 67}',
            subtype_posterior_json='{"sporadic_tdp43": 1.0}',
            reasoning_engine=engine,
        )

        assert isinstance(result, InterventionScore)
        assert result.relevance_score == pytest.approx(0.72)

    def test_score_intervention_returns_none_on_llm_failure(self):
        from unittest.mock import MagicMock

        from world_model.intervention_scorer import score_intervention

        engine = MagicMock()
        engine.reason.return_value = None

        result = score_intervention(
            intervention=self._make_intervention_dict(),
            evidence_items=[],
            patient_state_json='{"age": 67}',
            subtype_posterior_json='{"sporadic_tdp43": 1.0}',
            reasoning_engine=engine,
        )

        assert result is None

    def test_score_intervention_passes_template(self):
        from unittest.mock import MagicMock

        from world_model.intervention_scorer import score_intervention
        from world_model.prompts.templates import INTERVENTION_SCORING_TEMPLATE

        engine = MagicMock()
        engine.reason.return_value = None

        score_intervention(
            intervention=self._make_intervention_dict(),
            evidence_items=[],
            patient_state_json='{"age": 67}',
            subtype_posterior_json='{}',
            reasoning_engine=engine,
        )

        call_args = engine.reason.call_args
        # template is first positional arg
        assert call_args[0][0] == INTERVENTION_SCORING_TEMPLATE or \
               call_args[1].get("template") == INTERVENTION_SCORING_TEMPLATE


# ---------------------------------------------------------------------------
# score_all_interventions
# ---------------------------------------------------------------------------

class TestScoreAllInterventions:
    def _make_intervention_dict(self, int_id: str, name: str) -> dict:
        return {
            "id": int_id,
            "name": name,
            "protocol_layer": "root_cause_suppression",
            "targets": ["tdp43"],
        }

    def _make_score_response(self, int_id: str, name: str, score: float) -> dict:
        return {
            "intervention_id": int_id,
            "intervention_name": name,
            "protocol_layer": "root_cause_suppression",
            "relevance_score": score,
            "mechanism_argument": f"Mechanism for {name}.",
            "evidence_strength": "moderate",
            "erik_eligible": True,
            "key_uncertainties": [],
            "cited_evidence": [],
            "contested_claims": [],
        }

    def test_score_all_returns_sorted_by_relevance(self):
        from unittest.mock import MagicMock, patch

        from world_model.intervention_scorer import score_all_interventions

        interventions = [
            self._make_intervention_dict("int:a", "DrugA"),
            self._make_intervention_dict("int:b", "DrugB"),
            self._make_intervention_dict("int:c", "DrugC"),
        ]

        scores_map = {
            "int:a": self._make_score_response("int:a", "DrugA", 0.4),
            "int:b": self._make_score_response("int:b", "DrugB", 0.9),
            "int:c": self._make_score_response("int:c", "DrugC", 0.6),
        }

        mock_store = MagicMock()
        mock_store.query_by_intervention_ref.return_value = []
        mock_store.query_by_mechanism_target.return_value = []

        engine = MagicMock()
        engine.reason.side_effect = lambda template, items, extra_context=None, **kw: (
            scores_map.get(
                extra_context.get("intervention_json", "") if extra_context else "",
                None,
            )
            if extra_context else None
        )

        # Patch score_intervention so we control scores directly
        with patch("world_model.intervention_scorer.score_intervention") as mock_score:
            def side_effect(intervention, **kwargs):
                int_id = intervention["id"]
                resp = scores_map[int_id]
                from world_model.intervention_scorer import _parse_score_response
                return _parse_score_response(resp)

            mock_score.side_effect = side_effect

            results = score_all_interventions(
                interventions=interventions,
                evidence_store=mock_store,
                patient_state_json='{"age": 67}',
                subtype_posterior_json='{}',
                reasoning_engine=engine,
            )

        assert len(results) == 3
        # Sorted descending by relevance_score
        assert results[0].relevance_score >= results[1].relevance_score
        assert results[1].relevance_score >= results[2].relevance_score
        assert results[0].intervention_id == "int:b"

    def test_score_all_deduplicates_evidence(self):
        """Evidence items with same ID should not be duplicated."""
        from unittest.mock import MagicMock, patch

        from world_model.intervention_scorer import score_all_interventions

        dup_evidence = [{"id": "evi:shared", "claim": "shared evidence"}]

        mock_store = MagicMock()
        # Both methods return the same evidence item
        mock_store.query_by_intervention_ref.return_value = dup_evidence
        mock_store.query_by_mechanism_target.return_value = dup_evidence

        interventions = [self._make_intervention_dict("int:a", "DrugA")]

        captured_evidence = []

        with patch("world_model.intervention_scorer.score_intervention") as mock_score:
            def side_effect(intervention, evidence_items, **kwargs):
                captured_evidence.extend(evidence_items)
                from world_model.intervention_scorer import _parse_score_response
                return _parse_score_response(
                    self._make_score_response("int:a", "DrugA", 0.5)
                )

            mock_score.side_effect = side_effect

            score_all_interventions(
                interventions=interventions,
                evidence_store=mock_store,
                patient_state_json='{}',
                subtype_posterior_json='{}',
            )

        # Evidence should be deduplicated — "evi:shared" appears only once
        ids = [e["id"] for e in captured_evidence]
        assert ids.count("evi:shared") == 1

    def test_score_all_skips_none_scores(self):
        """Interventions where score_intervention returns None are excluded."""
        from unittest.mock import MagicMock, patch

        from world_model.intervention_scorer import score_all_interventions

        interventions = [
            self._make_intervention_dict("int:good", "GoodDrug"),
            self._make_intervention_dict("int:bad", "BadDrug"),
        ]

        mock_store = MagicMock()
        mock_store.query_by_intervention_ref.return_value = []
        mock_store.query_by_mechanism_target.return_value = []

        with patch("world_model.intervention_scorer.score_intervention") as mock_score:
            def side_effect(intervention, **kwargs):
                if intervention["id"] == "int:good":
                    from world_model.intervention_scorer import _parse_score_response
                    return _parse_score_response(
                        self._make_score_response("int:good", "GoodDrug", 0.8)
                    )
                return None  # LLM failure for "int:bad"

            mock_score.side_effect = side_effect

            results = score_all_interventions(
                interventions=interventions,
                evidence_store=mock_store,
                patient_state_json='{}',
                subtype_posterior_json='{}',
            )

        assert len(results) == 1
        assert results[0].intervention_id == "int:good"

    def test_score_all_verifies_top_5(self):
        """Top 5 by score should have _verify_claim called on mechanism_argument."""
        from unittest.mock import MagicMock, patch

        from world_model.intervention_scorer import score_all_interventions, _parse_score_response

        # 6 interventions: top 5 should be verified
        interventions = [
            self._make_intervention_dict(f"int:{i}", f"Drug{i}")
            for i in range(6)
        ]

        mock_store = MagicMock()
        mock_store.query_by_intervention_ref.return_value = []
        mock_store.query_by_mechanism_target.return_value = []

        engine = MagicMock()
        engine._verify_claim.return_value = {"verdict": "supported"}

        responses = [
            self._make_score_response(f"int:{i}", f"Drug{i}", float(i) / 10.0)
            for i in range(6)
        ]

        with patch("world_model.intervention_scorer.score_intervention") as mock_score:
            idx = [0]

            def side_effect(intervention, **kwargs):
                score = _parse_score_response(responses[int(intervention["id"].split(":")[1])])
                return score

            mock_score.side_effect = side_effect

            results = score_all_interventions(
                interventions=interventions,
                evidence_store=mock_store,
                patient_state_json='{}',
                subtype_posterior_json='{}',
                reasoning_engine=engine,
            )

        # _verify_claim should have been called 5 times (top 5)
        assert engine._verify_claim.call_count == 5

    def test_score_all_creates_engine_when_none(self):
        """When reasoning_engine is None, a lazy engine is created."""
        from unittest.mock import MagicMock, patch

        from world_model.intervention_scorer import score_all_interventions

        mock_store = MagicMock()
        mock_store.query_by_intervention_ref.return_value = []
        mock_store.query_by_mechanism_target.return_value = []

        mock_engine = MagicMock()
        mock_engine._verify_claim.return_value = None

        with patch("world_model.intervention_scorer.ReasoningEngine", return_value=mock_engine) as MockEngine:
            with patch("world_model.intervention_scorer.score_intervention", return_value=None):
                score_all_interventions(
                    interventions=[self._make_intervention_dict("int:x", "X")],
                    evidence_store=mock_store,
                    patient_state_json='{}',
                    subtype_posterior_json='{}',
                    reasoning_engine=None,
                )

        MockEngine.assert_called_once_with(lazy=True)
