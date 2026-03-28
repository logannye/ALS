"""Tests for protocol convergence detection."""
from __future__ import annotations
import pytest
from ontology.enums import ApprovalState, ProtocolLayer
from ontology.protocol import CureProtocolCandidate, ProtocolLayerEntry
from research.convergence import is_converged, get_top_interventions

def _protocol(layer_interventions: dict[str, list[str]], proto_id: str = "proto:v1") -> CureProtocolCandidate:
    layers = []
    for layer_val in [l.value for l in ProtocolLayer]:
        int_refs = layer_interventions.get(layer_val, [])
        layers.append(ProtocolLayerEntry(
            layer=ProtocolLayer(layer_val),
            intervention_refs=int_refs,
            notes="ABSTENTION" if not int_refs else "ok",
        ))
    return CureProtocolCandidate(
        id=proto_id, subject_ref="traj:test", objective="test",
        layers=layers, approval_state=ApprovalState.pending,
    )

class TestGetTopInterventions:
    def test_extracts_tops(self):
        proto = _protocol({"root_cause_suppression": ["int:vtx002"], "pathology_reversal": ["int:pridopidine"], "circuit_stabilization": ["int:riluzole"]})
        tops = get_top_interventions(proto)
        assert tops["root_cause_suppression"] == "int:vtx002"
        assert tops["pathology_reversal"] == "int:pridopidine"

    def test_empty_layer_returns_none(self):
        proto = _protocol({})
        tops = get_top_interventions(proto)
        assert tops["root_cause_suppression"] is None

class TestIsConverged:
    def test_not_converged_too_few(self):
        history = [_protocol({"root_cause_suppression": ["int:a"]})]
        assert is_converged(history, window=3) is False

    def test_converged_when_stable(self):
        same = {"root_cause_suppression": ["int:vtx002"], "pathology_reversal": ["int:pridopidine"]}
        history = [_protocol(same, f"proto:v{i}") for i in range(3)]
        assert is_converged(history, window=3) is True

    def test_not_converged_when_changing(self):
        history = [
            _protocol({"root_cause_suppression": ["int:vtx002"]}, "proto:v1"),
            _protocol({"root_cause_suppression": ["int:tofersen"]}, "proto:v2"),
            _protocol({"root_cause_suppression": ["int:vtx002"]}, "proto:v3"),
        ]
        assert is_converged(history, window=3) is False

    def test_converged_ignores_abstained_layers(self):
        history = [_protocol({"root_cause_suppression": ["int:vtx002"]}, f"proto:v{i}") for i in range(3)]
        assert is_converged(history, window=3) is True


class TestUncertaintyScore:
    def test_returns_float_between_0_and_1(self):
        from research.convergence import compute_uncertainty_score
        from research.state import initial_state
        state = initial_state(subject_ref="traj:draper_001")
        score = compute_uncertainty_score(state)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_initial_state_high_uncertainty(self):
        from research.convergence import compute_uncertainty_score
        from research.state import initial_state
        state = initial_state(subject_ref="traj:draper_001")
        score = compute_uncertainty_score(state)
        assert score > 0.8

    def test_decreases_with_evidence(self):
        from research.convergence import compute_uncertainty_score
        from research.state import initial_state
        state = initial_state(subject_ref="traj:draper_001")
        score_before = compute_uncertainty_score(state)
        state.evidence_by_layer["root_cause_suppression"] = 15
        state.evidence_by_layer["pathology_reversal"] = 12
        score_after = compute_uncertainty_score(state)
        assert score_after < score_before

    def test_full_evidence_low_uncertainty(self):
        from research.convergence import compute_uncertainty_score
        from research.state import initial_state
        state = initial_state(subject_ref="traj:draper_001")
        for layer in state.evidence_by_layer:
            state.evidence_by_layer[layer] = 30
        state.causal_chains = {"int:a": 5, "int:b": 5}
        score = compute_uncertainty_score(state)
        assert score < 0.3
