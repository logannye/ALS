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
