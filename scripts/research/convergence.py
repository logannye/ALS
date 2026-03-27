"""Protocol convergence detection.

The protocol has converged when the top intervention per layer is
stable across `window` consecutive regenerations.
"""
from __future__ import annotations
from typing import Optional
from ontology.protocol import CureProtocolCandidate

def get_top_interventions(protocol: CureProtocolCandidate) -> dict[str, Optional[str]]:
    tops: dict[str, Optional[str]] = {}
    for layer_entry in protocol.layers:
        layer_name = layer_entry.layer.value
        if layer_entry.intervention_refs:
            tops[layer_name] = layer_entry.intervention_refs[0]
        else:
            tops[layer_name] = None
    return tops

def is_converged(history: list[CureProtocolCandidate], window: int = 3) -> bool:
    if len(history) < window:
        return False
    recent = history[-window:]
    top_maps = [get_top_interventions(p) for p in recent]
    all_layers = set()
    for tm in top_maps:
        all_layers.update(tm.keys())
    for layer in all_layers:
        values = [tm.get(layer) for tm in top_maps]
        if len(set(values)) > 1:
            return False
    return True
