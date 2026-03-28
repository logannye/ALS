"""Protocol convergence detection.

The protocol has converged when the top intervention per layer is
stable across `window` consecutive regenerations.
"""
from __future__ import annotations
from typing import Optional
from ontology.protocol import CureProtocolCandidate
from research.state import ResearchState

def get_top_interventions(protocol: CureProtocolCandidate) -> dict[str, Optional[str]]:
    tops: dict[str, Optional[str]] = {}
    for layer_entry in protocol.layers:
        layer_name = layer_entry.layer.value
        if layer_entry.intervention_refs:
            tops[layer_name] = layer_entry.intervention_refs[0]
        else:
            tops[layer_name] = None
    return tops

def compute_uncertainty_score(state: ResearchState) -> float:
    """Compute 0-1 uncertainty score from evidence distribution.

    Components (weighted):
    - Layer coverage (50%): proportion of layers with < 10 evidence items
    - Chain depth (30%): proportion of chains below depth 3
    - Missing measurements (20%): proportion unresolved
    """
    layers = state.evidence_by_layer
    if layers:
        sparse = sum(1 for v in layers.values() if v < 10)
        layer_unc = sparse / max(len(layers), 1)
    else:
        layer_unc = 1.0

    chains = state.causal_chains
    if chains:
        shallow = sum(1 for v in chains.values() if v < 3)
        chain_unc = shallow / max(len(chains), 1)
    else:
        chain_unc = 1.0

    missing_unc = len(state.missing_measurements) / 7

    return 0.5 * layer_unc + 0.3 * chain_unc + 0.2 * missing_unc


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
