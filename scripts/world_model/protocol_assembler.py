"""Stage 4 — Assemble a cure protocol from scored interventions.

Selects the top eligible intervention(s) per protocol layer, applies
default timing offsets, and packages everything into a
:class:`~ontology.protocol.CureProtocolCandidate`.

Public API
----------
select_layer_interventions(scores, layer, max_per_layer)
    -> list[InterventionScore]

assemble_protocol(scores, subject_ref)
    -> CureProtocolCandidate
"""
from __future__ import annotations

from ontology.enums import ApprovalState, ProtocolLayer
from ontology.protocol import CureProtocolCandidate, ProtocolLayerEntry
from world_model.intervention_scorer import InterventionScore


# All 5 protocol layers in therapeutic priority order
ALL_LAYERS = [
    ProtocolLayer.root_cause_suppression,
    ProtocolLayer.pathology_reversal,
    ProtocolLayer.circuit_stabilization,
    ProtocolLayer.regeneration_reinnervation,
    ProtocolLayer.adaptive_maintenance,
]

# Default start-offset days per layer: root-cause and stabilization
# start immediately; pathology reversal after 1 week to allow baseline
# labs; regeneration after 3 weeks once core layers are established.
DEFAULT_TIMING: dict[str, int] = {
    "root_cause_suppression": 0,
    "pathology_reversal": 7,
    "circuit_stabilization": 0,
    "regeneration_reinnervation": 21,
    "adaptive_maintenance": 0,
}


# ---------------------------------------------------------------------------
# select_layer_interventions
# ---------------------------------------------------------------------------

def select_layer_interventions(
    scores: list[InterventionScore],
    layer: str,
    max_per_layer: int = 2,
) -> list[InterventionScore]:
    """Return the top eligible interventions for *layer*.

    Filters out interventions where ``erik_eligible is False`` (the literal
    boolean ``False``).  The string ``"pending_genetics"`` is treated as
    eligible so that conditional-arm interventions are retained.

    Parameters
    ----------
    scores:
        All scored interventions (any layer).
    layer:
        The ``ProtocolLayer.value`` string to filter on.
    max_per_layer:
        Maximum interventions to select per layer.

    Returns
    -------
    list[InterventionScore]
        Sorted descending by ``relevance_score``, capped at *max_per_layer*.
    """
    layer_scores = [
        s for s in scores
        if s.protocol_layer == layer and s.erik_eligible is not False
    ]
    layer_scores.sort(key=lambda s: s.relevance_score, reverse=True)
    return layer_scores[:max_per_layer]


# ---------------------------------------------------------------------------
# assemble_protocol
# ---------------------------------------------------------------------------

def assemble_protocol(
    scores: list[InterventionScore],
    subject_ref: str,
) -> CureProtocolCandidate:
    """Assemble a :class:`CureProtocolCandidate` from scored interventions.

    For each of the 5 protocol layers:
    - Select top eligible intervention(s) via :func:`select_layer_interventions`.
    - If none are eligible, emit an ``ABSTENTION`` layer with empty refs.
    - Assign default timing offsets.

    Parameters
    ----------
    scores:
        All scored interventions (output of ``score_all_interventions``).
    subject_ref:
        Patient trajectory identifier (e.g. ``"traj:draper_001"``).

    Returns
    -------
    CureProtocolCandidate
        Protocol with ``approval_state=pending``.
    """
    layers: list[ProtocolLayerEntry] = []
    all_cited: list[str] = []
    all_uncertainties: list[str] = []
    all_failure_modes: list[str] = []

    for protocol_layer in ALL_LAYERS:
        layer_name = protocol_layer.value
        selected = select_layer_interventions(scores, layer_name)

        if not selected:
            layers.append(ProtocolLayerEntry(
                layer=protocol_layer,
                intervention_refs=[],
                start_offset_days=DEFAULT_TIMING.get(layer_name, 0),
                notes=f"ABSTENTION: No eligible interventions scored for {layer_name}",
            ))
            all_failure_modes.append(f"no_intervention_{layer_name}")
        else:
            int_refs = [s.intervention_id for s in selected]
            notes_parts = [
                f"{s.intervention_name} (score={s.relevance_score:.2f})"
                for s in selected
            ]
            layers.append(ProtocolLayerEntry(
                layer=protocol_layer,
                intervention_refs=int_refs,
                start_offset_days=DEFAULT_TIMING.get(layer_name, 0),
                notes="; ".join(notes_parts),
            ))
            for s in selected:
                all_cited.extend(s.cited_evidence)
                all_uncertainties.extend(s.key_uncertainties)
                if s.contested_claims:
                    all_failure_modes.extend(s.contested_claims)

    return CureProtocolCandidate(
        id=f"proto:{subject_ref.split(':')[-1]}_v1",
        subject_ref=subject_ref,
        objective="maximize_durable_disease_arrest_and_functional_recovery",
        layers=layers,
        dominant_failure_modes=list(set(all_failure_modes))[:10],
        approval_state=ApprovalState.pending,
        evidence_bundle_refs=list(set(all_cited)),
        body={
            "all_intervention_scores": [s.model_dump() for s in scores],
            "total_evidence_items_cited": len(set(all_cited)),
            "key_uncertainties": list(set(all_uncertainties))[:20],
        },
    )
