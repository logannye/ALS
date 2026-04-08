"""Layer orchestrator — determines the current research phase.

The research engine progresses through four layers, each building on
the previous one:

  Layer 1 (Normal Biology):    Model healthy motor neuron function
  Layer 2 (ALS Mechanisms):    Map how ALS disrupts normal biology
  Layer 3 (Erik's Case):       Narrow to Erik's specific pathways
                                (requires genetic testing results, or
                                 provisional inference when enabled)
  Layer 4 (Drug Design):       Design/identify molecules targeting
                                Erik's validated causal targets

Transitions are gated:
  1 → 2:  Evidence count >= LAYER_1_THRESHOLD (basic biology mapped)
  2 → 3:  Genetic profile uploaded (non-None), OR
           provisional_genetics_enabled=True with evidence >= provisional_genetics_min_evidence
  3 → 4:  Validated causal targets >= LAYER_3_TARGET_THRESHOLD
           AND confirmed genetics (never on provisional profile alone)
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class ResearchLayer(Enum):
    NORMAL_BIOLOGY = "normal_biology"
    ALS_MECHANISMS = "als_mechanisms"
    ERIK_SPECIFIC = "erik_specific"
    DRUG_DESIGN = "drug_design"


# Evidence count thresholds for layer transitions
LAYER_1_THRESHOLD = 100  # Advance to Layer 2 after basic biology foundation
LAYER_3_TARGET_THRESHOLD = 2  # Advance to Layer 4 after validating causal targets

# Provisional genetics defaults (overridden by config when passed to determine_layer)
_DEFAULT_PROVISIONAL_ENABLED = False
_DEFAULT_PROVISIONAL_MIN_EVIDENCE = 500


def determine_layer(
    evidence_count: int,
    genetic_profile: Optional[dict[str, Any]],
    validated_targets: int,
    *,
    provisional_genetics_enabled: bool = _DEFAULT_PROVISIONAL_ENABLED,
    provisional_genetics_min_evidence: int = _DEFAULT_PROVISIONAL_MIN_EVIDENCE,
) -> ResearchLayer:
    """Determine the current research layer from state signals.

    Args:
        evidence_count: Total evidence items in the knowledge graph.
        genetic_profile: Erik's genetic testing results (None if not yet received).
            Expected keys: gene, variant, subtype.
        validated_targets: Number of causal targets with L2+ evidence linking
            them to Erik's specific disease mechanism.
        provisional_genetics_enabled: When True, allow Layer 3 access even without
            confirmed genetics, provided evidence_count >= provisional_genetics_min_evidence.
            Passed from config key ``provisional_genetics_enabled``.
        provisional_genetics_min_evidence: Minimum evidence count required to enter
            Layer 3 on a provisional profile. Default 500 (matches config default).

    Returns:
        The current ResearchLayer.

    Note:
        Drug design (Layer 4) always requires *confirmed* genetics — the provisional
        pathway never unlocks Layer 4 regardless of validated_targets.
    """
    if genetic_profile is None:
        if evidence_count < LAYER_1_THRESHOLD:
            return ResearchLayer.NORMAL_BIOLOGY

        # Soft-gate: allow Layer 3 on provisional inference when config permits
        if (
            provisional_genetics_enabled
            and evidence_count >= provisional_genetics_min_evidence
        ):
            return ResearchLayer.ERIK_SPECIFIC

        return ResearchLayer.ALS_MECHANISMS

    # Confirmed genetics received — full Layer 3/4 progression available
    if validated_targets >= LAYER_3_TARGET_THRESHOLD:
        return ResearchLayer.DRUG_DESIGN

    return ResearchLayer.ERIK_SPECIFIC


# -----------------------------------------------------------------------
# Layer-specific query templates
# -----------------------------------------------------------------------

LAYER_QUERIES: dict[ResearchLayer, list[str]] = {
    ResearchLayer.NORMAL_BIOLOGY: [
        "motor neuron survival signaling pathway BDNF GDNF",
        "neuromuscular junction formation maintenance agrin LRP4 MuSK",
        "proteostasis chaperone system neuron protein quality control",
        "axonal transport dynein kinesin motor neuron",
        "glutamate receptor signaling motor neuron AMPA NMDA homeostasis",
        "mitochondrial function electron transport chain neuron ATP",
        "RNA processing splicing regulation motor neuron TDP-43 FUS normal",
        "superoxide dismutase SOD1 normal function reactive oxygen species",
        "autophagy lysosome pathway neuron protein clearance",
        "neurotrophic factor signaling motor neuron IGF-1 VEGF CNTF",
        "upper motor neuron corticospinal tract normal physiology",
        "Schwann cell myelination peripheral nerve motor function",
    ],
    ResearchLayer.ALS_MECHANISMS: [
        "ALS TDP-43 aggregation pathological cascade motor neuron",
        "ALS SOD1 misfolding toxic gain of function mechanism",
        "ALS C9orf72 repeat expansion dipeptide repeat RNA foci",
        "ALS FUS mutation RNA processing defect mechanism",
        "ALS glutamate excitotoxicity EAAT2 motor neuron death",
        "ALS neuroinflammation microglia astrocyte activation",
        "ALS mitochondrial dysfunction oxidative stress",
        "ALS axonal transport disruption neurofilament accumulation",
        "ALS neuromuscular junction denervation dying back",
        "ALS protein aggregation stress granule pathology",
        "ALS cortical hyperexcitability upper motor neuron",
        "ALS cryptic exon splicing STMN2 UNC13A loss",
    ],
    # Layer 3 queries are generated dynamically from the genetic profile
    ResearchLayer.ERIK_SPECIFIC: [],
    # Layer 4 queries are generated dynamically from validated targets
    ResearchLayer.DRUG_DESIGN: [],
}


def get_layer_queries(
    layer: ResearchLayer,
    genetic_profile: Optional[dict[str, Any]] = None,
    validated_targets: Optional[list[str]] = None,
) -> list[str]:
    """Get query templates for the current research layer.

    For Layers 1-2, returns static query banks.
    For Layer 3, generates queries from the genetic profile (confirmed or provisional).
    For Layer 4, generates queries from validated causal targets.

    When ``genetic_profile`` is None and the layer is ERIK_SPECIFIC (provisional
    pathway), a provisional profile is inferred from default clinical parameters
    (matching Erik's known clinical features) and used to generate queries.
    """
    if layer in (ResearchLayer.NORMAL_BIOLOGY, ResearchLayer.ALS_MECHANISMS):
        return LAYER_QUERIES[layer]

    if layer == ResearchLayer.ERIK_SPECIFIC:
        profile = genetic_profile
        if profile is None:
            # Provisional pathway: infer from Erik's known clinical features
            from research.provisional_genetics import infer_provisional_profile
            profile = infer_provisional_profile()

        gene = profile.get("gene", "")
        variant = profile.get("variant", "")
        subtype = profile.get("subtype", "")
        provisional = profile.get("provisional", False)
        qualifier = "provisional " if provisional else ""
        return [
            f"{gene} ALS mutation mechanism motor neuron",
            f"{gene} {variant} functional impact pathogenesis",
            f"{subtype} ALS subtype disease progression",
            f"{gene} ALS causal pathway downstream effects",
            f"{gene} protein structure function loss mutation",
            f"{gene} ALS patient genotype phenotype correlation",
            f"{gene} motor neuron selective vulnerability mechanism",
            f"{gene} ALS biomarker disease monitoring",
            f"{subtype} ALS prognosis trajectory prediction",
            f"{gene} ALS therapeutic target druggable site {qualifier}profile",
        ]

    if layer == ResearchLayer.DRUG_DESIGN and validated_targets:
        queries = []
        for target in validated_targets:
            queries.extend([
                f"{target} drug binding affinity inhibitor",
                f"{target} small molecule modulator ALS",
                f"{target} structure activity relationship drug design",
                f"{target} ADMET pharmacokinetics blood brain barrier",
            ])
        return queries if queries else [
            "ALS drug target structure based design",
            "ALS small molecule therapeutic development",
        ]

    # Fallback
    return LAYER_QUERIES[ResearchLayer.ALS_MECHANISMS]
