"""Research intelligence — the reasoning brain of Erik's research loop.

This module provides the high-level cognitive functions that decide
WHAT to investigate and WHY, based on deep analysis of the current
protocol, causal chains, evidence gaps, and Erik's specific situation.

Three core capabilities:

1. **Protocol gap analysis** — identifies the weakest parts of the
   current protocol (weakest evidence, shallowest chains, contested
   claims, missing measurements) and prioritizes them.

2. **Targeted hypothesis generation** — produces specific, testable,
   high-quality mechanistic hypotheses that would strengthen the
   weakest protocol links if confirmed.

3. **Intelligent search planning** — translates hypotheses into
   precise PubMed queries, target-specific database queries, and
   pathway lookups designed to find exactly the evidence needed.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from research.state import ResearchState


# ---------------------------------------------------------------------------
# Protocol gap analysis
# ---------------------------------------------------------------------------

def analyze_protocol_gaps(
    state: ResearchState,
    evidence_store: Any,
) -> list[dict]:
    """Identify the most important gaps in the current protocol.

    Returns a ranked list of gaps, each with:
    - gap_type: 'weak_evidence' | 'shallow_chain' | 'missing_data' | 'sparse_layer' | 'unvalidated_safety'
    - description: human-readable explanation
    - intervention_id: affected intervention (if applicable)
    - layer: protocol layer
    - priority: float 0-1 (higher = more urgent)
    - suggested_action: what to do about it
    - search_queries: specific PubMed queries to find evidence
    """
    gaps: list[dict] = []

    # 1. Sparse evidence layers — which protocol layers have the least evidence?
    layer_counts = {}
    for layer in ["root_cause_suppression", "pathology_reversal", "circuit_stabilization",
                  "regeneration_reinnervation", "adaptive_maintenance"]:
        items = evidence_store.query_by_protocol_layer(layer)
        layer_counts[layer] = len(items)

    if layer_counts:
        min_layer = min(layer_counts, key=layer_counts.get)
        min_count = layer_counts[min_layer]
        if min_count < 30:
            gaps.append({
                "gap_type": "sparse_layer",
                "description": f"Layer '{min_layer}' has only {min_count} evidence items — weakest evidence base",
                "layer": min_layer,
                "priority": 0.9,
                "suggested_action": "search_pubmed",
                "search_queries": _generate_layer_queries(min_layer),
            })

    # 2. Shallow causal chains — which interventions lack deep mechanism understanding?
    for int_id, depth in state.causal_chains.items():
        if depth < 3:
            int_name = int_id.replace("int:", "")
            gaps.append({
                "gap_type": "shallow_chain",
                "description": f"Intervention '{int_name}' has causal chain depth {depth}/5 — mechanism poorly understood",
                "intervention_id": int_id,
                "priority": 0.85 - depth * 0.1,
                "suggested_action": "deepen_causal_chain",
                "search_queries": [
                    f"ALS {int_name} mechanism of action motor neuron",
                    f"{int_name} neuroprotection pathway signaling",
                ],
            })

    # 3. Missing critical measurements for Erik
    missing_priority = {
        "genetic_testing": 0.95,  # Highest — changes entire Layer A
        "csf_biomarkers": 0.6,
        "cryptic_exon_splicing_assay": 0.5,
        "tdp43_in_vivo_measurement": 0.4,
        "cortical_excitability_tms": 0.3,
    }
    for measurement in state.missing_measurements:
        priority = missing_priority.get(measurement, 0.3)
        gaps.append({
            "gap_type": "missing_data",
            "description": f"Missing measurement: {measurement}",
            "priority": priority,
            "suggested_action": "generate_hypothesis",
            "search_queries": [f"ALS {measurement.replace('_', ' ')} clinical utility prognosis"],
        })

    # 4. Drug interaction safety — check if protocol combinations have been validated
    interventions = list(state.causal_chains.keys())
    if len(interventions) >= 2:
        gaps.append({
            "gap_type": "unvalidated_safety",
            "description": f"Drug combination safety for {len(interventions)} protocol interventions needs PharmGKB validation",
            "priority": 0.7,
            "suggested_action": "check_pharmacogenomics",
            "search_queries": [f"ALS drug combination interaction {interventions[0].replace('int:', '')} {interventions[1].replace('int:', '')}"],
        })

    # Sort by priority descending
    gaps.sort(key=lambda g: g.get("priority", 0), reverse=True)
    return gaps


# ---------------------------------------------------------------------------
# Targeted hypothesis generation
# ---------------------------------------------------------------------------

def build_hypothesis_prompt(
    gap: dict,
    state: ResearchState,
    evidence_items: list[dict],
) -> str:
    """Build a detailed, context-rich prompt for hypothesis generation
    that targets a specific protocol gap.

    Unlike the generic "generate a hypothesis" prompt, this provides:
    - The specific gap being addressed
    - Erik's clinical context
    - Relevant evidence items
    - The exact type of hypothesis needed
    - Clear output schema
    """
    gap_type = gap.get("gap_type", "")
    description = gap.get("description", "")
    intervention_id = gap.get("intervention_id", "")
    int_name = intervention_id.replace("int:", "") if intervention_id else ""

    # Erik's clinical context (always included)
    erik_context = (
        "PATIENT: Erik Draper, 67M, limb-onset ALS diagnosed March 2026.\n"
        "ALSFRS-R: 43/48 (Bulbar 12, Fine Motor 11, Gross Motor 8, Respiratory 12)\n"
        "Decline rate: -0.39 points/month. NfL: 5.82 pg/mL (elevated).\n"
        "FVC: 100% predicted. Genetics: PENDING (Invitae panel).\n"
        "Current treatment: Riluzole. Classification: Definite ALS (Gold Coast).\n"
        "Working subtype hypothesis: sporadic TDP-43 (posterior 0.65).\n"
    )

    if gap_type == "shallow_chain":
        return (
            f"TASK: Generate a specific, testable mechanistic hypothesis about how "
            f"the intervention '{int_name}' produces neuroprotective effects in ALS.\n\n"
            f"{erik_context}\n"
            f"CURRENT GAP: {description}\n\n"
            f"The causal chain from '{int_name}' to motor neuron survival is incomplete. "
            f"We need to understand the intermediate biological steps:\n"
            f"  {int_name} → [target binding] → [pathway activation] → [cellular effect] → "
            f"[neuroprotection] → motor neuron survival\n\n"
            f"Generate a hypothesis about ONE specific intermediate step that is currently "
            f"missing or weakly supported. The hypothesis must be testable by searching "
            f"PubMed or querying a protein interaction database.\n\n"
            f"EVIDENCE ITEMS:\n{{evidence_items_json}}\n\n"
            f"Return JSON:\n"
            f'{{"statement": "<specific testable claim about {int_name} mechanism>",\n'
            f' "mechanism_step": "<which step in the chain this addresses>",\n'
            f' "search_terms": ["<3-5 specific PubMed search terms to validate this>"],\n'
            f' "target_genes": ["<gene symbols involved, for STRING/Reactome queries>"],\n'
            f' "if_confirmed_impact": "<how this changes Erik\'s protocol>",\n'
            f' "cited_evidence": ["<evidence IDs supporting this hypothesis>"]}}'
        )

    elif gap_type == "sparse_layer":
        layer = gap.get("layer", "root_cause_suppression")
        return (
            f"TASK: Generate a hypothesis about a potentially overlooked therapeutic "
            f"mechanism for the '{layer}' protocol layer in ALS.\n\n"
            f"{erik_context}\n"
            f"CURRENT GAP: {description}\n\n"
            f"This protocol layer has the least evidence support. We need hypotheses "
            f"about mechanisms or interventions that may have been missed.\n\n"
            f"Consider: Are there approved drugs for other conditions that could be "
            f"repurposed for this ALS mechanism? Are there preclinical compounds "
            f"showing promise? Are there combination strategies that haven't been explored?\n\n"
            f"EVIDENCE ITEMS:\n{{evidence_items_json}}\n\n"
            f"Return JSON:\n"
            f'{{"statement": "<specific testable claim about a {layer} mechanism>",\n'
            f' "mechanism_step": "<biological pathway or target involved>",\n'
            f' "search_terms": ["<3-5 specific PubMed search terms>"],\n'
            f' "target_genes": ["<gene symbols for database queries>"],\n'
            f' "if_confirmed_impact": "<how this changes the protocol>",\n'
            f' "cited_evidence": ["<evidence IDs>"]}}'
        )

    elif gap_type == "missing_data":
        return (
            f"TASK: Generate a hypothesis about what Erik's missing measurement "
            f"would reveal and how it would change his treatment protocol.\n\n"
            f"{erik_context}\n"
            f"MISSING MEASUREMENT: {description}\n\n"
            f"Given Erik's clinical presentation (limb-onset, age 67, NfL elevated, "
            f"widespread EMG denervation, mother with Alzheimer's), what would you "
            f"predict this measurement would show? How would each possible result "
            f"change the subtype posterior and intervention selection?\n\n"
            f"EVIDENCE ITEMS:\n{{evidence_items_json}}\n\n"
            f"Return JSON:\n"
            f'{{"statement": "<specific prediction about what the measurement would show for Erik>",\n'
            f' "predicted_result": "<most likely result given Erik\'s presentation>",\n'
            f' "search_terms": ["<PubMed terms to find similar patient outcomes>"],\n'
            f' "target_genes": [],\n'
            f' "if_confirmed_impact": "<how this result changes the protocol>",\n'
            f' "cited_evidence": ["<evidence IDs>"]}}'
        )

    # Default generic
    return (
        f"TASK: Generate a testable hypothesis about ALS biology relevant to "
        f"Erik Draper's treatment.\n\n"
        f"{erik_context}\n"
        f"CURRENT GAP: {description}\n\n"
        f"EVIDENCE ITEMS:\n{{evidence_items_json}}\n\n"
        f"Return JSON:\n"
        f'{{"statement": "<specific testable claim>",\n'
        f' "search_terms": ["<PubMed search terms>"],\n'
        f' "target_genes": [],\n'
        f' "if_confirmed_impact": "<how this changes the protocol>",\n'
        f' "cited_evidence": ["<evidence IDs>"]}}'
    )


# ---------------------------------------------------------------------------
# Intelligent search planning
# ---------------------------------------------------------------------------

def plan_search_from_hypothesis(
    hypothesis_result: dict,
) -> list[dict]:
    """Convert a generated hypothesis into concrete search actions.

    Uses the hypothesis's search_terms and target_genes to build
    a targeted research plan — not generic queries.
    """
    actions: list[dict] = []

    # PubMed searches from the hypothesis's own search terms
    search_terms = hypothesis_result.get("search_terms", [])
    for term in search_terms[:3]:
        actions.append({
            "action": "search_pubmed",
            "query": term,
            "max_results": 15,
            "source": "hypothesis_targeted",
        })

    # STRING/Reactome queries for target genes
    target_genes = hypothesis_result.get("target_genes", [])
    for gene in target_genes[:2]:
        actions.append({
            "action": "query_ppi_network",
            "gene_symbol": gene,
            "source": "hypothesis_targeted",
        })
        actions.append({
            "action": "query_pathways",
            "target_name": gene,
            "source": "hypothesis_targeted",
        })

    return actions


def build_validation_query(hypothesis_statement: str) -> str:
    """Build a precise PubMed query from a hypothesis statement.

    Extracts key biomedical terms and constructs a targeted query
    instead of searching for a hash.
    """
    # Extract meaningful terms (>4 chars, not common words)
    stop_words = {
        "that", "this", "with", "from", "have", "been", "will", "would",
        "could", "should", "their", "there", "which", "about", "after",
        "before", "between", "through", "during", "because", "specific",
        "specifically", "likely", "suggest", "suggests", "hypothesis",
        "hypothesize", "propose", "proposed", "evidence", "based",
        "mechanism", "mechanistic", "pathway", "pathways", "may",
    }

    words = hypothesis_statement.replace(",", " ").replace(".", " ").replace("(", " ").replace(")", " ").split()
    terms = [w for w in words if len(w) > 4 and w.lower() not in stop_words]

    # Keep biomedical terms (capitalized, contains numbers, or known ALS terms)
    als_terms = {"ALS", "TDP-43", "SOD1", "FUS", "C9orf72", "TARDBP", "STMN2", "UNC13A",
                 "sigma-1R", "SIGMAR1", "EAAT2", "riluzole", "pridopidine", "tofersen",
                 "masitinib", "rapamycin", "ibudilast", "edaravone", "NfL", "ALSFRS",
                 "motor", "neuron", "neuroprotection", "proteostasis", "aggregation",
                 "splicing", "cryptic", "exon", "autophagy", "excitotoxicity",
                 "neuroinflammation", "denervation", "reinnervation"}

    priority_terms = [w for w in terms if w in als_terms or w[0].isupper() or any(c.isdigit() for c in w)]
    other_terms = [w for w in terms if w not in priority_terms]

    # Build query: prioritize biomedical terms, cap at 8 terms
    query_terms = priority_terms[:5] + other_terms[:3]
    if not query_terms:
        query_terms = terms[:5]

    return "ALS " + " ".join(query_terms[:8])


# ---------------------------------------------------------------------------
# Layer-specific query generation
# ---------------------------------------------------------------------------

def _generate_layer_queries(layer: str) -> list[str]:
    """Generate targeted PubMed queries for a specific protocol layer."""
    layer_queries = {
        "root_cause_suppression": [
            "ALS TDP-43 nuclear import rescue therapeutic 2024 2025",
            "ALS gene therapy intrabody antisense oligonucleotide clinical trial",
            "ALS C9orf72 repeat expansion treatment ASO 2025",
            "ALS SOD1 silencing tofersen long-term outcome",
        ],
        "pathology_reversal": [
            "ALS TDP-43 aggregation clearance autophagy therapeutic",
            "ALS sigma-1R agonist ER stress proteostasis",
            "ALS cryptic exon splicing STMN2 UNC13A rescue",
            "ALS protein misfolding chaperone therapeutic strategy",
        ],
        "circuit_stabilization": [
            "ALS neuroprotection glutamate excitotoxicity combination therapy",
            "ALS neuroinflammation microglia CSF1R inhibitor clinical",
            "ALS synaptic protection NMJ stabilization compound",
            "ALS riluzole combination augmentation clinical trial",
        ],
        "regeneration_reinnervation": [
            "ALS motor neuron regeneration neurotrophic factor delivery",
            "ALS NMJ reinnervation terminal sprouting therapeutic",
            "ALS stem cell transplant motor neuron replacement 2025",
            "ALS axonal growth factor BDNF GDNF gene therapy",
        ],
        "adaptive_maintenance": [
            "ALS biomarker treatment response monitoring NfL pNfH",
            "ALS ALSFRS-R trajectory prediction machine learning",
            "ALS respiratory decline FVC prediction intervention timing",
            "ALS multidisciplinary care outcome survival benefit",
        ],
    }
    return layer_queries.get(layer, [f"ALS {layer.replace('_', ' ')} treatment 2025"])
