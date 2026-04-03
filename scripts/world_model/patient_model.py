"""Patient-specific molecular model for Erik Draper.

When genetic test results arrive, this module:
1. Ingests variant data as structured EvidenceItems
2. Collapses the 8-way subtype posterior to near-certainty
3. Maps Erik's specific variants to the causal graph
4. Re-ranks drug design targets by relevance to HIS specific biology
5. Identifies which pathways are disrupted in his motor neurons

This is Stage 3 of the grand pipeline: Precision Mapping.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from db.pool import get_connection
from evidence.evidence_store import EvidenceStore
from ontology.base import BaseEnvelope
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem

logger = logging.getLogger(__name__)

# ALS subtype mapping: gene → subtype
GENE_TO_SUBTYPE = {
    "SOD1": "sod1",
    "C9orf72": "c9orf72",
    "FUS": "fus",
    "TARDBP": "tardbp",
    "TBK1": "sporadic_tdp43",
    "OPTN": "sporadic_tdp43",
    "UBQLN2": "sporadic_tdp43",
    "VCP": "sporadic_tdp43",
    "SQSTM1": "sporadic_tdp43",
    "ANXA11": "sporadic_tdp43",
    "NEK1": "sporadic_tdp43",
    "KIF5A": "sporadic_tdp43",
}

# Subtype → relevant drug design targets
SUBTYPE_TO_TARGETS = {
    "sod1": ["ddt:sod1_aggregation", "ddt:eaat2_modulation", "ddt:csf1r_kinase"],
    "c9orf72": ["ddt:tdp43_rrm", "ddt:tdp43_ctd", "ddt:unc13a_cryptic_splice", "ddt:stmn2_cryptic_splice"],
    "fus": ["ddt:sg_kinases", "ddt:eaat2_modulation"],
    "tardbp": ["ddt:tdp43_rrm", "ddt:tdp43_ctd", "ddt:unc13a_cryptic_splice", "ddt:stmn2_cryptic_splice"],
    "sporadic_tdp43": ["ddt:tdp43_ctd", "ddt:unc13a_cryptic_splice", "ddt:stmn2_cryptic_splice", "ddt:sigma1r_pocket", "ddt:mtor_allosteric"],
    "glia_amplified": ["ddt:csf1r_kinase", "ddt:eaat2_modulation", "ddt:sigma1r_pocket"],
    "mixed": ["ddt:tdp43_ctd", "ddt:unc13a_cryptic_splice", "ddt:eaat2_modulation", "ddt:sigma1r_pocket"],
    "unresolved": ["ddt:sigma1r_pocket", "ddt:eaat2_modulation", "ddt:mtor_allosteric"],
}


def ingest_genetic_results(results: dict[str, Any]) -> list[str]:
    """Ingest genetic test results and create evidence items.

    Parameters
    ----------
    results:
        Dict with keys:
        - "variants": list of {"gene": str, "variant": str, "classification": str, "zygosity": str}
        - "panel": str (e.g. "Invitae ALS Panel")
        - "date": str (ISO date)
        - "negative_genes": list of str (genes tested negative)

    Returns
    -------
    List of evidence item IDs created.
    """
    store = EvidenceStore()
    created_ids: list[str] = []

    variants = results.get("variants", [])
    panel = results.get("panel", "genetic_panel")
    date = results.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    negative_genes = results.get("negative_genes", [])

    # Create evidence items for each variant found
    for variant in variants:
        gene = variant.get("gene", "unknown")
        var_name = variant.get("variant", "unknown")
        classification = variant.get("classification", "VUS")  # Pathogenic, Likely Pathogenic, VUS, Benign
        zygosity = variant.get("zygosity", "heterozygous")

        # Determine evidence strength from classification
        if classification.lower() in ("pathogenic", "likely pathogenic"):
            strength = "strong"
            direction = "supports"
        elif classification.lower() == "vus":
            strength = "emerging"
            direction = "mixed"
        else:
            strength = "preclinical"
            direction = "refutes"

        evi_id = f"evi:genetic_{gene.lower()}_{var_name.replace(' ', '_').lower()}"
        subtype = GENE_TO_SUBTYPE.get(gene, "unresolved")

        fact = BaseEnvelope(
            id=evi_id,
            type="EvidenceItem",
            status="active",
            body={
                "claim": f"Genetic testing ({panel}): {gene} {var_name} ({classification}, {zygosity})",
                "direction": direction,
                "strength": strength,
                "source": "genetic_testing",
                "experiment_type": "genetic_result",
                "pch_layer": 3,  # Genetic data is causal/counterfactual
                "protocol_layer": "root_cause_suppression",
                "gene": gene,
                "variant": var_name,
                "classification": classification,
                "zygosity": zygosity,
                "inferred_subtype": subtype,
                "panel": panel,
                "test_date": date,
            },
        )
        store.upsert_object(fact)
        created_ids.append(evi_id)
        logger.info("Ingested genetic result: %s %s (%s)", gene, var_name, classification)

    # Create evidence items for genes tested negative (important for narrowing subtype)
    for gene in negative_genes:
        evi_id = f"evi:genetic_{gene.lower()}_negative"
        fact = BaseEnvelope(
            id=evi_id,
            type="EvidenceItem",
            status="active",
            body={
                "claim": f"Genetic testing ({panel}): {gene} — no pathogenic variants detected",
                "direction": "refutes",
                "strength": "strong",
                "source": "genetic_testing",
                "experiment_type": "genetic_result",
                "pch_layer": 3,
                "protocol_layer": "root_cause_suppression",
                "gene": gene,
                "variant": "none",
                "classification": "negative",
                "panel": panel,
                "test_date": date,
            },
        )
        store.upsert_object(fact)
        created_ids.append(evi_id)

    # Update the genetics_received causal gap
    try:
        from research.causal_gaps import resolve_gap
        resolve_gap("gap:erik_genetic_subtype", created_ids)
        logger.info("Resolved causal gap: erik_genetic_subtype")
    except Exception:
        pass

    return created_ids


def determine_subtype(variants: list[dict]) -> tuple[str, float]:
    """Determine ALS subtype from genetic variants.

    Returns (subtype, confidence) tuple.
    """
    pathogenic = [v for v in variants if v.get("classification", "").lower() in ("pathogenic", "likely pathogenic")]

    if not pathogenic:
        # No pathogenic variants → likely sporadic TDP-43 (most common)
        return "sporadic_tdp43", 0.65

    # Map the most significant pathogenic variant to subtype
    for var in pathogenic:
        gene = var.get("gene", "")
        if gene in GENE_TO_SUBTYPE:
            return GENE_TO_SUBTYPE[gene], 0.95

    return "unresolved", 0.3


def get_priority_targets(subtype: str) -> list[str]:
    """Return drug design target IDs ranked by relevance for a given subtype."""
    return SUBTYPE_TO_TARGETS.get(subtype, SUBTYPE_TO_TARGETS["unresolved"])


def build_molecular_profile(
    subtype: str,
    variants: list[dict],
) -> dict[str, Any]:
    """Build a molecular-level profile of Erik's disease state.

    This is the computational twin — maps his specific genetics + biomarkers
    to a molecular-level representation of which pathways are disrupted.
    """
    profile: dict[str, Any] = {
        "subtype": subtype,
        "pathogenic_variants": [v for v in variants if v.get("classification", "").lower() in ("pathogenic", "likely pathogenic")],
        "disrupted_pathways": [],
        "priority_targets": get_priority_targets(subtype),
        "predicted_protein_effects": [],
    }

    # Map subtype → disrupted pathways
    pathway_map = {
        "sod1": [
            "SOD1 toxic gain-of-function → protein aggregation",
            "Oxidative stress from misfolded SOD1",
            "Proteasome overload from aggregated SOD1",
        ],
        "c9orf72": [
            "GGGGCC repeat expansion → RNA foci formation",
            "Dipeptide repeat protein (DPR) toxicity",
            "TDP-43 nuclear depletion (downstream)",
            "UNC13A/STMN2 cryptic exon inclusion (downstream)",
        ],
        "fus": [
            "FUS nuclear depletion → cytoplasmic aggregation",
            "RNA processing disruption",
            "Stress granule dynamics impairment",
        ],
        "tardbp": [
            "TDP-43 mutation → enhanced aggregation propensity",
            "Nuclear depletion → splicing dysregulation",
            "UNC13A/STMN2 cryptic exon inclusion",
        ],
        "sporadic_tdp43": [
            "TDP-43 cytoplasmic mislocalization (unknown trigger)",
            "Nuclear depletion → UNC13A/STMN2 cryptic exon inclusion",
            "Progressive motor neuron vulnerability to glutamate excitotoxicity",
            "Neuroinflammation amplification via microglia/astrocyte activation",
        ],
    }
    profile["disrupted_pathways"] = pathway_map.get(subtype, pathway_map["sporadic_tdp43"])

    # Predict protein-level effects from variants
    for var in profile["pathogenic_variants"]:
        gene = var.get("gene", "")
        variant = var.get("variant", "")
        if gene == "SOD1":
            profile["predicted_protein_effects"].append(
                f"SOD1 {variant}: destabilized dimer interface, enhanced aggregation propensity"
            )
        elif gene == "TARDBP":
            profile["predicted_protein_effects"].append(
                f"TDP-43 {variant}: altered C-terminal domain phase transition, increased fibril formation"
            )
        elif gene == "FUS":
            profile["predicted_protein_effects"].append(
                f"FUS {variant}: impaired nuclear localization, cytoplasmic aggregation"
            )

    return profile
