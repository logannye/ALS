"""Canonical ALS drug target definitions for the Erik ALS causal research engine.

Each target is a dict with the following required fields:
    name              : str  — canonical display name
    gene              : str  — HGNC gene symbol
    uniprot_id        : str  — UniProt accession
    description       : str  — one-line biological role
    subtypes          : list[str]  — SubtypeClass values this target is relevant to
    protocol_layers   : list[str]  — ProtocolLayer values for therapeutic positioning
    druggable         : bool — whether small-molecule / biologic drugging is feasible
    druggability_notes: str  — rationale for druggability assessment
"""
from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Canonical target definitions
# ---------------------------------------------------------------------------

ALS_TARGETS: dict[str, dict] = {
    "TDP-43": {
        "name": "TDP-43",
        "gene": "TARDBP",
        "uniprot_id": "Q13148",
        "description": (
            "TAR DNA-binding protein 43; nuclear RNA-binding protein whose cytoplasmic "
            "mislocalization and aggregation is the hallmark pathology in ~97% of ALS."
        ),
        "subtypes": ["sporadic_tdp43", "tardbp", "c9orf72"],
        "protocol_layers": ["root_cause_suppression", "pathology_reversal"],
        "druggable": True,
        "druggability_notes": (
            "Multiple drugging strategies: ASOs to reduce expression, small molecules "
            "to inhibit aggregation, nuclear import enhancers, cryptic exon splicing rescue."
        ),
    },
    "SOD1": {
        "name": "SOD1",
        "gene": "SOD1",
        "uniprot_id": "P00441",
        "description": (
            "Copper-zinc superoxide dismutase 1; gain-of-toxic-function mutations cause "
            "~20% of familial ALS. ASO tofersen (approved) suppresses SOD1 protein."
        ),
        "subtypes": ["sod1"],
        "protocol_layers": ["root_cause_suppression"],
        "druggable": True,
        "druggability_notes": (
            "Validated ASO target (tofersen FDA-approved 2023). Small-molecule aggregation "
            "inhibitors (e.g. phenyl-butyl nitrone derivatives) also in preclinical study."
        ),
    },
    "FUS": {
        "name": "FUS",
        "gene": "FUS",
        "uniprot_id": "P35637",
        "description": (
            "Fused in sarcoma RNA-binding protein; mutations cause ~5% of familial ALS "
            "with cytoplasmic mislocalization and arginine methylation defects."
        ),
        "subtypes": ["fus"],
        "protocol_layers": ["root_cause_suppression"],
        "druggable": True,
        "druggability_notes": (
            "ASO knockdown strategy validated in mouse models. PRMT1 inhibitors targeting "
            "arginine methylation being explored."
        ),
    },
    "C9orf72": {
        "name": "C9orf72",
        "gene": "C9orf72",
        "uniprot_id": "Q96LT7",
        "description": (
            "C9orf72 protein; GGGGCC hexanucleotide repeat expansion causes ~40% of "
            "familial and ~8% of sporadic ALS via RNA foci and dipeptide repeat toxicity."
        ),
        "subtypes": ["c9orf72"],
        "protocol_layers": ["root_cause_suppression"],
        "druggable": True,
        "druggability_notes": (
            "ASOs targeting repeat RNA in clinical trials. Small molecules disrupting "
            "G-quadruplex RNA structures under development."
        ),
    },
    "STMN2": {
        "name": "STMN2",
        "gene": "STMN2",
        "uniprot_id": "Q93045",
        "description": (
            "Stathmin-2 (SCG10); axonal regeneration factor whose expression depends on "
            "nuclear TDP-43. Loss of STMN2 impairs motor neuron axon regeneration."
        ),
        "subtypes": ["sporadic_tdp43", "tardbp", "c9orf72"],
        "protocol_layers": ["pathology_reversal"],
        "druggable": True,
        "druggability_notes": (
            "ASO-based cryptic exon skipping to restore STMN2 expression (clinical "
            "candidate). Downstream of TDP-43 loss — TDP-43 rescue also restores STMN2."
        ),
    },
    "UNC13A": {
        "name": "UNC13A",
        "gene": "UNC13A",
        "uniprot_id": "Q9UPW8",
        "description": (
            "Protein unc-13 homolog A; synaptic vesicle priming protein whose cryptic "
            "exon inclusion upon TDP-43 loss reduces neuromuscular junction function."
        ),
        "subtypes": ["sporadic_tdp43", "tardbp", "c9orf72"],
        "protocol_layers": ["pathology_reversal"],
        "druggable": True,
        "druggability_notes": (
            "ASO-based cryptic exon skipping strategy analogous to STMN2. UNC13A risk "
            "variant rs12608932 is the strongest GWAS hit in ALS, validating this target."
        ),
    },
    "SIGMAR1": {
        "name": "SIGMAR1",
        "gene": "SIGMAR1",
        "uniprot_id": "Q99720",
        "description": (
            "Sigma non-opioid intracellular receptor 1 (Sigma-1R); ER chaperone that "
            "regulates ER-mitochondria contact sites, UPR, and motor neuron ER stress."
        ),
        "subtypes": ["sporadic_tdp43", "sod1", "fus", "tardbp", "c9orf72", "glia_amplified", "mixed", "unresolved"],
        "protocol_layers": ["pathology_reversal"],
        "druggable": True,
        "druggability_notes": (
            "Established pharmacological target with agonists (fluvoxamine, PRE-084) "
            "shown to delay disease in SOD1 mice. Phase 2 trials ongoing."
        ),
    },
    "EAAT2": {
        "name": "EAAT2",
        "gene": "SLC1A2",
        "uniprot_id": "P43004",
        "description": (
            "Excitatory amino acid transporter 2; primary glutamate transporter in "
            "astrocytes. Loss leads to excitotoxic motor neuron death in ALS."
        ),
        "subtypes": ["sporadic_tdp43", "sod1", "fus", "tardbp", "c9orf72", "glia_amplified", "mixed", "unresolved"],
        "protocol_layers": ["circuit_stabilization"],
        "druggable": True,
        "druggability_notes": (
            "Riluzole approved for ALS partly through glutamate pathway modulation. "
            "LDN/OSU-0212320 and ceftriaxone shown to upregulate EAAT2 expression."
        ),
    },
    "BDNF": {
        "name": "BDNF",
        "gene": "BDNF",
        "uniprot_id": "P23560",
        "description": (
            "Brain-derived neurotrophic factor; critical motor neuron survival and "
            "synaptic plasticity factor. Declines in ALS motor cortex and spinal cord."
        ),
        "subtypes": ["sporadic_tdp43", "sod1", "fus", "tardbp", "c9orf72", "glia_amplified", "mixed", "unresolved"],
        "protocol_layers": ["regeneration_reinnervation"],
        "druggable": True,
        "druggability_notes": (
            "Direct BDNF delivery hampered by BBB. TrkB agonists, exercise-induced BDNF "
            "upregulation, and gene therapy vectors (AAV-BDNF) in preclinical development."
        ),
    },
    "GDNF": {
        "name": "GDNF",
        "gene": "GDNF",
        "uniprot_id": "P39905",
        "description": (
            "Glial cell line-derived neurotrophic factor; potent motor neuron survival "
            "factor acting via RET receptor. Promotes neuromuscular junction reinnervation."
        ),
        "subtypes": ["sporadic_tdp43", "sod1", "fus", "tardbp", "c9orf72", "glia_amplified", "mixed", "unresolved"],
        "protocol_layers": ["regeneration_reinnervation"],
        "druggable": True,
        "druggability_notes": (
            "Delivery via AAV gene therapy or encapsulated cell therapy being pursued. "
            "Intrathecal protein delivery in early clinical exploration."
        ),
    },
    "OPTN": {
        "name": "OPTN",
        "gene": "OPTN",
        "uniprot_id": "Q96CV9",
        "description": (
            "Optineurin; selective autophagy receptor and NF-kB signaling regulator. "
            "Loss-of-function mutations cause ALS with defective mitophagy."
        ),
        "subtypes": ["sporadic_tdp43", "mixed"],
        "protocol_layers": ["pathology_reversal"],
        "druggable": False,
        "druggability_notes": (
            "Protein-protein interaction hub; no approved small molecules. Gene "
            "replacement or autophagy-flux enhancers (e.g. rapamycin analogues) "
            "are indirect approaches."
        ),
    },
    "TBK1": {
        "name": "TBK1",
        "gene": "TBK1",
        "uniprot_id": "Q9UHD2",
        "description": (
            "TANK-binding kinase 1; phosphorylates OPTN and p62/SQSTM1 to promote "
            "selective autophagy and neuroinflammation control. Haploinsufficiency "
            "causes ALS."
        ),
        "subtypes": ["sporadic_tdp43", "mixed"],
        "protocol_layers": ["pathology_reversal", "circuit_stabilization"],
        "druggable": True,
        "druggability_notes": (
            "Kinase with established small-molecule inhibitors (e.g. BX795, MRT67307). "
            "Paradoxical — activation needed for autophagy but inhibition may reduce "
            "neuroinflammation; context-dependent dosing strategy required."
        ),
    },
    "NEK1": {
        "name": "NEK1",
        "gene": "NEK1",
        "uniprot_id": "Q96PY6",
        "description": (
            "Never-in-mitosis kinase 1; regulates DNA damage response, axonal "
            "polarity, and ciliogenesis. Second most common ALS GWAS risk gene."
        ),
        "subtypes": ["sporadic_tdp43", "mixed"],
        "protocol_layers": ["root_cause_suppression"],
        "druggable": False,
        "druggability_notes": (
            "No selective NEK1 inhibitors in clinical use. Downstream pathway "
            "targets (HDAC6, DNA-PK) may be more tractable indirect approaches."
        ),
    },
    "Complement C5": {
        "name": "Complement C5",
        "gene": "C5",
        "uniprot_id": "P01031",
        "description": (
            "Complement component C5; cleavage releases C5a (pro-inflammatory) and "
            "C5b (membrane attack complex initiator). Complement over-activation drives "
            "neuromuscular junction destruction and microglial activation in ALS."
        ),
        "subtypes": ["sporadic_tdp43", "sod1", "c9orf72", "glia_amplified"],
        "protocol_layers": ["circuit_stabilization"],
        "druggable": True,
        "druggability_notes": (
            "Eculizumab (anti-C5 mAb) is approved for other complement disorders. "
            "Phase 2 ALS trial initiated 2023. Ravulizumab (long-acting anti-C5) also "
            "a candidate."
        ),
    },
    "CSF1R": {
        "name": "CSF1R",
        "gene": "CSF1R",
        "uniprot_id": "P07333",
        "description": (
            "Colony stimulating factor 1 receptor; master regulator of microglial "
            "proliferation and survival. Inhibition reprograms neuroinflammatory microglia "
            "in ALS models."
        ),
        "subtypes": ["sporadic_tdp43", "sod1", "c9orf72", "glia_amplified"],
        "protocol_layers": ["circuit_stabilization"],
        "druggable": True,
        "druggability_notes": (
            "Multiple approved/clinical CSF1R inhibitors: pexidartinib (FDA-approved), "
            "PLX5622, and others. Microglial depletion-replenishment strategy in SOD1 "
            "mice extended survival."
        ),
    },
    "mTOR": {
        "name": "mTOR",
        "gene": "MTOR",
        "uniprot_id": "P42345",
        "description": (
            "Mechanistic target of rapamycin; central growth/autophagy regulatory kinase. "
            "mTORC1 hyperactivation suppresses autophagy, increasing TDP-43 aggregate "
            "burden and reducing proteostasis in ALS."
        ),
        "subtypes": ["sporadic_tdp43", "sod1", "fus", "tardbp", "c9orf72", "glia_amplified", "mixed", "unresolved"],
        "protocol_layers": ["pathology_reversal"],
        "druggable": True,
        "druggability_notes": (
            "Rapamycin and rapalogs (everolimus, temsirolimus) are approved drugs. "
            "Intermittent low-dose rapamycin delays disease in SOD1 mice. mTORC1/2 "
            "dual inhibitors offer broader autophagy induction."
        ),
    },
}


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_target(name: str) -> Optional[dict]:
    """Return the target dict for the given canonical name, or None if not found."""
    return ALS_TARGETS.get(name)


def get_targets_for_subtype(subtype: str) -> list[dict]:
    """Return all targets relevant to the given ALS subtype string."""
    return [t for t in ALS_TARGETS.values() if subtype in t["subtypes"]]


def get_targets_for_protocol_layer(layer: str) -> list[dict]:
    """Return all targets positioned in the given protocol layer string."""
    return [t for t in ALS_TARGETS.values() if layer in t["protocol_layers"]]
